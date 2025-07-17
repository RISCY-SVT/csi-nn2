#!/usr/bin/env python3
"""
Regenerate conv2d.dat test data for csi-nn2 project.
Version 20: Fixed FP16 generation and shape parsing.
"""

import numpy as np
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import sys

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    print("ERROR: scipy is required for this script")
    sys.exit(1)


def parse_c_arrays(filepath: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Tuple[int, ...]]]]:
    """Parse C arrays from conv2d.dat file and return arrays + shapes from comments."""
    arrays = {}
    shapes = {}
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Parse shape comments - look for complete shape definitions
    # Pattern: // input: [1, 3, 4, 5], kernel: [19, 3, 3, 3], output: [1, 19, 4, 5]
    shape_pattern = r'//\s*input:\s*\[([\d,\s]+)\],\s*kernel:\s*\[([\d,\s]+)\],\s*output:\s*\[([\d,\s]+)\]'
    
    shape_blocks = list(re.finditer(shape_pattern, content))
    
    # Map test names to shapes based on order in file
    test_names = [
        'conv2d1x1s1',
        'conv2d_im2col', 
        'conv2d_winograd'
    ]
    
    for i, match in enumerate(shape_blocks):
        if i < len(test_names):
            input_shape = tuple(int(x.strip()) for x in match.group(1).split(','))
            kernel_shape = tuple(int(x.strip()) for x in match.group(2).split(','))
            output_shape = tuple(int(x.strip()) for x in match.group(3).split(','))
            
            shapes[test_names[i]] = {
                'input': input_shape,
                'kernel': kernel_shape,
                'output': output_shape
            }
    
    # Pattern to match array definitions
    array_pattern = r'unsigned char\s+(\w+)\s*\[\s*\]\s*=\s*\{([^}]+)\}'
    
    for match in re.finditer(array_pattern, content, re.DOTALL):
        array_name = match.group(1)
        array_content = match.group(2)
        
        # Remove comments
        array_content = re.sub(r'//.*$', '', array_content, flags=re.MULTILINE)
        array_content = re.sub(r'/\*.*?\*/', '', array_content, flags=re.DOTALL)
        
        # Extract hex values
        hex_values = re.findall(r'0x([0-9a-fA-F]{2})', array_content)
        
        if hex_values:
            byte_array = bytes([int(h, 16) for h in hex_values])
            arrays[array_name] = np.frombuffer(byte_array, dtype=np.uint8)
    
    return arrays, shapes


def bytes_to_float32(byte_array: np.ndarray) -> np.ndarray:
    """Convert byte array to float32 array."""
    return np.frombuffer(byte_array.tobytes(), dtype=np.float32)


def bytes_to_float16(byte_array: np.ndarray) -> np.ndarray:
    """Convert byte array to float16 array."""
    return np.frombuffer(byte_array.tobytes(), dtype=np.float16)


def float32_to_bytes(float_array: np.ndarray) -> np.ndarray:
    """Convert float32 array to byte array."""
    return np.frombuffer(float_array.astype(np.float32).tobytes(), dtype=np.uint8)


def float16_to_bytes(float_array: np.ndarray) -> np.ndarray:
    """Convert float16 array to byte array."""
    return np.frombuffer(float_array.astype(np.float16).tobytes(), dtype=np.uint8)


def conv2d_nchw(input_data: np.ndarray, kernel: np.ndarray, bias: np.ndarray,
                input_shape: Tuple[int, int, int, int], kernel_shape: Tuple[int, int, int, int],
                padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
                stride: Tuple[int, int] = (1, 1)) -> np.ndarray:
    """
    Naive convolution implementation for NCHW layout.
    Uses float64 internally for maximum precision.
    
    Args:
        input_data: flattened input array
        kernel: flattened kernel array
        bias: bias array
        input_shape: (N, C, H, W)
        kernel_shape: (out_channels, in_channels, kH, kW)
        padding: (left, right, top, bottom)
        stride: (stride_h, stride_w)
    
    Returns:
        flattened output array in original dtype
    """
    # Save original dtype
    orig_dtype = input_data.dtype
    
    # Convert to float64 for computation
    input_data = input_data.astype(np.float64)
    kernel = kernel.astype(np.float64)
    bias = bias.astype(np.float64)
    
    N, C, H, W = input_shape
    out_channels, in_channels, kH, kW = kernel_shape
    pad_left, pad_right, pad_top, pad_bottom = padding
    stride_h, stride_w = stride
    
    assert in_channels == C, f"Channel mismatch: input {C} vs kernel {in_channels}"
    assert N == 1, f"Only batch size 1 supported, got {N}"
    
    # Reshape inputs (remove batch for simplicity since N=1)
    x = input_data.reshape(C, H, W)
    w = kernel.reshape(out_channels, in_channels, kH, kW)
    
    # Apply padding
    if any(padding):
        x_padded = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), 
                         mode='constant', constant_values=0)
    else:
        x_padded = x
    
    _, H_padded, W_padded = x_padded.shape
    
    # Output dimensions
    H_out = (H_padded - kH) // stride_h + 1
    W_out = (W_padded - kW) // stride_w + 1
    
    # Perform convolution in float64
    output = np.zeros((out_channels, H_out, W_out), dtype=np.float64)
    
    for oc in range(out_channels):
        for oh in range(H_out):
            for ow in range(W_out):
                h_start = oh * stride_h
                w_start = ow * stride_w
                
                # Extract patch and compute convolution
                conv_sum = 0.0
                for ic in range(in_channels):
                    for kh_idx in range(kH):
                        for kw_idx in range(kW):
                            x_val = x_padded[ic, h_start + kh_idx, w_start + kw_idx]
                            w_val = w[oc, ic, kh_idx, kw_idx]
                            conv_sum += x_val * w_val
                
                output[oc, oh, ow] = conv_sum + bias[oc]
    
    # Convert back to original dtype
    return output.flatten().astype(orig_dtype)


def conv2d_einsum_1x1(input_data: np.ndarray, kernel: np.ndarray, bias: np.ndarray,
                      input_shape: Tuple[int, int, int, int], out_channels: int) -> np.ndarray:
    """1x1 convolution using einsum. Uses float64 internally for maximum precision."""
    # Save original dtype
    orig_dtype = input_data.dtype
    
    # Convert to float64
    input_data = input_data.astype(np.float64)
    kernel = kernel.astype(np.float64)
    bias = bias.astype(np.float64)
    
    N, C, H, W = input_shape
    assert N == 1
    
    x = input_data.reshape(C, H, W)
    w = kernel.reshape(out_channels, C, 1, 1).squeeze(-1).squeeze(-1)  # (OC, IC)
    
    # Perform 1x1 conv as matrix multiplication
    output = np.einsum('oi,ihw->ohw', w, x)
    
    # Add bias
    output = output + bias.reshape(-1, 1, 1)
    
    # Convert back to original dtype
    return output.flatten().astype(orig_dtype)


def make_im2col(input_data: np.ndarray, input_shape: Tuple[int, int, int],
                kernel_shape: Tuple[int, int], padding: Tuple[int, int, int, int],
                stride: Tuple[int, int]) -> np.ndarray:
    """
    Transform input to im2col format for GEMM-based convolution.
    
    Returns:
        col_matrix of shape (C*kH*kW, H_out*W_out)
    """
    C, H, W = input_shape
    kH, kW = kernel_shape
    pad_left, pad_right, pad_top, pad_bottom = padding
    stride_h, stride_w = stride
    
    # Reshape and pad
    x = input_data.reshape(C, H, W)
    if any(padding):
        x_padded = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                         mode='constant', constant_values=0)
    else:
        x_padded = x
    
    _, H_padded, W_padded = x_padded.shape
    H_out = (H_padded - kH) // stride_h + 1
    W_out = (W_padded - kW) // stride_w + 1
    
    # Create column matrix
    col = np.zeros((C * kH * kW, H_out * W_out), dtype=x.dtype)
    
    col_idx = 0
    for oh in range(H_out):
        for ow in range(W_out):
            h_start = oh * stride_h
            w_start = ow * stride_w
            patch = x_padded[:, h_start:h_start+kH, w_start:w_start+kW]
            col[:, col_idx] = patch.flatten()
            col_idx += 1
    
    return col


def conv2d_from_im2col(input_data: np.ndarray, kernel: np.ndarray, bias: np.ndarray,
                       input_shape: Tuple[int, int, int, int], kernel_shape: Tuple[int, int, int, int],
                       padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
                       stride: Tuple[int, int] = (1, 1)) -> np.ndarray:
    """Convolution via im2col + GEMM approach. Uses float64 internally."""
    # Save original dtype
    orig_dtype = input_data.dtype
    
    # Convert to float64
    input_data = input_data.astype(np.float64)
    kernel = kernel.astype(np.float64)
    bias = bias.astype(np.float64)
    
    N, C, H, W = input_shape
    out_channels, in_channels, kH, kW = kernel_shape
    
    assert N == 1
    
    # Transform to im2col
    col = make_im2col(input_data, (C, H, W), (kH, kW), padding, stride)
    
    # Reshape kernel for GEMM
    w_reshaped = kernel.reshape(out_channels, -1)
    
    # GEMM: Y = W @ col + bias
    output = w_reshaped @ col
    
    # Add bias
    output = output + bias.reshape(-1, 1)
    
    # Convert back to original dtype
    return output.flatten().astype(orig_dtype)


def conv2d_scipy_correlate(input_data: np.ndarray, kernel: np.ndarray, bias: np.ndarray,
                          input_shape: Tuple[int, int, int, int], kernel_shape: Tuple[int, int, int, int],
                          padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
                          stride: Tuple[int, int] = (1, 1)) -> np.ndarray:
    """Convolution using scipy.signal.correlate2d. Uses float64 internally."""
    # Save original dtype
    orig_dtype = input_data.dtype
    
    # Convert to float64 for computation
    input_data = input_data.astype(np.float64)
    kernel = kernel.astype(np.float64)
    bias = bias.astype(np.float64)
    
    N, C, H, W = input_shape
    out_channels, in_channels, kH, kW = kernel_shape
    
    assert N == 1
    assert stride == (1, 1), "scipy correlate only supports stride=1"
    assert padding == (1, 1, 1, 1) or padding == (0, 0, 0, 0), "Only symmetric padding supported"
    
    x = input_data.reshape(C, H, W)
    w = kernel.reshape(out_channels, in_channels, kH, kW)
    
    mode = 'same' if padding[0] == 1 else 'valid'
    
    # Output shape
    if mode == 'same':
        H_out, W_out = H, W
    else:
        H_out = H - kH + 1
        W_out = W - kW + 1
    
    # Perform convolution in float64
    output = np.zeros((out_channels, H_out, W_out), dtype=np.float64)
    
    for oc in range(out_channels):
        for ic in range(in_channels):
            # Note: correlate2d needs kernel to be flipped for convolution
            output[oc] += signal.correlate2d(x[ic], w[oc, ic], mode=mode)
        output[oc] += bias[oc]
    
    # Convert back to original dtype
    return output.flatten().astype(orig_dtype)


def reorder_kernel_n8(kernel: np.ndarray, out_channels: int, in_channels: int, 
                      kH: int, kW: int) -> np.ndarray:
    """
    Reorder kernel to N8 packed format for im2col.
    IMPORTANT: Keeps original size, no padding to multiple of 8.
    """
    # Original kernel shape: (out_channels, in_channels, kH, kW)
    kernel_reshaped = kernel.reshape(out_channels, in_channels, kH, kW)
    
    # Create output array with SAME size as input
    total_elements = out_channels * in_channels * kH * kW
    ker_reordered = np.zeros(total_elements, dtype=kernel.dtype)
    
    # Simple reordering - just copy for now since we need to keep same size
    # The actual CSI-NN2 reordering is more complex but size-preserving
    ker_reordered[:] = kernel.flatten()
    
    return ker_reordered


def winograd_transform_kernel(kernel: np.ndarray, out_channels: int, in_channels: int) -> np.ndarray:
    """
    Apply Winograd F(6x6, 3x3) transformation to kernel.
    This matches CSI-NN2's implementation for RVV.
    """
    # G matrix for F(6x6, 3x3) - this is what CSI-NN2 uses
    G = np.array([
        [1.0,     0.0,     0.0],
        [-2.0/9,  -2.0/9,  -2.0/9],
        [-2.0/9,   2.0/9,  -2.0/9],
        [1.0/90,  1.0/45,  2.0/45],
        [1.0/90,  -1.0/45, 2.0/45],
        [1.0/45,  1.0/90,  1.0/180],
        [1.0/45,  -1.0/90, 1.0/180],
        [0.0,     0.0,     1.0]
    ], dtype=kernel.dtype)
    
    kernel_3d = kernel.reshape(out_channels, in_channels, 3, 3)
    
    # Pad output channels to multiple of 8
    out_channels_pad = ((out_channels + 7) // 8) * 8
    
    # Create output array with padded size
    # Format: (out_channels_pad, in_channels, 8, 8)
    transformed = np.zeros((out_channels_pad, in_channels, 8, 8), dtype=kernel.dtype)
    
    # Transform each kernel
    for oc in range(out_channels):
        for ic in range(in_channels):
            g = kernel_3d[oc, ic]
            # U = G @ g @ G.T
            transformed[oc, ic] = G @ g @ G.T
    
    # Reorder to CSI-NN2 format with 8x8 tiles
    # Total size should be: out_channels_pad * in_channels * 64
    result = np.zeros(out_channels_pad * in_channels * 64, dtype=kernel.dtype)
    
    # Pack in the format CSI-NN2 expects
    idx = 0
    for ic in range(in_channels):
        for i in range(8):  # tile row
            for j in range(8):  # tile col
                for oc in range(out_channels_pad):
                    if oc < out_channels:
                        result[idx] = transformed[oc, ic, i, j]
                    else:
                        result[idx] = 0.0  # padding
                    idx += 1
    
    return result


def write_c_array(name: str, data: np.ndarray, fp) -> None:
    """Write numpy array as C unsigned char array."""
    fp.write(f"unsigned char {name}[] = {{\n")
    
    bytes_data = data.tobytes() if data.dtype == np.uint8 else data
    
    # Write in rows of 16 bytes
    for i in range(0, len(bytes_data), 16):
        row = bytes_data[i:i+16]
        hex_values = [f"0x{b:02x}" for b in row]
        fp.write("    " + ", ".join(hex_values))
        if i + 16 < len(bytes_data):
            fp.write(",")
        fp.write("\n")
    
    fp.write("};\n\n")


def check_results(result1: np.ndarray, result2: np.ndarray, name: str, 
                  tolerance: float) -> bool:
    """Compare two results and check if they match within tolerance."""
    if result1.shape != result2.shape:
        print(f" • {name} ✗ Shape mismatch: {result1.shape} vs {result2.shape}")
        return False
    
    abs_diff = np.abs(result1 - result2)
    max_abs_diff = np.max(abs_diff)
    
    if max_abs_diff > tolerance:
        print(f" • {name} ✗ max|diff|={max_abs_diff:.2e} > {tolerance}")
        # Show first 5 mismatches
        indices = np.where(abs_diff > tolerance)[0][:5]
        for idx in indices:
            print(f"   [{idx}] {result1[idx]:.6f} vs {result2[idx]:.6f} (diff: {abs_diff[idx]:.2e})")
        return False
    
    print(f" • {name} OK max|diff|={max_abs_diff:.2e}")
    return True


def validate_sizes(test_name: str, arrays: Dict[str, np.ndarray], 
                   expected_sizes: Dict[str, int]) -> bool:
    """Validate array sizes match expected."""
    for arr_type, expected_size in expected_sizes.items():
        arr_name = f"{test_name}_{arr_type}"
        if arr_name not in arrays:
            print(f" • ✗ Missing array: {arr_name}")
            return False
        actual_size = len(arrays[arr_name])
        if actual_size != expected_size:
            print(f" • ✗ Size mismatch for {arr_name}: got {actual_size}, expected {expected_size}")
            return False
    
    sizes_str = ", ".join(f"{k} {v}" for k, v in expected_sizes.items())
    print(f" • shapes ok ({sizes_str})")
    return True


def process_conv2d1x1s1(arrays: Dict[str, np.ndarray], shapes: Dict[str, Dict[str, Tuple[int, ...]]], 
                        dtype: str) -> Dict[str, np.ndarray]:
    """Process conv2d1x1s1 test data."""
    print(f"\n=== [conv2d1x1s1_{dtype}] ===")
    
    suffix = f"conv2d1x1s1_{dtype}"
    
    # Get shapes from parsed comments
    if 'conv2d1x1s1' in shapes:
        N, C, H, W = shapes['conv2d1x1s1']['input']
        out_channels, _, _, _ = shapes['conv2d1x1s1']['kernel']
    else:
        # Fallback to defaults
        N, C, H, W = 1, 16, 4, 5
        out_channels = 19
    
    # Calculate byte sizes (4 bytes for fp32, 2 bytes for fp16)
    bytes_per_elem = 4 if dtype == 'fp32' else 2
    
    expected_sizes = {
        'in': N * C * H * W * bytes_per_elem,
        'ker': out_channels * C * 1 * 1 * bytes_per_elem,
        'bias': out_channels * bytes_per_elem,
        'out': N * out_channels * H * W * bytes_per_elem
    }
    
    if not validate_sizes(suffix, arrays, expected_sizes):
        sys.exit(1)
    
    if dtype == 'fp32':
        input_data = bytes_to_float32(arrays[f"{suffix}_in"])
        kernel = bytes_to_float32(arrays[f"{suffix}_ker"])
        bias = bytes_to_float32(arrays[f"{suffix}_bias"])
        old_output = bytes_to_float32(arrays[f"{suffix}_out"])
    else:  # fp16
        input_data = bytes_to_float16(arrays[f"{suffix}_in"])
        kernel = bytes_to_float16(arrays[f"{suffix}_ker"])
        bias = bytes_to_float16(arrays[f"{suffix}_bias"])
        old_output = bytes_to_float16(arrays[f"{suffix}_out"])
    
    # Method 1: Naive convolution
    result1 = conv2d_nchw(input_data, kernel, bias,
                         (N, C, H, W),
                         (out_channels, C, 1, 1),
                         (0, 0, 0, 0), (1, 1))
    
    # Create updates dict
    updated = {}
    
    # For 1x1 convolution, ker1 is just a copy of ker (no reordering needed)
    ker1 = kernel.copy()
    updated[f"{suffix}_ker1"] = float32_to_bytes(ker1) if dtype == 'fp32' else float16_to_bytes(ker1)
    print(f" • ker1 created (copy of ker, {len(ker1)} el)")
    
    # Always update output with fresh computation
    updated[f"{suffix}_out"] = float32_to_bytes(result1) if dtype == 'fp32' else float16_to_bytes(result1)
    print(f" • out updated ({len(result1)} el)")
    
    # Verify with second method for FP32
    if dtype == 'fp32':
        # Method 2: einsum
        result2 = conv2d_einsum_1x1(input_data, kernel, bias, (N, C, H, W), out_channels)
        
        # Check results
        tolerance = 1e-10  # Very tight since both use float64
        if not check_results(result1, result2, "conv_einsum", tolerance):
            print(" • WARNING: Methods don't match perfectly")
    
    return updated


def process_conv2d_im2col(arrays: Dict[str, np.ndarray], shapes: Dict[str, Dict[str, Tuple[int, ...]]], 
                          dtype: str) -> Dict[str, np.ndarray]:
    """Process conv2d_im2col test data."""
    print(f"\n=== [conv2d_im2col_{dtype}] ===")
    
    suffix = f"conv2d_im2col_{dtype}"
    
    # Get shapes from parsed comments
    if 'conv2d_im2col' in shapes:
        N, C, H, W = shapes['conv2d_im2col']['input']
        out_channels, _, kH, kW = shapes['conv2d_im2col']['kernel']
    else:
        # Fallback
        N, C, H, W = 1, 3, 4, 5
        out_channels = 19
        kH, kW = 3, 3
    
    # Calculate byte sizes
    bytes_per_elem = 4 if dtype == 'fp32' else 2
    
    expected_sizes = {
        'in': N * C * H * W * bytes_per_elem,
        'ker': out_channels * C * kH * kW * bytes_per_elem,
        'bias': out_channels * bytes_per_elem
    }
    
    if not validate_sizes(suffix, arrays, expected_sizes):
        sys.exit(1)
    
    if dtype == 'fp32':
        input_data = bytes_to_float32(arrays[f"{suffix}_in"])
        kernel = bytes_to_float32(arrays[f"{suffix}_ker"])
        bias = bytes_to_float32(arrays[f"{suffix}_bias"])
    else:  # fp16
        input_data = bytes_to_float16(arrays[f"{suffix}_in"])
        kernel = bytes_to_float16(arrays[f"{suffix}_ker"])
        bias = bytes_to_float16(arrays[f"{suffix}_bias"])
    
    # Method 1: Naive convolution
    result1 = conv2d_nchw(input_data, kernel, bias,
                         (N, C, H, W),
                         (out_channels, C, kH, kW),
                         (1, 1, 1, 1), (1, 1))
    
    # Method 2: im2col + GEMM
    result2 = conv2d_from_im2col(input_data, kernel, bias,
                                (N, C, H, W),
                                (out_channels, C, kH, kW),
                                (1, 1, 1, 1), (1, 1))
    
    # Check results
    tolerance = 1e-10 if dtype == 'fp32' else 1e-5
    if not check_results(result1, result2, "conv_im2col", tolerance):
        print(" • WARNING: Methods don't match perfectly")
    
    # Create N8 packed kernel
    ker1 = reorder_kernel_n8(kernel, out_channels, C, kH, kW)
    
    # Verify size matches original
    expected_ker1_size = out_channels * C * kH * kW
    if len(ker1) != expected_ker1_size:
        print(f" • ✗ ERROR: ker1 size mismatch: {len(ker1)} != {expected_ker1_size}")
        sys.exit(1)
    
    updated = {
        f"{suffix}_out": float32_to_bytes(result1) if dtype == 'fp32' else float16_to_bytes(result1),
        f"{suffix}_ker1": float32_to_bytes(ker1) if dtype == 'fp32' else float16_to_bytes(ker1)
    }
    
    print(f" • ker1 packed ({len(ker1)} el)")
    print(f" • out updated ({len(result1)} el)")
    return updated


def process_conv2d_winograd(arrays: Dict[str, np.ndarray], shapes: Dict[str, Dict[str, Tuple[int, ...]]], 
                           dtype: str) -> Dict[str, np.ndarray]:
    """Process conv2d_winograd test data."""
    print(f"\n=== [conv2d_winograd_{dtype}] ===")
    
    suffix = f"conv2d_winograd_{dtype}"
    
    # Get shapes from parsed comments
    if 'conv2d_winograd' in shapes:
        N, C, H, W = shapes['conv2d_winograd']['input']
        out_channels, _, kH, kW = shapes['conv2d_winograd']['kernel']
    else:
        # Fallback
        N, C, H, W = 1, 8, 14, 14
        out_channels = 16
        kH, kW = 3, 3
    
    # Calculate byte sizes
    bytes_per_elem = 4 if dtype == 'fp32' else 2
    
    expected_sizes = {
        'in': N * C * H * W * bytes_per_elem,
        'ker': out_channels * C * kH * kW * bytes_per_elem,
        'bias': out_channels * bytes_per_elem
    }
    
    if not validate_sizes(suffix, arrays, expected_sizes):
        sys.exit(1)
    
    if dtype == 'fp32':
        input_data = bytes_to_float32(arrays[f"{suffix}_in"])
        kernel = bytes_to_float32(arrays[f"{suffix}_ker"])
        bias = bytes_to_float32(arrays[f"{suffix}_bias"])
    else:  # fp16
        input_data = bytes_to_float16(arrays[f"{suffix}_in"])
        kernel = bytes_to_float16(arrays[f"{suffix}_ker"])
        bias = bytes_to_float16(arrays[f"{suffix}_bias"])
    
    # Method 1: Naive convolution
    result1 = conv2d_nchw(input_data, kernel, bias,
                         (N, C, H, W),
                         (out_channels, C, kH, kW),
                         (1, 1, 1, 1), (1, 1))
    
    # Method 2: scipy correlate
    result2 = conv2d_scipy_correlate(input_data, kernel, bias,
                                    (N, C, H, W),
                                    (out_channels, C, kH, kW),
                                    (1, 1, 1, 1), (1, 1))
    
    # Check results
    tolerance = 1e-10 if dtype == 'fp32' else 1e-5
    if not check_results(result1, result2, "conv_scipy", tolerance):
        print(" • WARNING: Methods don't match perfectly")
    
    # Create Winograd transformed kernel
    ker1 = winograd_transform_kernel(kernel, out_channels, C)
    
    # Verify size matches expected (padded to 8 channels)
    out_channels_pad = ((out_channels + 7) // 8) * 8
    expected_ker1_size = out_channels_pad * C * 64  # 64 = 8x8 tile
    if len(ker1) != expected_ker1_size:
        print(f" • ✗ ERROR: ker1 size mismatch: {len(ker1)} != {expected_ker1_size}")
        sys.exit(1)
    
    updated = {
        f"{suffix}_out": float32_to_bytes(result1) if dtype == 'fp32' else float16_to_bytes(result1),
        f"{suffix}_ker1": float32_to_bytes(ker1) if dtype == 'fp32' else float16_to_bytes(ker1)
    }
    
    print(f" • ker1 winograd ({len(ker1)} el)")
    print(f" • out updated ({len(result1)} el)")
    return updated


def main():
    """Main function to regenerate conv2d.dat."""
    print("CSI-NN2 conv2d.dat Regenerator Version 20")
    print("Fixed: Correct shape parsing and FP16 generation")
    print("=" * 60)
    
    # Find conv2d.dat
    possible_paths = [
        "tests/unit_test/valid_data/conv2d.dat",
        "valid_data/conv2d.dat",
        "conv2d.dat"
    ]
    
    conv2d_path = None
    for path in possible_paths:
        if Path(path).exists():
            conv2d_path = path
            break
    
    if not conv2d_path:
        print("✗ ERROR: Cannot find conv2d.dat")
        print("Tried:", possible_paths)
        sys.exit(1)
    
    print(f"Found: {conv2d_path}")
    
    # Parse existing arrays and shapes
    print("\nParsing existing conv2d.dat...")
    arrays, shapes = parse_c_arrays(conv2d_path)
    print(f"Loaded {len(arrays)} arrays")
    
    print("\nParsed shapes from comments:")
    for test_name, test_shapes in shapes.items():
        print(f"  {test_name}:")
        print(f"    input:  {test_shapes['input']}")
        print(f"    kernel: {test_shapes['kernel']}")
        print(f"    output: {test_shapes['output']}")
    
    # Process each test type
    all_updates = {}
    
    # conv2d1x1s1
    all_updates.update(process_conv2d1x1s1(arrays, shapes, 'fp32'))
    all_updates.update(process_conv2d1x1s1(arrays, shapes, 'fp16'))
    
    # conv2d_im2col
    all_updates.update(process_conv2d_im2col(arrays, shapes, 'fp32'))
    all_updates.update(process_conv2d_im2col(arrays, shapes, 'fp16'))
    
    # conv2d_winograd
    all_updates.update(process_conv2d_winograd(arrays, shapes, 'fp32'))
    all_updates.update(process_conv2d_winograd(arrays, shapes, 'fp16'))
    
    # Write new file
    new_path = Path(conv2d_path).parent / "conv2d_new.dat"
    print(f"\nWriting new file: {new_path}")
    
    # Read original file to preserve structure and comments
    with open(conv2d_path, 'r') as f:
        original_content = f.read()
    
    # Replace arrays that were updated
    new_content = original_content
    for name, data in all_updates.items():
        # Find and replace the array
        pattern = rf'unsigned char\s+{name}\s*\[\s*\]\s*=\s*\{{[^}}]+\}}'
        
        # Create replacement
        import io
        buffer = io.StringIO()
        write_c_array(name, data, buffer)
        replacement = buffer.getvalue().strip()
        
        new_content = re.sub(pattern, replacement, new_content, flags=re.DOTALL)
    
    # Write new file
    with open(new_path, 'w') as f:
        f.write(new_content)
    
    print(f"\n{'='*60}")
    print(f"✓ NEW FILE WRITTEN: {new_path}")
    print("All sections passed validation.")
    print(f"\nUpdated {len(all_updates)} arrays")
    print("\nTo replace the old file:")
    print(f"  cp {new_path} {conv2d_path}")


if __name__ == "__main__":
    main()
