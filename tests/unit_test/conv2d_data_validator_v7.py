#!/usr/bin/env python3
"""
CSI-NN2 conv2d.dat validator with correct shape parsing.
Version 7: Fixed shape parsing from comments and added detailed diagnostics.
"""

import numpy as np
import re
from pathlib import Path
import sys
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ConvConfig:
    """Configuration for a convolution test."""
    name: str
    input_shape: Tuple[int, int, int, int]  # N, C, H, W
    kernel_shape: Tuple[int, int, int, int]  # OC, IC, kH, kW
    padding: Tuple[int, int, int, int]  # left, right, top, bottom
    stride: Tuple[int, int]
    description: str


def parse_c_arrays(filepath: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[int, ...]]]:
    """Parse C arrays from conv2d.dat file and extract shapes from comments."""
    arrays = {}
    shapes = {}
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"✗ ERROR: File not found: {filepath}")
        sys.exit(1)
    
    # Parse shape comments - look for patterns like:
    # // input: [1, 3, 4, 5], kernel: [19, 3, 3, 3], output: [1, 19, 4, 5]
    shape_pattern = r'//\s*input:\s*\[([\d,\s]+)\],\s*kernel:\s*\[([\d,\s]+)\],\s*output:\s*\[([\d,\s]+)\]'
    
    shape_blocks = list(re.finditer(shape_pattern, content))
    print(f"\nFound {len(shape_blocks)} shape comment blocks")
    
    # Map test names to shapes based on order
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
            print(f"  {test_names[i]}: input={input_shape}, kernel={kernel_shape}, output={output_shape}")
    
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


def conv2d_reference(input_data: np.ndarray, kernel: np.ndarray, bias: np.ndarray,
                    input_shape: Tuple[int, int, int, int],
                    kernel_shape: Tuple[int, int, int, int],
                    padding: Tuple[int, int, int, int],
                    stride: Tuple[int, int]) -> np.ndarray:
    """
    Reference convolution implementation using float64 for maximum accuracy.
    
    Args:
        input_data: flattened input array
        kernel: flattened kernel array
        bias: bias array
        input_shape: (N, C, H, W)
        kernel_shape: (out_channels, in_channels, kH, kW)
        padding: (left, right, top, bottom)
        stride: (stride_h, stride_w)
    
    Returns:
        flattened output array in float64
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
    
    # Reshape inputs (remove batch dimension for simplicity)
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
                patch = x_padded[:, h_start:h_start+kH, w_start:w_start+kW]
                conv_sum = 0.0
                for ic in range(in_channels):
                    for kh in range(kH):
                        for kw in range(kW):
                            conv_sum += patch[ic, kh, kw] * w[oc, ic, kh, kw]
                
                output[oc, oh, ow] = conv_sum + bias[oc]
    
    # Return in float64 for comparison
    return output.flatten()


def get_test_configs() -> Dict[str, ConvConfig]:
    """Get configuration for each test type."""
    return {
        'conv2d1x1s1': ConvConfig(
            name='conv2d1x1s1',
            input_shape=(1, 16, 4, 5),
            kernel_shape=(19, 16, 1, 1),
            padding=(0, 0, 0, 0),
            stride=(1, 1),
            description='Conv2D 1x1 stride=1 (16ch -> 19ch)'
        ),
        'conv2d_im2col': ConvConfig(
            name='conv2d_im2col',
            input_shape=(1, 3, 4, 5),
            kernel_shape=(19, 3, 3, 3),
            padding=(1, 1, 1, 1),
            stride=(1, 1),
            description='Conv2D im2col 3x3 (3ch -> 19ch)'
        ),
        'conv2d_winograd': ConvConfig(
            name='conv2d_winograd',
            input_shape=(1, 8, 14, 14),
            kernel_shape=(16, 8, 3, 3),
            padding=(1, 1, 1, 1),
            stride=(1, 1),
            description='Conv2D Winograd 3x3 (8ch -> 16ch)'
        )
    }


def validate_test(arrays: Dict[str, np.ndarray], config: ConvConfig, 
                  dtype: str, shapes_from_file: Dict) -> int:
    """
    Validate a single test configuration.
    Returns number of errors found.
    """
    print("=" * 80)
    print(f"VALIDATING: {config.name}_{dtype}")
    print("=" * 80)
    
    suffix = f"{config.name}_{dtype}"
    
    # Override config with shapes from file if available
    if config.name in shapes_from_file:
        file_shapes = shapes_from_file[config.name]
        # Update config with actual shapes from file
        config.input_shape = file_shapes['input']
        config.kernel_shape = file_shapes['kernel']
        # Recalculate expected output shape
        N, C, H, W = config.input_shape
        OC, IC, kH, kW = config.kernel_shape
    
    print(f"\nTest Description: {config.description}")
    print(f"Configuration:")
    print(f"  Input shape:  {config.input_shape[1:]} (C×H×W)")
    print(f"  Kernel shape: {config.kernel_shape} (Oc×Ic×kH×kW)")
    print(f"  Stride:       {config.stride}")
    print(f"  Padding:      L={config.padding[0]}, R={config.padding[1]}, T={config.padding[2]}, B={config.padding[3]}")
    
    # Check if arrays exist
    required_arrays = ['in', 'ker', 'bias', 'out']
    for arr in required_arrays:
        if f"{suffix}_{arr}" not in arrays:
            print(f"\n✗ ERROR: Missing array {suffix}_{arr}")
            return 1
    
    # Load arrays
    if dtype == 'fp32':
        input_data = bytes_to_float32(arrays[f"{suffix}_in"])
        kernel = bytes_to_float32(arrays[f"{suffix}_ker"])
        bias = bytes_to_float32(arrays[f"{suffix}_bias"])
        output_expected = bytes_to_float32(arrays[f"{suffix}_out"])
    else:  # fp16
        input_data = bytes_to_float16(arrays[f"{suffix}_in"])
        kernel = bytes_to_float16(arrays[f"{suffix}_ker"])
        bias = bytes_to_float16(arrays[f"{suffix}_bias"])
        output_expected = bytes_to_float16(arrays[f"{suffix}_out"])
    
    # Validate sizes
    N, C, H, W = config.input_shape
    OC, IC, kH, kW = config.kernel_shape
    
    expected_sizes = {
        'Input': (C * H * W, len(input_data)),
        'Kernel': (OC * IC * kH * kW, len(kernel)),
        'Bias': (OC, len(bias))
    }
    
    print("\nChecking array sizes:")
    size_ok = True
    for name, (expected, actual) in expected_sizes.items():
        if expected == actual:
            print(f"  {name}: got {actual}, expected {expected} ✓")
        else:
            print(f"  {name}: got {actual}, expected {expected} ✗")
            size_ok = False
    
    if not size_ok:
        return 1
    
    # Compute reference convolution
    print("\nComputing convolution...")
    output_computed = conv2d_reference(
        input_data, kernel, bias,
        config.input_shape,
        config.kernel_shape,
        config.padding,
        config.stride
    )
    
    # Calculate expected output size
    _, _, H_padded, W_padded = config.input_shape[0], config.input_shape[1], \
                               config.input_shape[2] + config.padding[2] + config.padding[3], \
                               config.input_shape[3] + config.padding[0] + config.padding[1]
    H_out = (H_padded - kH) // config.stride[0] + 1
    W_out = (W_padded - kW) // config.stride[1] + 1
    expected_output_size = OC * H_out * W_out
    
    print(f"\nOutput validation:")
    print(f"  Expected shape: ({OC}×{H_out}×{W_out}) = {expected_output_size} elements")
    print(f"  Expected array size: {len(output_expected)}")
    print(f"  Computed array size: {len(output_computed)}")
    
    if len(output_expected) != expected_output_size:
        print(f"  ✗ WARNING: Expected array size doesn't match calculated size!")
    
    # Compare values
    print("\nComparing values...")
    
    # Set tolerances based on dtype and algorithm
    if dtype == 'fp16':
        abs_tol = 2e-2  # Relaxed for FP16
        rel_tol = 1e-3
    elif 'winograd' in config.name:
        abs_tol = 3e-5  # Slightly relaxed for Winograd
        rel_tol = 1e-6
    else:
        abs_tol = 1e-5  # Standard FP32
        rel_tol = 1e-7
    
    # Convert expected output to float64 for comparison
    output_expected_f64 = output_expected.astype(np.float64)
    
    # Find mismatches
    abs_diff = np.abs(output_computed - output_expected_f64)
    rel_diff = np.where(output_expected_f64 != 0, 
                       abs_diff / np.abs(output_expected_f64), 
                       0)
    
    # Check both absolute and relative tolerance
    mismatches = (abs_diff > abs_tol) & (rel_diff > rel_tol)
    num_mismatches = np.sum(mismatches)
    
    if num_mismatches == 0:
        print(f"  ✓ ALL VALUES MATCH! (tolerance: {abs_tol:.0e})")
        print(f"  Max absolute difference: {np.max(abs_diff):.2e}")
        print(f"  Max relative difference: {np.max(rel_diff)*100:.2f}%")
    else:
        print(f"  ✗ FOUND {num_mismatches} MISMATCHES out of {len(output_expected)} values!")
        print(f"  Max absolute difference: {np.max(abs_diff):.2e}")
        print(f"  Max relative difference: {np.max(rel_diff)*100:.2f}%")
        
        # Show first few mismatches
        mismatch_indices = np.where(mismatches)[0][:5]
        print(f"\n  First {len(mismatch_indices)} mismatches:")
        print(f"  {'Index':>7} {'Ch':>3} {'H':>3} {'W':>3} {'Expected':>12} {'Computed':>12} {'AbsDiff':>12} {'RelDiff':>8}")
        print("  " + "-" * 75)
        
        for idx in mismatch_indices:
            ch = idx // (H_out * W_out)
            h = (idx % (H_out * W_out)) // W_out
            w = idx % W_out
            print(f"  {idx:>7} {ch:>3} {h:>3} {w:>3} {output_expected_f64[idx]:>12.6f} "
                  f"{output_computed[idx]:>12.6f} {abs_diff[idx]:>12.2e} {rel_diff[idx]*100:>7.1f}%")
    
    # Show first few values comparison
    print(f"\n  First 8 values comparison:")
    print(f"  {'Index':>5} {'Expected':>12} {'Computed':>12} {'Match':>6}")
    print("  " + "-" * 40)
    for i in range(min(8, len(output_expected))):
        match = "✓" if not mismatches[i] else "✗"
        print(f"  {i:>5} {output_expected_f64[i]:>12.6f} {output_computed[i]:>12.6f}      {match}")
    
    return num_mismatches


def main():
    """Main validation function."""
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Try to find conv2d.dat
        possible_paths = [
            "conv2d.dat",
            "valid_data/conv2d.dat",
            "tests/unit_test/valid_data/conv2d.dat"
        ]
        filepath = None
        for path in possible_paths:
            if Path(path).exists():
                filepath = path
                break
        
        if not filepath:
            print("✗ ERROR: Cannot find conv2d.dat")
            print("Usage: ./conv2d_data_validator.py [path/to/conv2d.dat]")
            sys.exit(1)
    
    print(f"Using file: {filepath}")
    
    print("\n" + "=" * 80)
    print("CONV2D DATA VALIDATOR Version 7")
    print("=" * 80)
    print(f"File: {filepath}")
    
    # Parse arrays and shapes
    print("\n" + "=" * 80)
    print("LOADING ARRAYS FROM FILE")
    print("=" * 80)
    
    arrays, shapes_from_file = parse_c_arrays(filepath)
    
    # Get test configurations
    configs = get_test_configs()
    
    # Load and display array info
    for test_name in ['conv2d1x1s1', 'conv2d_im2col', 'conv2d_winograd']:
        for dtype in ['fp32', 'fp16']:
            suffix = f"{test_name}_{dtype}"
            print(f"\nLoading {suffix} arrays ({dtype}):")
            
            # Expected arrays
            expected_arrays = ['in', 'ker', 'ker1', 'bias', 'out']
            for arr in expected_arrays:
                arr_name = f"{suffix}_{arr}"
                if arr_name in arrays:
                    num_elements = len(arrays[arr_name]) // (4 if dtype == 'fp32' else 2)
                    print(f"  ✓ {arr_name}: loaded {num_elements} elements")
                else:
                    print(f"  ✗ {arr_name}: NOT FOUND")
    
    # Validate each test
    total_errors = 0
    error_list = []
    
    for test_name in ['conv2d1x1s1', 'conv2d_im2col', 'conv2d_winograd']:
        for dtype in ['fp32', 'fp16']:
            config = configs[test_name]
            num_errors = validate_test(arrays, config, dtype, shapes_from_file)
            if num_errors > 0:
                total_errors += 1
                error_list.append(f"{test_name}_{dtype}: {num_errors} value mismatches")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if total_errors == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ Found {total_errors} ERRORS:")
        for i, error in enumerate(error_list, 1):
            print(f"  {i}. {error}")
    
    print(f"\nTotal tests checked: 6")
    print(f"Tests with errors: {total_errors}")
    
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
