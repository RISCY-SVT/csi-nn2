#!/usr/bin/env python3
"""
Final fix for conv2d.dat - this script will:
1. Read the existing conv2d.dat
2. Keep all existing data that passes validation
3. Only regenerate data that fails validation
4. Ensure FP16 data is properly generated
"""

import numpy as np
import re
from pathlib import Path
import sys
from typing import Dict, Tuple, List
import struct


def parse_existing_file(filepath: str) -> Tuple[Dict[str, bytes], str]:
    """Parse existing file and extract raw byte arrays."""
    arrays = {}
    
    with open(filepath, 'r') as f:
        content = f.read()
    
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
            arrays[array_name] = bytes([int(h, 16) for h in hex_values])
    
    return arrays, content


def validate_conv2d_output(test_name: str, input_bytes: bytes, kernel_bytes: bytes, 
                          bias_bytes: bytes, output_bytes: bytes, dtype: str) -> Tuple[bool, np.ndarray]:
    """
    Validate convolution output and return (is_valid, correct_output).
    """
    # Define test configurations
    configs = {
        'conv2d1x1s1': {
            'input_shape': (1, 16, 4, 5),
            'kernel_shape': (19, 16, 1, 1),
            'padding': (0, 0, 0, 0),
            'stride': (1, 1)
        },
        'conv2d_im2col': {
            'input_shape': (1, 3, 4, 5),
            'kernel_shape': (19, 3, 3, 3),
            'padding': (1, 1, 1, 1),
            'stride': (1, 1)
        },
        'conv2d_winograd': {
            'input_shape': (1, 8, 14, 14),
            'kernel_shape': (16, 8, 3, 3),
            'padding': (1, 1, 1, 1),
            'stride': (1, 1)
        }
    }
    
    config = configs[test_name]
    N, C, H, W = config['input_shape']
    OC, IC, kH, kW = config['kernel_shape']
    pad_l, pad_r, pad_t, pad_b = config['padding']
    stride_h, stride_w = config['stride']
    
    # Convert bytes to numpy arrays
    if dtype == 'fp32':
        input_data = np.frombuffer(input_bytes, dtype=np.float32)
        kernel = np.frombuffer(kernel_bytes, dtype=np.float32)
        bias = np.frombuffer(bias_bytes, dtype=np.float32)
        output_expected = np.frombuffer(output_bytes, dtype=np.float32)
    else:  # fp16
        input_data = np.frombuffer(input_bytes, dtype=np.float16)
        kernel = np.frombuffer(kernel_bytes, dtype=np.float16)
        bias = np.frombuffer(bias_bytes, dtype=np.float16)
        output_expected = np.frombuffer(output_bytes, dtype=np.float16)
    
    # Compute correct output using float64
    input_f64 = input_data.astype(np.float64).reshape(C, H, W)
    kernel_f64 = kernel.astype(np.float64).reshape(OC, IC, kH, kW)
    bias_f64 = bias.astype(np.float64)
    
    # Apply padding
    if any(config['padding']):
        input_padded = np.pad(input_f64, ((0, 0), (pad_t, pad_b), (pad_l, pad_r)), 
                             mode='constant', constant_values=0)
    else:
        input_padded = input_f64
    
    _, H_padded, W_padded = input_padded.shape
    H_out = (H_padded - kH) // stride_h + 1
    W_out = (W_padded - kW) // stride_w + 1
    
    # Compute convolution
    output_f64 = np.zeros((OC, H_out, W_out), dtype=np.float64)
    
    for oc in range(OC):
        for oh in range(H_out):
            for ow in range(W_out):
                h_start = oh * stride_h
                w_start = ow * stride_w
                
                conv_sum = 0.0
                for ic in range(IC):
                    for kh_idx in range(kH):
                        for kw_idx in range(kW):
                            x_val = input_padded[ic, h_start + kh_idx, w_start + kw_idx]
                            w_val = kernel_f64[oc, ic, kh_idx, kw_idx]
                            conv_sum += x_val * w_val
                
                output_f64[oc, oh, ow] = conv_sum + bias_f64[oc]
    
    # Convert to target dtype
    if dtype == 'fp32':
        output_correct = output_f64.astype(np.float32).flatten()
        tolerance = 1e-5
    else:  # fp16
        output_correct = output_f64.astype(np.float16).flatten()
        tolerance = 2e-2  # Relaxed for FP16
    
    # Check if output matches
    diff = np.abs(output_correct - output_expected)
    max_diff = np.max(diff)
    is_valid = max_diff <= tolerance
    
    return is_valid, output_correct


def generate_ker1(kernel_bytes: bytes, test_name: str, dtype: str) -> bytes:
    """Generate ker1 (reordered kernel) for a given test."""
    if dtype == 'fp32':
        kernel = np.frombuffer(kernel_bytes, dtype=np.float32)
    else:
        kernel = np.frombuffer(kernel_bytes, dtype=np.float16)
    
    if test_name == 'conv2d1x1s1':
        # For 1x1, ker1 is just a copy
        return kernel_bytes
    
    elif test_name == 'conv2d_im2col':
        # For im2col, simple reordering (keep same size)
        return kernel_bytes
    
    elif test_name == 'conv2d_winograd':
        # For Winograd, apply transformation
        OC = 16
        IC = 8
        
        # G matrix for F(6x6, 3x3)
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
        
        kernel_3d = kernel.reshape(OC, IC, 3, 3)
        
        # Pad output channels to multiple of 8
        OC_pad = ((OC + 7) // 8) * 8
        
        # Transform
        transformed = np.zeros((OC_pad, IC, 8, 8), dtype=kernel.dtype)
        for oc in range(OC):
            for ic in range(IC):
                transformed[oc, ic] = G @ kernel_3d[oc, ic] @ G.T
        
        # Reorder for CSI-NN2
        result = np.zeros(OC_pad * IC * 64, dtype=kernel.dtype)
        idx = 0
        for ic in range(IC):
            for i in range(8):
                for j in range(8):
                    for oc in range(OC_pad):
                        if oc < OC:
                            result[idx] = transformed[oc, ic, i, j]
                        else:
                            result[idx] = 0.0
                        idx += 1
        
        return result.tobytes()
    
    return kernel_bytes


def format_c_array(name: str, data: bytes) -> str:
    """Format bytes as C array."""
    lines = [f"unsigned char {name}[] = {{"]
    
    for i in range(0, len(data), 16):
        row = data[i:i+16]
        hex_values = [f"0x{b:02x}" for b in row]
        line = "    " + ", ".join(hex_values)
        if i + 16 < len(data):
            line += ","
        lines.append(line)
    
    lines.append("};")
    return "\n".join(lines)


def main():
    """Main function."""
    print("Conv2D Data Final Fixer")
    print("=" * 60)
    
    # Find conv2d.dat
    conv2d_path = None
    for path in ["valid_data/conv2d.dat", "tests/unit_test/valid_data/conv2d.dat", "conv2d.dat"]:
        if Path(path).exists():
            conv2d_path = path
            break
    
    if not conv2d_path:
        print("✗ ERROR: Cannot find conv2d.dat")
        sys.exit(1)
    
    print(f"Found: {conv2d_path}")
    
    # Parse existing file
    print("\nParsing existing file...")
    arrays, original_content = parse_existing_file(conv2d_path)
    print(f"Loaded {len(arrays)} arrays")
    
    # Process each test
    fixes_needed = {}
    
    test_configs = [
        ('conv2d1x1s1', 'fp32'),
        ('conv2d1x1s1', 'fp16'),
        ('conv2d_im2col', 'fp32'),
        ('conv2d_im2col', 'fp16'),
        ('conv2d_winograd', 'fp32'),
        ('conv2d_winograd', 'fp16'),
    ]
    
    for test_name, dtype in test_configs:
        suffix = f"{test_name}_{dtype}"
        print(f"\nChecking {suffix}...")
        
        # Get arrays
        input_bytes = arrays.get(f"{suffix}_in", b'')
        kernel_bytes = arrays.get(f"{suffix}_ker", b'')
        bias_bytes = arrays.get(f"{suffix}_bias", b'')
        output_bytes = arrays.get(f"{suffix}_out", b'')
        
        if not all([input_bytes, kernel_bytes, bias_bytes, output_bytes]):
            print(f"  ✗ Missing arrays for {suffix}")
            continue
        
        # Validate output
        is_valid, correct_output = validate_conv2d_output(
            test_name, input_bytes, kernel_bytes, bias_bytes, output_bytes, dtype
        )
        
        if not is_valid:
            print(f"  ✗ Output invalid, will regenerate")
            if dtype == 'fp32':
                fixes_needed[f"{suffix}_out"] = correct_output.astype(np.float32).tobytes()
            else:
                fixes_needed[f"{suffix}_out"] = correct_output.astype(np.float16).tobytes()
        else:
            print(f"  ✓ Output valid")
        
        # Generate ker1 if missing or update it
        ker1_name = f"{suffix}_ker1"
        if ker1_name not in arrays or test_name == 'conv2d_winograd':
            print(f"  → Generating {ker1_name}")
            fixes_needed[ker1_name] = generate_ker1(kernel_bytes, test_name, dtype)
    
    # Apply fixes
    if fixes_needed:
        print(f"\nApplying {len(fixes_needed)} fixes...")
        
        new_content = original_content
        for name, data in fixes_needed.items():
            # Find and replace the array
            pattern = rf'unsigned char\s+{name}\s*\[\s*\]\s*=\s*\{{[^}}]+\}}'
            replacement = format_c_array(name, data)
            
            if re.search(pattern, new_content):
                new_content = re.sub(pattern, replacement, new_content, flags=re.DOTALL)
                print(f"  ✓ Replaced {name}")
            else:
                # Array doesn't exist, need to add it
                print(f"  + Adding new array {name}")
                # Find a good place to insert (after the corresponding _ker array)
                base_name = name.replace('_ker1', '_ker')
                insert_pattern = rf'(unsigned char\s+{base_name}\s*\[\s*\]\s*=\s*\{{[^}}]+\}};\s*\n)'
                if re.search(insert_pattern, new_content):
                    new_content = re.sub(insert_pattern, r'\1\n' + replacement + '\n\n', new_content)
        
        # Write new file
        new_path = Path(conv2d_path).parent / "conv2d_fixed.dat"
        with open(new_path, 'w') as f:
            f.write(new_content)
        
        print(f"\n✓ Fixed file written to: {new_path}")
        print("\nTo use the fixed file:")
        print(f"  cp {new_path} {conv2d_path}")
    else:
        print("\n✓ All data is already valid!")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
