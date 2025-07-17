#!/usr/bin/env python3
"""
Debug script to check actual sizes in conv2d.dat files.
This will help us understand what's wrong with the data.
"""

import numpy as np
import re
from pathlib import Path
import sys


def parse_c_arrays(filepath: str) -> dict:
    """Parse C arrays and return their sizes."""
    arrays = {}
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"✗ ERROR: File not found: {filepath}")
        return arrays
    
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
            arrays[array_name] = len(hex_values)
        
        # Print names and sizes
        print(f"Array: {array_name}, Size: {len(hex_values)} bytes")
    
    return arrays


def bytes_to_float32(byte_array: np.ndarray) -> np.ndarray:
    """Convert byte array to float32 array."""
    return np.frombuffer(byte_array.tobytes(), dtype=np.float32)


def check_conv2d_file(filepath: str):
    """Check sizes and first few values of arrays in conv2d.dat."""
    print(f"\nChecking: {filepath}")
    print("=" * 80)
    
    # Parse arrays
    array_sizes = parse_c_arrays(filepath)
    
    # Parse actual content for first values
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except:
        return
    
    # Group by test type
    test_types = ['conv2d1x1s1', 'conv2d_im2col', 'conv2d_winograd']
    
    for test_type in test_types:
        print(f"\n{test_type}:")
        print("-" * 40)
        
        for dtype in ['fp32', 'fp16']:
            suffix = f"{test_type}_{dtype}"
            bytes_per_elem = 4 if dtype == 'fp32' else 2
            
            print(f"\n  {dtype}:")
            
            # Check each array type
            for arr_type in ['in', 'ker', 'ker1', 'bias', 'out']:
                arr_name = f"{suffix}_{arr_type}"
                if arr_name in array_sizes:
                    byte_size = array_sizes[arr_name]
                    elem_count = byte_size // bytes_per_elem
                    print(f"    {arr_type:>5}: {byte_size:>5} bytes = {elem_count:>4} elements")
                    
                    # Extract first few values
                    pattern = rf'unsigned char\s+{arr_name}\s*\[\s*\]\s*=\s*\{{([^}}]+)\}}'
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        array_content = match.group(1)
                        # Get first 32 bytes
                        hex_values = re.findall(r'0x([0-9a-fA-F]{2})', array_content)[:32]
                        if hex_values and arr_type == 'out':
                            byte_array = bytes([int(h, 16) for h in hex_values[:8 if dtype == 'fp32' else 4]])
                            if dtype == 'fp32':
                                first_vals = np.frombuffer(byte_array, dtype=np.float32)
                                print(f"           First 2 values: {first_vals[0]:.6f}, {first_vals[1]:.6f}")
                            else:
                                first_vals = np.frombuffer(byte_array, dtype=np.float16)
                                print(f"           First 2 values: {first_vals[0]:.6f}, {first_vals[1]:.6f}")
                else:
                    print(f"    {arr_type:>5}: NOT FOUND")
    
    # Check for shape comments
    print("\n\nShape comments found:")
    print("-" * 40)
    shape_pattern = r'//\s*input:\s*\[([\d,\s]+)\],\s*kernel:\s*\[([\d,\s]+)\],\s*output:\s*\[([\d,\s]+)\]'
    
    for i, match in enumerate(re.finditer(shape_pattern, content)):
        input_shape = match.group(1).strip()
        kernel_shape = match.group(2).strip()
        output_shape = match.group(3).strip()
        print(f"Block {i+1}:")
        print(f"  input:  [{input_shape}]")
        print(f"  kernel: [{kernel_shape}]")
        print(f"  output: [{output_shape}]")


def main():
    """Main function."""
    print("Conv2D Data Size Debugger")
    print("=" * 80)
    
    # Check for files
    files_to_check = [
        "valid_data/conv2d.dat",
        "valid_data/conv2d_new.dat",
        "tests/unit_test/valid_data/conv2d.dat",
        "tests/unit_test/valid_data/conv2d_new.dat"
    ]
    
    for filepath in files_to_check:
        if Path(filepath).exists():
            check_conv2d_file(filepath)
            
            # Get file info
            stat = Path(filepath).stat()
            print(f"\nFile info:")
            print(f"  Size: {stat.st_size} bytes")
            print(f"  Modified: {Path(filepath).stat().st_mtime}")


if __name__ == "__main__":
    main()
