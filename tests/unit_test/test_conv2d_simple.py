#!/usr/bin/env python3
"""
Simple test to verify our convolution implementation matches CSI-NN2 expectations.
"""

import numpy as np


def simple_conv2d_test():
    """Test with small, known values."""
    print("Simple Conv2D Test")
    print("=" * 50)
    
    # Test 1: 1x1 convolution
    print("\nTest 1: 1x1 convolution")
    print("-" * 30)
    
    # Input: 1x2x2x2 (N,C,H,W)
    input_1x1 = np.array([
        1.0, 2.0, 3.0, 4.0,  # channel 0
        5.0, 6.0, 7.0, 8.0   # channel 1
    ], dtype=np.float32)
    
    # Kernel: 3x2x1x1 (OC,IC,kH,kW)
    kernel_1x1 = np.array([
        1.0, 2.0,  # output channel 0
        3.0, 4.0,  # output channel 1
        5.0, 6.0   # output channel 2
    ], dtype=np.float32)
    
    # Bias: 3
    bias_1x1 = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    
    # Expected output calculation:
    # Out[0,0,0] = 1*1 + 5*2 + 0.5 = 11.5
    # Out[0,0,1] = 2*1 + 6*2 + 0.5 = 14.5
    # Out[0,1,0] = 3*1 + 7*2 + 0.5 = 17.5
    # Out[0,1,1] = 4*1 + 8*2 + 0.5 = 20.5
    
    print("Input shape: (1,2,2,2)")
    print("Input:", input_1x1)
    print("Kernel shape: (3,2,1,1)")
    print("Kernel:", kernel_1x1)
    print("Bias:", bias_1x1)
    
    # Compute manually
    input_2d = input_1x1.reshape(2, 2, 2)
    kernel_2d = kernel_1x1.reshape(3, 2, 1, 1)
    output = np.zeros((3, 2, 2))
    
    for oc in range(3):
        for h in range(2):
            for w in range(2):
                val = 0.0
                for ic in range(2):
                    val += input_2d[ic, h, w] * kernel_2d[oc, ic, 0, 0]
                output[oc, h, w] = val + bias_1x1[oc]
    
    print("\nExpected output:")
    for oc in range(3):
        print(f"  Channel {oc}: {output[oc].flatten()}")
    
    # Test 2: 3x3 convolution with padding
    print("\n\nTest 2: 3x3 convolution with padding")
    print("-" * 30)
    
    # Input: 1x1x3x3 (N,C,H,W)
    input_3x3 = np.array([
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    ], dtype=np.float32)
    
    # Kernel: 1x1x3x3 (OC,IC,kH,kW) - edge detector
    kernel_3x3 = np.array([
        -1.0, -1.0, -1.0,
        -1.0,  8.0, -1.0,
        -1.0, -1.0, -1.0
    ], dtype=np.float32)
    
    # Bias: 1
    bias_3x3 = np.array([0.0], dtype=np.float32)
    
    print("Input shape: (1,1,3,3)")
    print("Input:")
    print(input_3x3.reshape(3, 3))
    print("\nKernel shape: (1,1,3,3) - edge detector")
    print("Kernel:")
    print(kernel_3x3.reshape(3, 3))
    print("Bias:", bias_3x3)
    
    # With padding=1, output should be 3x3
    # Compute center pixel: 
    # (-1)*1 + (-1)*2 + (-1)*3 + (-1)*4 + 8*5 + (-1)*6 + (-1)*7 + (-1)*8 + (-1)*9
    # = -1 -2 -3 -4 +40 -6 -7 -8 -9 = 40 - 40 = 0
    
    print("\nWith padding=1, stride=1:")
    print("Center pixel computation:")
    center = 0.0
    for i in range(3):
        for j in range(3):
            center += input_3x3.reshape(3,3)[i,j] * kernel_3x3.reshape(3,3)[i,j]
    print(f"  Center value: {center}")
    
    # Test FP16 conversion
    print("\n\nTest 3: FP16 conversion")
    print("-" * 30)
    
    test_values = np.array([1.0, 2.5, -3.75, 100.0, 0.001], dtype=np.float32)
    print("Original FP32:", test_values)
    
    # Convert to FP16 and back
    fp16_values = test_values.astype(np.float16)
    print("As FP16:", fp16_values)
    
    # Show differences
    diff = np.abs(test_values - fp16_values.astype(np.float32))
    print("Absolute diff:", diff)
    print("Max diff:", np.max(diff))


def test_csi_nn2_data():
    """Test with actual CSI-NN2 test data sizes."""
    print("\n\nCSI-NN2 Data Size Analysis")
    print("=" * 50)
    
    # Based on the shape comments in conv2d.dat
    tests = {
        'conv2d1x1s1': {
            'input': (1, 16, 4, 5),
            'kernel': (19, 16, 1, 1),
            'output': (1, 19, 4, 5),
            'padding': (0, 0, 0, 0),
            'stride': (1, 1)
        },
        'conv2d_im2col': {
            'input': (1, 3, 4, 5),
            'kernel': (19, 3, 3, 3),
            'output': (1, 19, 4, 5),
            'padding': (1, 1, 1, 1),
            'stride': (1, 1)
        },
        'conv2d_winograd': {
            'input': (1, 8, 14, 14),
            'kernel': (16, 8, 3, 3),
            'output': (1, 16, 14, 14),
            'padding': (1, 1, 1, 1),
            'stride': (1, 1)
        }
    }
    
    for test_name, config in tests.items():
        print(f"\n{test_name}:")
        print("-" * 30)
        
        N, C, H, W = config['input']
        OC, IC, kH, kW = config['kernel']
        _, out_C, out_H, out_W = config['output']
        
        # Calculate expected sizes
        input_size = N * C * H * W
        kernel_size = OC * IC * kH * kW
        bias_size = OC
        output_size = N * out_C * out_H * out_W
        
        print(f"  Input:  {config['input']} = {input_size} elements")
        print(f"  Kernel: {config['kernel']} = {kernel_size} elements")
        print(f"  Bias:   ({OC},) = {bias_size} elements")
        print(f"  Output: {config['output']} = {output_size} elements")
        
        # For FP32 and FP16
        for dtype in ['fp32', 'fp16']:
            bytes_per_elem = 4 if dtype == 'fp32' else 2
            print(f"\n  {dtype} byte sizes:")
            print(f"    in:   {input_size * bytes_per_elem} bytes")
            print(f"    ker:  {kernel_size * bytes_per_elem} bytes")
            print(f"    bias: {bias_size * bytes_per_elem} bytes")
            print(f"    out:  {output_size * bytes_per_elem} bytes")
            
            if test_name == 'conv2d_winograd':
                # Winograd ker1 is padded to multiple of 8 channels
                OC_pad = ((OC + 7) // 8) * 8
                ker1_size = OC_pad * IC * 64  # 8x8 transformed tiles
                print(f"    ker1: {ker1_size * bytes_per_elem} bytes ({ker1_size} elements)")
            else:
                print(f"    ker1: {kernel_size * bytes_per_elem} bytes (same as ker)")


if __name__ == "__main__":
    simple_conv2d_test()
    test_csi_nn2_data()
