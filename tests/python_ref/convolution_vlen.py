#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn
import math

def getpackn(test_dtype, test_vlen):
    if int(test_dtype) == 8:
        return int(test_vlen)/int(test_dtype)/2
    else:
        return int(test_vlen)/int(test_dtype)

def convolution_f32(test_dtype, test_vlen, test_type):

    para = []
    batch       = int(np.random.randint(1, high=2, size=1))
    in_size_x   = int(np.random.randint(6, high=7, size=1)) #width
    in_size_y   = int(np.random.randint(6, high=7, size=1)) #height
    stride_x    = int(np.random.randint(2, high=3, size=1))
    stride_y    = int(np.random.randint(2, high=3, size=1))
    kernel_x    = int(np.random.randint(stride_x, high=7, size=1))
    kernel_y    = int(np.random.randint(stride_y, high=7, size=1))
    dilation_x  = int(np.random.randint(1, high=2, size=1))
    dilation_y  = int(np.random.randint(1, high=2, size=1))

    packn = int(getpackn(test_dtype, test_vlen))
    n = int(np.random.randint(2, high=4, size=1))

    print(packn)

    if "pack1_" in test_type:
        in_channel  = packn * n + 1
        out_channel = packn * n + 1
        if test_type == "pack1_conv1x1s1":
            stride_x    = 1
            stride_y    = 1
            kernel_x    = 1
            kernel_y    = 1
            dilation_x  = 1
            dilation_y  = 1
        elif test_type == "pack1_gemm":
            stride_x    = 1
            stride_y    = 1
            kernel_x    = 1
            kernel_y    = 1
            out_channel = 12 + 8 + 4 + 2 + 1
            in_size_x = 7
            in_size_y = 9


    elif "packnto1" in test_type:
        in_channel  = packn * n
        out_channel = packn * n + 1
        if test_type == "packnto1_conv1x1s1":
            stride_x    = 1
            stride_y    = 1
            kernel_x    = 1
            kernel_y    = 1
            dilation_x  = 1
            dilation_y  = 1

    elif "pack1ton" in test_type:
        in_channel  = packn * n + 1
        out_channel = packn * n
        if test_type == "pack1ton_conv1x1s1":
            stride_x    = 1
            stride_y    = 1
            kernel_x    = 1
            kernel_y    = 1
            dilation_x  = 1
            dilation_y  = 1

    elif "packn_" in test_type:
        in_channel  = packn * n
        out_channel = packn * n
        if test_type == "packn_conv1x1s1":
            stride_x    = 1
            stride_y    = 1
            kernel_x    = 1
            kernel_y    = 1
            dilation_x  = 1
            dilation_y  = 1
        elif "packn_conv3x3s1" in test_type:
            stride_x    = 1
            stride_y    = 1
            kernel_x    = 3
            kernel_y    = 3
            dilation_x  = 1
            dilation_y  = 1

            if test_type == "packn_conv3x3s1_linput":
                in_size_x   = int(np.random.randint(13, high=20, size=1)) #width
                in_size_y   = int(np.random.randint(13, high=20, size=1)) #height




    kernel_x_t  = kernel_x + (kernel_x - 1) * (dilation_x - 1)
    kernel_y_t  = kernel_y + (kernel_y - 1) * (dilation_y - 1)
    pad_left   = pad_right = pad_top = pad_down = 0

    pad_x      = (in_size_x - kernel_x_t) -  int((in_size_x - kernel_x_t) / stride_x) * stride_x
    if(pad_x !=0):
        pad_x      = int((in_size_x - kernel_x_t) / stride_x) * stride_x + stride_x - (in_size_x - kernel_x_t)
        pad_left   = int(np.random.randint(0, high=pad_x, size=1))
        pad_right  = pad_x - pad_left

    pad_y      = (in_size_y - kernel_y_t) -  int((in_size_y - kernel_y_t) / stride_y) * stride_y
    if(pad_y != 0):
        pad_y      = int((in_size_y - kernel_y_t) / stride_y) * stride_y + stride_y - (in_size_y - kernel_y_t)
        pad_top    = int(np.random.randint(0, high=pad_y, size=1))
        pad_down   = pad_y - pad_top


    zero_point1 = int(np.random.randint(-3, high=3, size=1))
    std1        = int(np.random.randint(1, high=3, size=1))
    zero_point2 = int(np.random.randint(-3, high=3, size=1))
    std2        = int(np.random.randint(1, high=3, size=1))
    zero_point3 = int(np.random.randint(-6, high=6, size=1))
    std3        = int(np.random.randint(1, high=10, size=1))

    src_in = np.random.normal(zero_point1, std1, (batch, in_channel, in_size_y, in_size_x))
    weight = np.random.normal(zero_point2, std2, (out_channel, in_channel, kernel_y, kernel_x))
    bias   = np.random.normal(zero_point3, std3, out_channel)
    src_in = src_in.astype(np.float32)
    weight = weight.astype(np.float32)
    bias   = bias.astype(np.float32)


    t_src_in  = tensor(src_in)
    t_weight  = tensor(weight)
    t_bias    = tensor(bias)

    t_src_in  = fn.pad(t_src_in, (pad_left, pad_right, pad_top, pad_down), 'constant', 0)
    t_src_out1 = fn.conv2d(t_src_in, t_weight, bias=t_bias, stride=(stride_y, stride_x), dilation=(dilation_y, dilation_x)).numpy()

    out_size_x = np.shape(t_src_out1)[3]
    out_size_y = np.shape(t_src_out1)[2]



    # nc1c0hw ==> nc1hwc0
    if "packnto1" in test_type or "packn_" in test_type:
        src_in = src_in.reshape([batch, math.ceil(in_channel/packn), packn, in_size_y, in_size_x]).transpose( [0, 1, 3, 4, 2])

    if "pack1ton" in test_type or "packn_" in test_type:
        t_src_out1 = t_src_out1.reshape([batch, math.ceil(out_channel/packn), packn, out_size_y, out_size_x]).transpose( [0, 1, 3, 4, 2])



    src_in_1   = src_in.flatten()
    weight_1   = weight.flatten()
    src_out_1  = t_src_out1.flatten()



    total_size = (len(src_in_1) + len(src_out_1)) + len(weight_1) + len(bias) + 17

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_size_y)  #height
    para.append(in_size_x)  #width
    para.append(stride_y)
    para.append(stride_x)
    para.append(kernel_y)
    para.append(kernel_x)
    para.append(pad_left)
    para.append(pad_right)
    para.append(pad_top)
    para.append(pad_down)
    para.append(out_channel)
    para.append(dilation_x)
    para.append(dilation_y)
    para.append(out_size_x) #width
    para.append(out_size_y) #height
    print(para)


    with open("convolution_nchw_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(weight_1)), *weight_1)
        fp.write(data)
        data = struct.pack(('%df' % len(bias)), *bias)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    test_dtype = sys.argv[1]
    test_vlen = sys.argv[2]
    test_type = sys.argv[3]
    convolution_f32(test_dtype, test_vlen, test_type)
    print("end")
