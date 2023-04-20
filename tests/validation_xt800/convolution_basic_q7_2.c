/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* SHL version 2.1.x */

#include "./valid_data/q7_conv_basic.dat"
#include "csi_nn.h"
#include "test_utils.h"

extern void verify_conv2d_q7(void *input_data, void *kernel_data, void *bias_data, void *ref_data,
                             uint16_t batch, uint16_t in_h, uint16_t in_w, uint16_t in_c,
                             uint16_t out_h, uint16_t out_w, uint16_t out_c, uint16_t kernel_h,
                             uint16_t kernel_w, uint16_t stride_h, uint16_t stride_w,
                             uint16_t pad_x, uint16_t pad_y, uint16_t bias_shift,
                             uint16_t out_shift, float difference);

int main(int argc, char **argv)
{
    init_testsuite("Second testing function of convolution basic q7 for xt800.\n");

    verify_conv2d_q7(q7_conv_input_1, q7_conv_weight_1, q7_conv_bias_1, q7_conv_result_2, 1, 32, 32,
                     16, 28, 28, 16, 5, 5, 1, 1, 0, 0, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_1, q7_conv_weight_1, q7_conv_bias_1, q7_conv_result_3, 1, 32, 32,
                     16, 32, 32, 16, 5, 5, 1, 1, 2, 2, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_1, q7_conv_weight_1, q7_conv_bias_1, q7_conv_result_4, 1, 32, 32,
                     16, 12, 12, 16, 5, 5, 3, 3, 3, 3, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_1, q7_conv_weight_1, q7_conv_bias_1, q7_conv_result_10, 1, 31,
                     31, 15, 27, 27, 15, 5, 5, 1, 1, 0, 0, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_1, q7_conv_weight_1, q7_conv_bias_1, q7_conv_result_11, 1, 31,
                     31, 15, 31, 31, 15, 5, 5, 1, 1, 2, 2, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_1, q7_conv_weight_1, q7_conv_bias_1, q7_conv_result_12, 1, 31,
                     31, 15, 11, 11, 15, 5, 5, 3, 3, 2, 2, 0, 12, 0.0f);
}
