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

#include "./valid_data/pool_data.dat"
#include "csi_nn.h"
#include "test_utils.h"

extern void verify_maxpool2d_q7(void *input_data, void *output_data, uint16_t batch, uint16_t in_h,
                                uint16_t in_w, uint16_t in_c, uint16_t out_h, uint16_t out_w,
                                uint16_t out_c, uint16_t kernel_h, uint16_t kernel_w,
                                uint16_t stride_h, uint16_t stride_w, uint16_t pad_x,
                                uint16_t pad_y, float difference);

int main(int argc, char **argv)
{
    init_testsuite("First testing function of maxpool q7 for xt800.\n");

    verify_maxpool2d_q7(pooling_input_00, maxpool2d_result_0, 1, 32, 32, 4, 30, 30, 4, 3, 3, 1, 1,
                        0, 0, 0.0f);

    verify_maxpool2d_q7(pooling_input_01, maxpool2d_result_1, 1, 32, 32, 4, 16, 16, 4, 2, 2, 2, 2,
                        0, 0, 0.0f);

    verify_maxpool2d_q7(pooling_input_02, maxpool2d_result_2, 1, 32, 32, 4, 17, 17, 4, 2, 2, 2, 2,
                        1, 1, 0.0f);

    verify_maxpool2d_q7(pooling_input_10, maxpool2d_result_3, 1, 64, 64, 1, 62, 62, 1, 3, 3, 1, 1,
                        0, 0, 0.0f);

    verify_maxpool2d_q7(pooling_input_11, maxpool2d_result_4, 1, 64, 64, 1, 32, 32, 1, 2, 2, 2, 2,
                        0, 0, 0.0f);

    verify_maxpool2d_q7(pooling_input_12, maxpool2d_result_5, 1, 64, 64, 1, 33, 33, 1, 2, 2, 2, 2,
                        1, 1, 0.0f);

    verify_maxpool2d_q7(pooling_input_20, maxpool2d_result_6, 1, 16, 16, 16, 14, 14, 16, 3, 3, 1, 1,
                        0, 0, 0.0f);

    verify_maxpool2d_q7(pooling_input_21, maxpool2d_result_7, 1, 16, 16, 16, 8, 8, 16, 2, 2, 2, 2,
                        0, 0, 0.0f);

    verify_maxpool2d_q7(pooling_input_22, maxpool2d_result_8, 1, 16, 16, 16, 9, 9, 16, 2, 2, 2, 2,
                        1, 1, 0.0f);
}
