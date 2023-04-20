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

#ifndef INCLUDE_SHL_C906_CAP_H_
#define INCLUDE_SHL_C906_CAP_H_

#include "csi_nn.h"
#include "shl_utils.h"

int shl_c906_conv2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                        struct csinn_conv2d_params *params);

int shl_c906_depthwise_conv2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);

int shl_c906_conv1d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                        struct csinn_conv1d_params *params);

int shl_c906_depthwise_conv1d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv1d_params *params);

int shl_c906_fullyconnected_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *weights, struct csinn_tensor *bias,
                                struct csinn_fc_params *params);

int shl_c906_maxpool2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_pool_params *params);

int shl_c906_avgpool2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_pool_params *params);

int shl_c906_div_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_c906_abs_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int shl_c906_add_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_c906_clip_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_clip_params *params);

int shl_c906_concat_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_clip_params *params);

int shl_c906_global_avgpool2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params);

int shl_c906_global_maxpool2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params);

int shl_c906_leaky_relu_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_relu_params *params);

int shl_c906_lrn_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_lrn_params *params);

int shl_c906_matmul_cap(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                        struct csinn_tensor *output, struct csinn_matmul_params *params);

int shl_c906_minimum_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_c906_mul_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_c906_prelu_cap(struct csinn_tensor *input, struct csinn_tensor *alpha,
                       struct csinn_tensor *output, struct csinn_prelu_params *params);

int shl_c906_relu_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_relu_params *params);

int shl_c906_relu1_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params);

int shl_c906_relu6_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params);

int shl_c906_split_cap(struct csinn_tensor *input, struct csinn_tensor **output,
                       struct csinn_split_params *params);

int shl_c906_sub_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_c906_reshape_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reshape_params *params);

int shl_c906_sum_stride_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_reduce_params *params);
#endif  // INCLUDE_SHL_C906_CAP_H_
