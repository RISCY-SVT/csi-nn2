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

#define DWCONV3X3S1_PACK4 shl_c906_dwconv3x3s1_pack4_fuse_relu
#define DWCONV3X3S2_PACK4 shl_c906_dwconv3x3s2_pack4_fuse_relu

#define FUSE_CONV_RELU

#include "./depthwise_convolution_3x3_pack4_fp32.c"
