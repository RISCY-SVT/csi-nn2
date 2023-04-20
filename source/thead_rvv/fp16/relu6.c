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

#include "shl_thead_rvv.h"

int shl_rvv_relu6_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;

    __fp16 real6 = 6.0f / input->qinfo->scale;  // support fp16 quantization
    int size = csinn_tensor_size(input);
    while (size > 0) {
        int vl = vsetvl_e16m2(size);
        vfloat16m2_t _input = vle16_v_f16m2(input_data, vl);
        input_data += vl;
        vfloat16m2_t _output = vfmin_vf_f16m2(vfmax_vf_f16m2(_input, 0.0f, vl), real6, vl);
        vse16_v_f16m2(output_data, _output, vl);
        output_data += vl;
        size -= vl;
    }
    // requantize
    shl_rvv_siso_op_requantize_fp16(input, output);
    return CSINN_TRUE;
}
