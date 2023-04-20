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

/*************************************************************
    note: VLEN = 128/256 ...
*************************************************************/
int shl_rvv_leaky_relu_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_relu_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float alpha = params->n;
    int size = csinn_tensor_size(input);
    while (size > 0) {
        int vl = vsetvl_e32m2(size);
        vfloat32m2_t _input = vle32_v_f32m2(input_data, vl);
        vbool16_t _mask = vmflt_vf_f32m2_b16(_input, 0.0f, vl);
        vfloat32m2_t _res = vfmul_vf_f32m2_m(_mask, _input, _input, alpha, vl);
        vse32_v_f32m2(output_data, _res, vl);
        input_data += vl;
        output_data += vl;
        size -= vl;
    }
    return CSINN_TRUE;
}
