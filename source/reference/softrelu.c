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

#include "shl_ref.h"

static float softrelu(float x, float y) { return log(1 + exp(fmax(fmin(x, y), y))); }

int shl_ref_softrelu_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_relu_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }

    for (int i = 0; i < size; i++) {
        output_data[i] = softrelu(input_data[i], params->n);
    }
    return CSINN_TRUE;
}

int shl_ref_softrelu_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_relu_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_softrelu_f32);
}
