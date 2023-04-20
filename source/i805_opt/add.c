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

#include "i805_function.h"
#include "shl_i805.h"

int shl_i805_add_init_u8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_diso_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    // update multiplier and shift for s1/s3, s2/s3
    shl_quantize_multiplier(input0->qinfo->scale / output->qinfo->scale,
                            &(input0->qinfo->multiplier), &(input0->qinfo->shift));
    shl_quantize_multiplier(input1->qinfo->scale / output->qinfo->scale,
                            &(input1->qinfo->multiplier), &(input1->qinfo->shift));
    cb->exec = shl_i805_add_u8;
    return CSINN_TRUE;
}

int shl_i805_add_u8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params)
{
    uint8_t *input0_data = (uint8_t *)input0->data;
    uint8_t *input1_data = (uint8_t *)input1->data;
    uint8_t *output_data = (uint8_t *)output->data;

    int32_t size = csinn_tensor_size(input0);

    shl_i805_elementwise_add_opt_u8(
        input0_data, input1_data, output_data, size, input0->qinfo->zero_point,
        input0->qinfo->multiplier, -input0->qinfo->shift, input1->qinfo->zero_point,
        input1->qinfo->multiplier, -input1->qinfo->shift, output->qinfo->zero_point,
        output->qinfo->multiplier, -output->qinfo->shift);
    return CSINN_TRUE;
}
