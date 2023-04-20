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

int shl_ref_not_u32(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params)

{
    uint32_t *input_data = input->data;
    uint32_t *output_data = output->data;
    int size = csinn_tensor_size(input);

    for (int i = 0; i < size; i++) {
        output_data[i] = ~(input_data[i]);
    }
    return CSINN_TRUE;
}

int shl_ref_not_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params)
{
    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    int size = csinn_tensor_size(input);

    for (int i = 0; i < size; i++) {
        output_data[i] = ~(input_data[i]);
    }
    return CSINN_TRUE;
}

int shl_ref_not_i8(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params)
{
    int8_t *input_data = input->data;
    int8_t *output_data = output->data;
    int size = csinn_tensor_size(input);

    for (int i = 0; i < size; i++) {
        output_data[i] = ~(input_data[i]);
    }
    return CSINN_TRUE;
}
