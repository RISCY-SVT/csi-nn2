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

int shl_ref_scatter_nd_f32(struct csinn_tensor *input, struct csinn_tensor *indices,
                           struct csinn_tensor *updates, struct csinn_tensor *output,
                           struct csinn_scatter_nd_params *params)
{
    if (input->dim_count != 5 && indices->dim[indices->dim_count - 1] != 5) {
        return CSINN_FALSE;
    }
    float *input_data = (float *)input->data;
    int32_t *indices_data = (int32_t *)indices->data;
    float *updates_data = (float *)updates->data;
    float *output_data = (float *)output->data;

    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }
    for (int i = 0; i < size; i++) {
        output_data[i] = input_data[i];
    }

    for (int i = 0; i < indices->dim[0]; i++) {
        for (int j = 0; j < indices->dim[1]; j++) {
            for (int k = 0; k < indices->dim[2]; k++) {
                for (int l = 0; l < indices->dim[3]; l++) {
                    for (int m = 0; m < indices->dim[4]; m++) {
                        int indices_base =
                            ((((i * indices->dim[1] + j) * indices->dim[2] + k) * indices->dim[3] +
                              l) *
                                 indices->dim[4] +
                             m) *
                            indices->dim[5];

                        int output_index = shl_ref_get_index_5(
                            input->dim, indices_data[indices_base], indices_data[indices_base + 1],
                            indices_data[indices_base + 2], indices_data[indices_base + 3],
                            indices_data[indices_base + 4]);

                        int updates_index = shl_ref_get_index_5(updates->dim, i, j, k, l, m);
                        output_data[output_index] = updates_data[updates_index];
                    }
                }
            }
        }
    }

    return CSINN_TRUE;
}

int shl_ref_scatter_nd_quant(struct csinn_tensor *input, struct csinn_tensor *indices,
                             struct csinn_tensor *updates, struct csinn_tensor *output,
                             struct csinn_scatter_nd_params *params)
{
    struct csinn_tensor *float_input = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *float_updates = shl_ref_tensor_transform_f32(updates);
    struct csinn_tensor *float_output = shl_ref_tensor_transform_f32(output);
    int ret = shl_ref_scatter_nd_f32(float_input, indices, float_updates, float_output, params);
    csinn_tensor_data_convert(output, float_output);
    shl_ref_tensor_transform_free_f32(float_input);
    shl_ref_tensor_transform_free_f32(float_output);
    shl_ref_tensor_transform_free_f32(float_updates);
    return ret;
}
