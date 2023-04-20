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

// tf.nn.space_to_batch:the input mast a  4-D Tensor with shape [batch, height, width, depth].

int shl_ref_space_to_batch_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_space_to_batch_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int batch = input->dim[0];
    int in_channel = input->dim[1];
    int in_height = input->dim[2];
    int in_width = input->dim[3];

    int block_size = params->block_size;
    int block_size2 = block_size * block_size;

    int out_batch = output->dim[0];    // out_batch = in_batch * block_size * block_size;
    int out_channel = output->dim[1];  // out_channel = in_channel;
    int out_height = output->dim[2];   // out_height = (in_height) / block_size;
    int out_width = output->dim[3];    // out_width = (in_width) / block_size;

    for (int in_b = 0; in_b < batch; ++in_b) {
        for (int out_h = 0; out_h < out_height * block_size; out_h = out_h + block_size) {
            for (int out_w = 0; out_w < out_width * block_size; out_w = out_w + block_size) {
                for (int out_c = 0; out_c < in_channel; ++out_c) {
                    float *temp = (float *)shl_mem_alloc(block_size2 * sizeof(float));
                    int h_origin = out_h - params->pad_top;
                    int w_origin = out_w - params->pad_left;
                    for (int h = 0; h < block_size; ++h) {
                        for (int w = 0; w < block_size; ++w) {
                            int h_now = h_origin + h;
                            int w_now = w_origin + w;
                            if (h_now >= 0 && h_now < in_height && w_now >= 0 && w_now < in_width) {
                                int in_addr =
                                    shl_ref_get_index(input->dim, in_b, out_c, h_now, w_now);
                                temp[h * block_size + w] = input_data[in_addr];
                            }
                        }
                    }
                    int out_start_addr = shl_ref_get_index(output->dim, in_b, out_c,
                                                           out_h / block_size, out_w / block_size);
                    for (int i = 0; i < block_size2; ++i) {
                        output_data[out_start_addr +
                                    i * batch * out_channel * out_height * out_width] = temp[i];
                    }
                    shl_mem_free(temp);
                }
            }
        }
    }
    return CSINN_TRUE;
}

int shl_ref_space_to_batch_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_space_to_batch_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_space_to_batch_f32);
}
