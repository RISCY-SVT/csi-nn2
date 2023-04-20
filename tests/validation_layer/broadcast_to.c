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

#include "csi_nn.h"
#include "test_utils.h"

int main(int argc, char **argv)
{
    init_testsuite("Testing function of broadcast_to(layer).\n");

    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_LAYER;
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);
    struct csinn_broadcast_to_params *params =
        csinn_alloc_params(sizeof(struct csinn_broadcast_to_params), sess);
    int in_size = 1;
    int out_size = 1;

    int *buffer = read_input_data_f32(argv[1]);

    input->dim_count = buffer[0];
    params->shape_count = buffer[1];
    output->dim_count = buffer[1];

    for (int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[2 + i];
        in_size = in_size * input->dim[i];
    }

    params->shape = (int *)malloc(params->shape_count * sizeof(int));

    for (int i = 0; i < params->shape_count; i++) {
        output->dim[i] = buffer[2 + input->dim_count + i];
        out_size = out_size * output->dim[i];
        params->shape[i] = output->dim[i];
    }
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;
    params->base.api = CSINN_API;

    input->data = (float *)(buffer + 2 + input->dim_count + params->shape_count);
    reference->data = (float *)(buffer + 2 + input->dim_count + params->shape_count + in_size);
    output->data = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

    test_broadcast_to_CSINN_QUANT_FLOAT32(input, output, params, &difference);
    test_broadcast_to_CSINN_QUANT_UINT8_ASYM(input, output, params, &difference);
    test_broadcast_to_CSINN_QUANT_INT8_SYM(input, output, params, &difference);

    return done_testing();
}
