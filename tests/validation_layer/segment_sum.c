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
    init_testsuite("Testing function of segment sum(layer).\n");

    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_LAYER;
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);
    struct csinn_tensor *segment = csinn_alloc_tensor(sess);
    struct csinn_segment_params *params =
        csinn_alloc_params(sizeof(struct csinn_segment_params), sess);
    int in_size, out_size;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim[0] = buffer[0];
    input->dim[1] = buffer[1];
    input->dim[2] = buffer[2];
    input->dim[3] = buffer[3];
    output->dim[0] = buffer[4];
    output->dim[1] = buffer[1];
    output->dim[2] = buffer[2];
    output->dim[3] = buffer[3];

    input->dim_count = 4;
    output->dim_count = 4;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;
    params->num_segments = buffer[4];
    params->unsorted = CSINN_FALSE;
    params->base.api = CSINN_API;

    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];

    input->data = (float *)(buffer + 5);
    segment->data = (int *)(buffer + 5 + in_size);
    reference->data = (float *)(buffer + 5 + in_size + buffer[0]);
    output->data = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

    test_segment_sum_CSINN_QUANT_FLOAT32(input, segment, output, params, &difference);
    test_segment_sum_CSINN_QUANT_UINT8_ASYM(input, segment, output, params, &difference);
    test_segment_sum_CSINN_QUANT_INT8_SYM(input, segment, output, params, &difference);

    return done_testing();
}
