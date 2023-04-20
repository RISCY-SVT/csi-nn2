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
    init_testsuite("Testing function of logical xor i8.\n");

    struct csinn_tensor *input0 = csinn_alloc_tensor(NULL);
    struct csinn_tensor *input1 = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_diso_params *params = csinn_alloc_params(sizeof(struct csinn_diso_params), NULL);
    int in_size;
    float error[2] = {0};
    float max_error;

    int *buffer = read_input_data_f32(argv[1]);
    int flag = buffer[4];
    input0->dim[0] = buffer[0];
    input0->dim[1] = buffer[1];
    input0->dim[2] = buffer[2];
    input0->dim[3] = buffer[3];

    input1->dim[0] = buffer[0];
    input1->dim[1] = buffer[1];
    input1->dim[2] = buffer[2];
    input1->dim[3] = buffer[3];

    output->dim[0] = input0->dim[0];
    output->dim[1] = input0->dim[1];
    output->dim[2] = input0->dim[2];
    output->dim[3] = input0->dim[3];

    in_size = input0->dim[0] * input0->dim[1] * input0->dim[2] * input0->dim[3];
    input0->dim_count = 4;
    input1->dim_count = 4;
    output->dim_count = 4;
    input0->dtype = CSINN_DTYPE_INT8;
    input0->layout = CSINN_LAYOUT_NCHW;
    input0->is_const = 0;
    input0->quant_channel = 1;

    input1->dtype = CSINN_DTYPE_INT8;
    input1->layout = CSINN_LAYOUT_NCHW;
    input1->is_const = 0;
    input1->quant_channel = 1;

    output->dtype = CSINN_DTYPE_INT8;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;

    params->base.api = CSINN_API;
    params->base.layout = CSINN_LAYOUT_NCHW;

    float *src0_in = (float *)(buffer + 4);
    float *src1_in = (float *)(buffer + 4 + in_size);
    float *ref = (float *)(buffer + 4 + 2 * in_size);
    int8_t *src0_tmp = malloc(in_size * sizeof(char));
    int8_t *src1_tmp = malloc(in_size * sizeof(char));

    input0->data = src0_in;
    get_quant_info(input0);

    for (int i = 0; i < in_size; i++) {
        src0_tmp[i] = shl_ref_quantize_f32_to_i8(src0_in[i], input0->qinfo);
    }

    /* compute the max quantize error */
    for (int i = 0; i < in_size; i++) {
        float error1;
        float output_tmp = shl_ref_dequantize_i8_to_f32(src0_tmp[i], input0->qinfo);
        if (isinf(src0_in[i]) || isnan(src0_in[i])) {
            continue;
        } else {
            error1 = fabs(src0_in[i] - output_tmp);
            if (error1 > 1e-6) {
                error1 = fabs(src0_in[i] - output_tmp) / fabs(src0_in[i] + 1e-9);
            }
        }
        if (error1 > error[0]) {
            error[0] = error1;
        }
    }

    input1->data = src1_in;
    get_quant_info(input1);

    for (int i = 0; i < in_size; i++) {
        src1_tmp[i] = shl_ref_quantize_f32_to_i8(src1_in[i], input1->qinfo);
    }

    /* compute the max quantize error */
    for (int i = 0; i < in_size; i++) {
        float error1;
        float output_tmp = shl_ref_dequantize_i8_to_f32(src1_tmp[i], input1->qinfo);
        if (isinf(src1_in[i]) || isnan(src1_in[i])) {
            continue;
        } else {
            error1 = fabs(src1_in[i] - output_tmp);
            if (error1 > 1e-6) {
                error1 = fabs(src1_in[i] - output_tmp) / fabs(src1_in[i] + 1e-9);
            }
        }
        if (error1 > error[1]) {
            error[1] = error1;
        }
    }

    max_error = (error[0] + error[1]);

    output->data = ref;
    get_quant_info(output);

    input0->data = src0_tmp;
    input1->data = src1_tmp;
    reference->data = ref;
    output->data = malloc(in_size * sizeof(char));

    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csinn_logical_xor_init(input0, input1, output, params) == CSINN_TRUE) {
        csinn_logical_xor(input0, input1, output, params);
    }

    result_verify_8(reference->data, output, input0->data, difference, in_size, false);

    free(buffer);
    free(src0_tmp);
    free(src1_tmp);
    free(output->data);
    return done_testing();
}
