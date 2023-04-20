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
    init_testsuite("Testing function of transpose u8.\n");
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_transpose_params *params =
        csinn_alloc_params(sizeof(struct csinn_transpose_params), NULL);
    int in_size = 1, out_size = 1;
    float max_error = 0.0f;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim_count = buffer[0];  // input->dim_count == 4
    output->dim_count = input->dim_count;

    int32_t *perm = (int32_t *)malloc(input->dim_count * sizeof(int32_t));

    for (int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[i + 1];
        perm[i] = buffer[input->dim_count + i + 1];
        output->dim[i] = buffer[2 * input->dim_count + i + 1];
        in_size *= input->dim[i];
    }
    out_size = in_size;

    input->dtype = CSINN_DTYPE_UINT8;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    output->dtype = CSINN_DTYPE_UINT8;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;
    params->base.api = CSINN_API;
    params->permute = perm;
    params->permute_num = input->dim_count;
    params->base.layout = CSINN_LAYOUT_NCHW;

    float *src_in = (float *)(buffer + 1 + input->dim_count * 3);
    float *ref = (float *)(buffer + 1 + input->dim_count * 3 + in_size);
    uint8_t *src_tmp = malloc(in_size * sizeof(uint8_t));

    input->data = src_in;
    get_quant_info(input);

    for (int i = 0; i < in_size; i++) {
        src_tmp[i] = shl_ref_quantize_f32_to_u8(src_in[i], input->qinfo);
    }

    /* compute the max quantize error */
    for (int i = 0; i < in_size; i++) {
        float error1;
        float output_tmp = shl_ref_dequantize_u8_to_f32(src_tmp[i], input->qinfo);
        if (isinf(src_in[i]) || isnan(src_in[i])) {
            continue;
        } else {
            error1 = fabs(src_in[i] - output_tmp);
            if (error1 > 1e-6) {
                error1 = fabs(src_in[i] - output_tmp) / fabs(src_in[i] + 1e-9);
            }
        }
        if (error1 > max_error) {
            max_error = error1;
        }
    }

    output->data = ref;
    get_quant_info(output);

    input->data = src_tmp;
    reference->data = ref;
    output->data = malloc(out_size * sizeof(char));

    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csinn_transpose_init(input, output, params) == CSINN_TRUE) {
        csinn_transpose(input, output, params);
    }

    result_verify_8(reference->data, output, input->data, difference, out_size, false);

    free(buffer);
    free(src_tmp);
    free(output->data);
    free(perm);
    return done_testing();
}
