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
    init_testsuite("Testing function of pad nchw u8.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_pad_params *params = csinn_alloc_params(sizeof(struct csinn_pad_params), NULL);
    int in_size = 0, out_size = 0;

    int *buffer = read_input_data_f32(argv[1]);

    input->dim[0] = buffer[0];  // batch
    input->dim[1] = buffer[1];  // channel
    input->dim[2] = buffer[2];  // height
    input->dim[3] = buffer[3];  // width
    input->dim_count = 4;
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];

    output->dim[0] = input->dim[0];
    output->dim[1] = input->dim[1];
    output->dim[2] = input->dim[2] + buffer[6] + buffer[7];
    output->dim[3] = input->dim[3] + buffer[4] + buffer[5];
    output->dim_count = 4;
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];

    input->dtype = CSINN_DTYPE_UINT8;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    output->dtype = CSINN_DTYPE_UINT8;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;
    params->base.api = CSINN_API;
    params->base.layout = CSINN_LAYOUT_NCHW;
    params->pad_mode = CSINN_PAD_CONSTANT;
    params->pad_value = 0.0f;

    int32_t pad_left = buffer[4];
    int32_t pad_right = buffer[5];
    int32_t pad_top = buffer[6];
    int32_t pad_down = buffer[7];

    int32_t pad_before[4] = {0, pad_top, pad_left, 0};   // NHWC
    int32_t pad_after[4] = {0, pad_down, pad_right, 0};  // NHWC

    params->pad_before = pad_before;
    params->pad_after = pad_after;
    params->pad_num = input->dim_count;

    float *src_in = (float *)(buffer + 8);
    float *ref = (float *)(buffer + 8 + in_size);
    uint8_t *src_tmp = malloc(in_size * sizeof(char));

    input->data = src_in;
    get_quant_info(input);

    for (int i = 0; i < in_size; i++) {
        src_tmp[i] = shl_ref_quantize_f32_to_u8(src_in[i], input->qinfo);
    }

    output->data = ref;
    get_quant_info(output);

    input->data = src_tmp;
    reference->data = ref;
    output->data = malloc(out_size * sizeof(char));
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;

    if (csinn_pad_init(input, output, params) == CSINN_TRUE) {
        csinn_pad(input, output, params);
    }

    result_verify_8(reference->data, output, input->data, difference, out_size, false);

    free(buffer);
    free(src_tmp);
    free(output->data);
    return done_testing();
}
