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
#include "testutil.h"

int main(int argc, char **argv)
{
    init_testsuite("Testing function of split(layer).\n");

    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_LAYER;
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    struct csinn_split_params *params =
        (csinn_split_params *)csinn_alloc_params(sizeof(struct csinn_split_params), sess);

    int *buffer = read_input_data_f32(argv[1]);
    input->dim[0] = buffer[0];  // batch
    input->dim[1] = buffer[1];  // channel
    input->dim[2] = buffer[2];  // height
    input->dim[3] = buffer[3];  // width
    int axis = buffer[4];
    int output_cnt = buffer[5];

    int in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    int out_size[output_cnt];
    int acc_out_size = 0;

    input->dim_count = 4;
    input->data = (float *)(buffer + 6);
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    params->base.api = CSINN_API;
    params->base.layout = CSINN_LAYOUT_NCHW;
    params->axis = axis;
    params->output_num = output_cnt;

    struct csinn_tensor *output[output_cnt];
    struct csinn_tensor *reference[output_cnt];

    int32_t *split_index = (int32_t *)malloc(output_cnt * sizeof(int32_t));
    for (int i = 0; i < output_cnt; i++) {
        split_index[i] = buffer[axis] / output_cnt;
    }

    for (int i = 0; i < output_cnt; i++) {
        reference[i] = csinn_alloc_tensor(sess);
        output[i] = csinn_alloc_tensor(sess);
        for (int j = 0; j < 4; j++) {
            if (j == axis) {
                output[i]->dim[j] = split_index[i];
            } else {
                output[i]->dim[j] = input->dim[j];
            }
        }
        output[i]->dim_count = 4;
        out_size[i] = output[i]->dim[0] * output[i]->dim[1] * output[i]->dim[2] * output[i]->dim[3];

        reference[i]->data = (float *)(buffer + 6 + in_size + acc_out_size);
        output[i]->data = reference[i]->data;
        acc_out_size += out_size[i];
        output[i]->dtype = CSINN_DTYPE_FLOAT32;
        output[i]->is_const = 0;
        output[i]->layout = CSINN_LAYOUT_NCHW;
        output[i]->quant_channel = 1;
    }

    int temp = 0;
    for (int i = 0; i < output_cnt; i++) {
        temp += split_index[i];
        split_index[i] = temp;
    }
    params->split_index = split_index;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

#if (DTYPE==32)
    test_split_op(input, output, params, CSINN_QUANT_FLOAT32, csinn_split_init,
                  csinn_split, &difference);
#elif (DTYPE==16)
    test_split_op(input, output, params, CSINN_QUANT_FLOAT16, csinn_split_init,
                  csinn_split, &difference);
#elif (DTYPE==8)
    test_split_op(input, output, params, CSINN_QUANT_INT8_SYM, csinn_split_init,
                  csinn_split, &difference);
#endif
    return done_testing();
}
