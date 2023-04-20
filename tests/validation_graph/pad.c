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

void op_test_run(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_pad_params *params, struct csinn_session *sess,
                 struct csinn_tensor *real_input, float *output_data, float diff)
{
    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);
    csinn_pad_init(input, output, params);

    csinn_set_tensor_entry(input, sess);
    csinn_set_input(0, input, sess);

    csinn_pad(input, output, params);

    csinn_set_output(0, output, sess);
    csinn_session_setup(sess);

    csinn_update_input(0, real_input, sess);
    csinn_session_run(sess);
    csinn_get_output(0, output, sess);

    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    result_verify_f32(output_data, foutput->data, input->data, diff, csinn_tensor_size(output),
                      false);

    free_input(real_input);
    shl_ref_tensor_transform_free_f32(foutput);
    csinn_session_deinit(sess);
    csinn_free_session(sess);
}

void test_pad(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_pad_params *params, float difference);

int main(int argc, char **argv)
{
    init_testsuite("Testing function of pad(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);

    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    int in_size = 0, out_size = 0;

    /* input tensor configuration */
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = buffer[0];  // batch
    input->dim[1] = buffer[1];  // channel
    input->dim[2] = buffer[2];  // height
    input->dim[3] = buffer[3];  // width
    input->dim_count = 4;
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    input->name = "input";
    float *input_data = (float *)(buffer + 8);
    input->data = input_data;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;

    /* output tensor configuration */
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->dim[0] = input->dim[0];
    output->dim[1] = input->dim[1];
    output->dim[2] = input->dim[2] + buffer[6] + buffer[7];
    output->dim[3] = input->dim[3] + buffer[4] + buffer[5];
    output->dim_count = 4;
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    reference->data = (float *)(buffer + 8 + in_size);
    output->data = reference->data;
    output->name = "output";
    output->layout = CSINN_LAYOUT_NCHW;
    output->dtype = CSINN_DTYPE_FLOAT32;

    /* operator parameter configuration */
    struct csinn_pad_params *params = csinn_alloc_params(sizeof(struct csinn_pad_params), NULL);
    params->base.name = "params";
    params->base.layout = CSINN_LAYOUT_NCHW;
    params->pad_mode = CSINN_PAD_CONSTANT;
    params->pad_value = 0.0f;
    int32_t pad_left = buffer[4];
    int32_t pad_right = buffer[5];
    int32_t pad_top = buffer[6];
    int32_t pad_down = buffer[7];
    int32_t pad_before[4] = {0, 0, pad_top, pad_left};   // NCHW
    int32_t pad_after[4] = {0, 0, pad_down, pad_right};  // NCHW
    params->pad_before = pad_before;
    params->pad_after = pad_after;
    params->pad_num = input->dim_count;

    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
    test_pad(input, output, params, difference);

    return done_testing();
}
