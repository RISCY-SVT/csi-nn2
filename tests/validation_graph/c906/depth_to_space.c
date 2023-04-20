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
                 struct csinn_depth_to_space_params *params, struct csinn_session *sess,
                 struct csinn_tensor *real_input, float *output_data, float diff);

void test_f16(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_depth_to_space_params *params, float difference)
{
    printf("test depth_to_space f16\n");
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_api = CSINN_C906;
    sess->base_run_mode = CSINN_RM_CPU_GRAPH;
    sess->base_dtype = CSINN_DTYPE_FLOAT16;
    sess->base_quant_type = CSINN_QUANT_FLOAT16;
    // sess->debug_level = CSINN_DEBUG_LEVEL_INFO;
    params->base.sess = sess;
    enum csinn_dtype_enum test_dtype = CSINN_DTYPE_FLOAT16;

    struct csinn_tensor *qinput = convert_f32_input(input, test_dtype, sess);
    struct csinn_tensor *qoutput = convert_f32_input(output, test_dtype, sess);
    struct csinn_tensor *real_input = convert_f32_input(input, test_dtype, sess);

    op_test_run(qinput, qoutput, params, sess, real_input, output->data, difference);
}

void test_f32(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_depth_to_space_params *params, float difference)
{
    printf("test depth_to_space f32\n");
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_api = CSINN_C906;
    sess->base_run_mode = CSINN_RM_CPU_GRAPH;
    sess->base_dtype = CSINN_DTYPE_FLOAT32;
    sess->base_quant_type = CSINN_QUANT_FLOAT32;
    // sess->debug_level = CSINN_DEBUG_LEVEL_INFO;
    params->base.sess = sess;
    enum csinn_dtype_enum test_dtype = CSINN_DTYPE_FLOAT32;

    struct csinn_tensor *qinput = convert_f32_input(input, test_dtype, sess);
    struct csinn_tensor *qoutput = convert_f32_input(output, test_dtype, sess);
    struct csinn_tensor *real_input = convert_f32_input(input, test_dtype, sess);

    op_test_run(qinput, qoutput, params, sess, real_input, output->data, difference);
}

void test_depth_to_space(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_depth_to_space_params *params, float difference)
{
    params->base.api = CSINN_C906;

    test_f16(input, output, params, difference);
    test_f32(input, output, params, difference);
}
