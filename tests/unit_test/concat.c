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

#include "./valid_data/concat.dat"

#include "csi_nn.h"
#include "shl_thead_rvv.h"
#include "test_utils.h"

void verify_concat(void *input0_data, void *input1_data, void *ref_data, int (*func)(), int in_c,
                   int in_h, int in_w, int axis, enum csinn_dtype_enum dtype)
{
    struct csinn_tensor *input[2];

    input[0] = csinn_alloc_tensor(NULL);
    input[0]->dim[0] = 1;
    input[0]->dim[1] = in_c;
    input[0]->dim[2] = in_h;
    input[0]->dim[3] = in_w;
    input[0]->dim_count = 4;
    input[0]->name = "input0";

    input[1] = csinn_alloc_tensor(NULL);
    input[1]->dim[0] = 1;
    input[1]->dim[1] = in_c;
    input[1]->dim[2] = in_h;
    input[1]->dim[3] = in_w;
    input[1]->dim_count = 4;
    input[1]->name = "input1";

    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->dim[0] = 1;
    output->dim[1] = in_c;
    output->dim[2] = 2 * in_h;
    output->dim[3] = in_w;
    output->dim_count = 4;
    output->name = "output";
    int out_size = csinn_tensor_size(output);

    struct csinn_concat_params *params =
        csinn_alloc_params(sizeof(struct csinn_concat_params), NULL);
    params->base.name = "params";
    params->axis = axis;
    params->inputs_count = 2;

    input[0]->data = input0_data;
    input[1]->data = input1_data;
    output->data = shl_mem_alloc(out_size * sizeof(float));

    func((struct csinn_tensor **)input, output, params);

    evaluate_error(output->data, ref_data, out_size, dtype);

    csinn_free_tensor(input[0]);
    csinn_free_tensor(input[1]);
    shl_mem_free(output->data);
    csinn_free_tensor(output);
}

int main(int argc, char **argv)
{
    init_testsuite("Test function of concat for RVV.\n");
    verify_concat(concat_fp32_in0, concat_fp32_in1, concat_fp32_out, shl_rvv_concat_fp32, 2, 3, 10,
                  2, CSINN_DTYPE_FLOAT32);
    verify_concat(concat_fp16_in0, concat_fp16_in1, concat_fp16_out, shl_rvv_concat_fp16, 2, 3, 10,
                  2, CSINN_DTYPE_FLOAT16);
    // verify_concat(concat_int8_in0, concat_int8_in1, concat_int8_out, shl_rvv_concat_int8, 2,
    //               3, 10, 2, CSINN_DTYPE_FLOAT32);
    return done_testing();
}
