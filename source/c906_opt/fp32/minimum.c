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

#include "shl_c906.h"

static void element_minimum_f32(float *input0, float *input1, float *output, int size)
{
    asm volatile(
        "1:\n\t"
        "vsetvli    t0, %4, e32, m2\n\t"
        "vle.v      v8, (%2)\n\t"
        "sub        %4, %4, t0\n\t"
        "slli       t0, t0, 2\n\t"  // element: 4 bytes
        "add        %2, %2, t0\n\t"
        "vle.v      v12, (%3)\n\t"
        "add        %3, %3, t0\n\t"
        "vfmin.vv   v16, v8, v12\n\t"
        "vse.v      v16, (%0)\n\t"
        "add        %0, %0, t0\n\t"
        "bnez       %4, 1b\n\t"
        : "=r"(output)  // %0
        : "0"(output),  // %1
          "r"(input0),  // %2
          "r"(input1),  // %3
          "r"(size)     // %4
        : "v8", "v9", "v12", "v13", "v16", "v17", "t0");
}

int shl_c906_minimum_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_diso_params *params)
{
    float *input0_data = (float *)input0->data;
    float *input1_data = (float *)input1->data;
    float *output_data = (float *)output->data;
    int in_size0 = csinn_tensor_size(input0);
    int in_size1 = csinn_tensor_size(input1);
    int out_size = csinn_tensor_size(output);
    // example: [1, 3, 224, 224] + [1] = [1, 3, 224, 224]
    if (in_size1 == 1) {
        asm volatile(
            "flw        ft0, 0(%3)\n\t"
            "1:\n\t"
            "vsetvli    t0, %4, e32, m2\n\t"
            "vle.v      v8, (%2)\n\t"
            "sub        %4, %4, t0\n\t"
            "slli       t0, t0, 2\n\t"  // element: 4 bytes
            "add        %2, %2, t0\n\t"
            "vfmin.vf   v16, v8, ft0\n\t"
            "vse.v      v16, (%0)\n\t"
            "add        %0, %0, t0\n\t"
            "bnez       %4, 1b\n\t"
            : "=r"(output_data)  // %0
            : "0"(output_data),  // %1
              "r"(input0_data),  // %2
              "r"(input1_data),  // %3
              "r"(out_size)      // %4
            : "v8", "v9", "v16", "v17", "t0", "ft0");
    }
    // example: [1, 3, 224, 224] + [1, 3, 224, 224] = [1, 3, 224, 224]
    else if (in_size0 == in_size1) {
        element_minimum_f32(input0_data, input1_data, output_data, out_size);
    } else {
        int flag = 1;
        for (int i = input1->dim_count - 1, j = input0->dim_count - 1; i >= 0; i--, j--) {
            if (input0->dim[j] != input1->dim[i]) {
                flag = 0;
            }
        }
        // example: [1, 3, 224, 224] + [3, 224, 1] or [1, 3, 224, 224] + [3, 1, 224]
        if (!flag) {
            float *in0_data_b = shl_mem_alloc(out_size * 4);
            float *in1_data_b = shl_mem_alloc(out_size * 4);
            struct csinn_tensor *b_input0 = csinn_alloc_tensor(NULL);
            struct csinn_tensor *b_input1 = csinn_alloc_tensor(NULL);
            csinn_tensor_copy(b_input0, output);
            csinn_tensor_copy(b_input1, output);
            b_input0->data = in0_data_b;
            b_input1->data = in1_data_b;
            shl_ref_broadcast_to_shape_f32(input0, b_input0, output->dim, output->dim_count);
            shl_ref_broadcast_to_shape_f32(input1, b_input1, output->dim, output->dim_count);
            input0_data = b_input0->data;
            input1_data = b_input1->data;
            element_minimum_f32(input0_data, input1_data, output_data, out_size);
            shl_mem_free(in0_data_b);
            shl_mem_free(in1_data_b);
            shl_mem_free(b_input0);
            shl_mem_free(b_input1);
        }
        // example: [1, 3, 224, 224] + [224] = [1, 3, 224, 224]  ;  [1, 3, 224, 224] + [224, 224] =
        // [1, 3, 224, 224]
        else {
            int inner_size = in_size1;
            int outer_size = out_size / in_size1;
            for (int i = 0; i < outer_size; i++) {
                element_minimum_f32(input0_data, input1_data, output_data, inner_size);
                input0_data += inner_size;
                output_data += inner_size;
            }
        }
    }
    return CSINN_TRUE;
}
