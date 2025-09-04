/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

#include "c920v2/c920v2.h"

bool shl_c920v2_get_binary_model_op_init(struct csinn_session *sess)
{
    struct shl_c920v2_option *option = shl_c920v2_get_graph_option(sess);
    if (option && option->base.binary_model_op_init) {
        return true;
    } else {
        return false;
    }
}

void shl_c920v2_set_binary_model_op_init(struct csinn_session *sess, bool value)
{
    struct shl_c920v2_option *option = shl_c920v2_get_graph_option(sess);
    option->base.binary_model_op_init = value;
}

void *shl_c920v2_f32_to_input_dtype(uint32_t index, float *data, struct csinn_session *sess) {
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    csinn_get_input(index, input, sess);
    
    void *ret = NULL;
    
    if (input->dtype == CSINN_DTYPE_FLOAT32) {
        ret = data;
    } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
        int size = csinn_tensor_size(input);
        int16_t *ret_fp16 = (int16_t *)shl_mem_alloc(size * sizeof(int16_t));
        // Convert float32 to float16
        for (int i = 0; i < size; i++) {
            ret_fp16[i] = shl_ref_float32_to_float16(data[i]);
        }
        ret = ret_fp16;
    } else if (input->dtype == CSINN_DTYPE_INT8) {
        // Заглушка для int8 - не поддерживается на K1X
        shl_debug_warning("INT8 input not supported on K1X/C920V2\n");
        return NULL;
    } else if (input->dtype == CSINN_DTYPE_UINT8) {
        // Заглушка для uint8 - не поддерживается на K1X
        shl_debug_warning("UINT8 input not supported on K1X/C920V2\n");
        return NULL;
    }
    
    csinn_free_tensor(input);
    return ret;
}

float *shl_c920v2_output_to_f32_dtype(uint32_t index, void *data, struct csinn_session *sess) {
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    csinn_get_output(index, output, sess);
    
    float *ret = NULL;
    
    if (output->dtype == CSINN_DTYPE_FLOAT32) {
        ret = (float *)data;
    } else if (output->dtype == CSINN_DTYPE_FLOAT16) {
        int size = csinn_tensor_size(output);
        float *ret_fp32 = (float *)shl_mem_alloc(size * sizeof(float));
        int16_t *data_fp16 = (int16_t *)data;
        // Convert float16 to float32
        for (int i = 0; i < size; i++) {
            ret_fp32[i] = shl_ref_float16_to_float32(data_fp16[i]);
        }
        ret = ret_fp32;
    } else if (output->dtype == CSINN_DTYPE_INT8) {
        // Заглушка для int8
        shl_debug_warning("INT8 output not supported on K1X/C920V2\n");
        return NULL;
    } else if (output->dtype == CSINN_DTYPE_UINT8) {
        // Заглушка для uint8
        shl_debug_warning("UINT8 output not supported on K1X/C920V2\n");
        return NULL;
    }
    
    csinn_free_tensor(output);
    return ret;
}
