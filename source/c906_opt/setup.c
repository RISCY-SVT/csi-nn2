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
#include "shl_c906_cap.h"

static struct shl_cb_op_list shl_c906_cb_op_list;

int shl_c906_reg_op(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *init, void *exec)
{
    struct shl_cb_op_list *list_end = shl_cb_list_end(&shl_c906_cb_op_list);
    struct shl_cb_op_list *next = shl_mem_alloc(sizeof(struct shl_cb_op_list));
    next->cb = shl_mem_alloc(sizeof(struct csinn_callback));
    next->cb->init = init;
    next->cb->exec = exec;
    next->dtype = dtype;
    next->op_name = op_name;
    list_end->next = next;
    return CSINN_TRUE;
}

int shl_c906_reg_op_est(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *est)
{
    struct csinn_callback *cb = shl_cb_list_match(&shl_c906_cb_op_list, dtype, op_name);
    if (cb == NULL) {
        shl_debug_info("%s: cannot find c906 est\n", __func__);
    } else {
        cb->est = est;
    }

    return CSINN_TRUE;
}

int shl_c906_reg_op_cap(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *caps)
{
    struct csinn_callback *cb = shl_cb_list_match(&shl_c906_cb_op_list, dtype, op_name);
    if (cb == NULL) {
        shl_debug_info("%s: cannot find c906 caps\n", __func__);
    } else {
        cb->caps = caps;
    }

    return CSINN_TRUE;
}

struct csinn_callback *__attribute__((weak)) shl_cb_map_rvv(int op, int dtype);
struct csinn_callback *shl_cb_map_c906(int op, int dtype)
{
    struct csinn_callback *cb = shl_cb_list_match(&shl_c906_cb_op_list, dtype, op);
    if (cb == NULL) {
        cb = shl_cb_map_rvv(op, dtype);
    }
    return cb;
}

void __attribute__((weak)) shl_target_init_c906()
{
    shl_register_runtime_callback(CSINN_C906, NULL);
    shl_register_op_callback(CSINN_C906, shl_cb_map_c906);

#ifndef CONFIG_C906_CONVOLUTION_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_c906_conv2d_init_fp16, NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GROUP_CONV2D, shl_c906_conv2d_init_fp16, NULL);
#endif
#ifndef CONFIG_C906_CONVOLUTION_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D, shl_c906_conv2d_init_fp32, NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GROUP_CONV2D, shl_c906_conv2d_init_fp32, NULL);
#endif
#ifndef CONFIG_C906_CONVOLUTION1D_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV1D, shl_c906_conv1d_init_fp16, NULL);
#endif
#ifndef CONFIG_C906_CONVOLUTION1D_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV1D, shl_c906_conv1d_init_fp32, NULL);
#endif
#ifndef CONFIG_C906_MAXPOOL_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MAXPOOL2D, shl_c906_maxpool2d_init_fp16, NULL);
#endif
#ifndef CONFIG_C906_MAXPOOL_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MAXPOOL2D, shl_c906_maxpool2d_init_fp32, NULL);
#endif
#ifndef CONFIG_C906_AVERAGEPOOL_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_AVGPOOL2D, shl_c906_avgpool2d_init_fp16, NULL);
#endif
#ifndef CONFIG_C906_AVERAGEPOOL_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_AVGPOOL2D, shl_c906_avgpool2d_init_fp32, NULL);
#endif
#ifndef CONFIG_C906_DEPTHWISE_CONVOLUTION_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D,
                    shl_c906_depthwise_conv2d_init_fp16, NULL);
#endif
#ifndef CONFIG_C906_DEPTHWISE_CONVOLUTION_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DEPTHWISE_CONV2D,
                    shl_c906_depthwise_conv2d_init_fp32, NULL);
#endif
#ifndef CONFIG_C906_DEPTHWISE_CONVOLUTION1D_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV1D,
                    shl_c906_depthwise_conv1d_init_fp16, NULL);
#endif
#ifndef CONFIG_C906_FULLYCONNECTED_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_FULLYCONNECTED, shl_c906_fullyconnected_init_fp16,
                    NULL);
#endif
#ifndef CONFIG_C906_DIV_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DIV, shl_c906_div_init_fp16, NULL);
#endif
#ifndef CONFIG_C906_DIV_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DIV, shl_c906_div_init_fp32, NULL);
#endif
#ifndef CONFIG_C906_ABS_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ABS, NULL, shl_c906_abs_fp16);
#endif
#ifndef CONFIG_C906_ADD_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ADD, NULL, shl_c906_add_fp16);
#endif
#ifndef CONFIG_C906_CACHE_CONV1D_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CACHE_MATMUL, shl_c906_cache_matmul_init,
                    shl_c906_cache_matmul_fp16);
#endif
#ifndef CONFIG_C906_CACHE_CONV1D_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CACHE_CONV1D, shl_c906_cache_conv1d_init,
                    shl_c906_cache_conv1d_fp16);
#endif
#ifndef CONFIG_C906_CLIP_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CLIP, NULL, shl_c906_clip_fp16);
#endif
#ifndef CONFIG_C906_CONCAT_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONCAT, NULL, shl_c906_concat_fp16);
#endif
#ifndef CONFIG_C906_GLOBAL_AVERAGEPOOL_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_AVGPOOL2D, NULL,
                    shl_c906_global_avgpool2d_fp16);
#endif
#ifndef CONFIG_C906_GLOBAL_MAXPOOL_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_MAXPOOL2D, NULL,
                    shl_c906_global_maxpool2d_fp16);
#endif
#ifndef CONFIG_C906_LEAKY_RELU_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LEAKY_RELU, NULL, shl_c906_leaky_relu_fp16);
#endif
#ifndef CONFIG_C906_LRN_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LRN, NULL, shl_c906_lrn_fp16);
#endif
#ifndef CONFIG_C906_MATMUL_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MATMUL, shl_c906_matmul_init_fp16, NULL);
#endif
#ifndef CONFIG_C906_MINIMUM_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MINIMUM, NULL, shl_c906_minimum_fp16);
#endif
#ifndef CONFIG_C906_MUL_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MUL, NULL, shl_c906_mul_fp16);
#endif
#ifndef CONFIG_C906_PRELU_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_PRELU, NULL, shl_c906_prelu_fp16);
#endif
#ifndef CONFIG_C906_RELU_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU, NULL, shl_c906_relu_fp16);
#endif
#ifndef CONFIG_C906_RELU1_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU1, NULL, shl_c906_relu1_fp16);
#endif
#ifndef CONFIG_C906_RELU6_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU6, NULL, shl_c906_relu6_fp16);
#endif
#ifndef CONFIG_C906_RESHAPE_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RESHAPE, NULL, shl_c906_reshape_fp16);
#endif
#ifndef CONFIG_C906_SPLIT_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SPLIT, NULL, shl_c906_split_fp16);
#endif
#ifndef CONFIG_C906_SUN_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SUB, NULL, shl_c906_sub_fp16);
#endif
#ifndef CONFIG_C906_SUM_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SUM, NULL, shl_c906_sum_stride_fp16);
#endif
#ifndef CONFIG_C906_ABS_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ABS, NULL, shl_c906_abs_f32);
#endif
#ifndef CONFIG_C906_ADD_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ADD, NULL, shl_c906_add_f32);
#endif
#ifndef CONFIG_C906_CLIP_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CLIP, NULL, shl_c906_clip_f32);
#endif
#ifndef CONFIG_C906_CONCAT_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONCAT, NULL, shl_c906_concat_f32);
#endif
#ifndef CONFIG_C906_GLOBAL_AVERAGEPOOL_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_AVGPOOL2D, NULL,
                    shl_c906_global_avgpool2d_f32);
#endif
#ifndef CONFIG_C906_GLOBAL_MAXPOOL_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_MAXPOOL2D, NULL,
                    shl_c906_global_maxpool2d_f32);
#endif
#ifndef CONFIG_C906_LEAKY_RELU_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_LEAKY_RELU, NULL, shl_c906_leaky_relu_f32);
#endif
#ifndef CONFIG_C906_MINIMUM_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MINIMUM, NULL, shl_c906_minimum_f32);
#endif
#ifndef CONFIG_C906_MUL_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MUL, NULL, shl_c906_mul_f32);
#endif
#ifndef CONFIG_C906_PRELU_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_PRELU, NULL, shl_c906_prelu_f32);
#endif
#ifndef CONFIG_C906_RELU_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU, NULL, shl_c906_relu_f32);
#endif
#ifndef CONFIG_C906_RELU1_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU1, NULL, shl_c906_relu1_f32);
#endif
#ifndef CONFIG_C906_RELU6_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU6, NULL, shl_c906_relu6_f32);
#endif
#ifndef CONFIG_C906_SPLIT_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SPLIT, NULL, shl_c906_split_f32);
#endif
#ifndef CONFIG_C906_SUB_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SUB, NULL, shl_c906_sub_f32);
#endif

#ifdef SHL_BUILD_GREF
    shl_register_runtime_callback(CSINN_C906, shl_gref_runtime_callback);
#ifndef CONFIG_GRAPH_REFERENCE_CONVOLUTION_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_gref_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D, shl_gref_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_INT8, CSINN_OP_CONV2D, shl_gref_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_INT8, CSINN_OP_CONV2D_RELU, shl_gref_conv2d_relu);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_GROUP_CONV2D, shl_gref_group_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_GROUP_CONV2D, shl_gref_group_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D, shl_gref_depthwise_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_DEPTHWISE_CONV2D, shl_gref_depthwise_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D, shl_gref_depthwise_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D_RELU,
                        shl_gref_depthwise_conv2d_relu);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CONVOLUTION1D_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV1D, shl_gref_conv1d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV1D, shl_gref_conv1d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV1D, shl_gref_depthwise_conv1d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MAXPOOL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_MAXPOOL2D, shl_gref_maxpool2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_MAXPOOL2D, shl_gref_maxpool2d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_AVERAGEPOOL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_AVGPOOL2D, shl_gref_avgpool2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_AVGPOOL2D, shl_gref_avgpool2d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_FULLYCONNECTED_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_FULLYCONNECTED, shl_gref_fullyconnected);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_DIV_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_DIV, shl_gref_div);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_DIV, shl_gref_div);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ABS_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_ABS, shl_gref_abs);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ADD_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_ADD, shl_gref_add);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CACHE_MATMUL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CACHE_MATMUL, shl_gref_cache_matmul);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CACHE_CONV1D_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CACHE_CONV1D, shl_gref_cache_conv1d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CLIP_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CLIP, shl_gref_clip);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CONCAT_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CONCAT, shl_gref_concat);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GLOBAL_AVERAGEPOOL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_AVGPOOL2D, shl_gref_global_avgpool2d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GLOBAL_MAXPOOL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_MAXPOOL2D, shl_gref_global_maxpool2d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LEAKY_RELU_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_LEAKY_RELU, shl_gref_leaky_relu);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LRN_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_LRN, shl_gref_lrn);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MATMUL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_MATMUL, shl_gref_matmul);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MINIMUM_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_MINIMUM, shl_gref_minimum);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MUL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_MUL, shl_gref_mul);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_PRELU_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_PRELU, shl_gref_prelu);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU, shl_gref_relu);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU1_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU1, shl_gref_relu1);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU6_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU6, shl_gref_relu6);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RESHAPE_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_RESHAPE, shl_gref_reshape);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SPLIT_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_SPLIT, shl_gref_split);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SUB_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_SUB, shl_gref_sub);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SUM_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_SUM, shl_gref_sum);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ABS_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_ABS, shl_gref_abs);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ADD_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_ADD, shl_gref_add);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CLIP_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_CLIP, shl_gref_clip);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CONCAT_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_CONCAT, shl_gref_concat);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GLOBAL_AVERAGEPOOL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_AVGPOOL2D, shl_gref_global_avgpool2d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GLOBAL_MAXPOOL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_MAXPOOL2D, shl_gref_global_maxpool2d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LEAKY_RELU_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_LEAKY_RELU, shl_gref_leaky_relu);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MINIMUM_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_MINIMUM, shl_gref_minimum);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MUL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_MUL, shl_gref_mul);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_PRELU_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_PRELU, shl_gref_prelu);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU, shl_gref_relu);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU1_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU1, shl_gref_relu1);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU6_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU6, shl_gref_relu6);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SPLIT_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_SPLIT, shl_gref_split);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SUB_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_SUB, shl_gref_sub);
#endif
#endif
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D, shl_c906_conv2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_GROUP_CONV2D, shl_c906_conv2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_DEPTHWISE_CONV2D,
                        shl_c906_depthwise_conv2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV1D, shl_c906_conv1d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_MAXPOOL2D, shl_c906_maxpool2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_AVGPOOL2D, shl_c906_avgpool2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_DIV, shl_c906_div_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_ABS, shl_c906_abs_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_ADD, shl_c906_add_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_CLIP, shl_c906_clip_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_CONCAT, shl_c906_concat_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_AVGPOOL2D,
                        shl_c906_global_avgpool2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_MAXPOOL2D,
                        shl_c906_global_maxpool2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_LEAKY_RELU, shl_c906_leaky_relu_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_MINIMUM, shl_c906_minimum_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_MUL, shl_c906_mul_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_PRELU, shl_c906_prelu_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU, shl_c906_relu_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU1, shl_c906_relu1_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU6, shl_c906_relu6_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_SPLIT, shl_c906_split_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_SUB, shl_c906_sub_cap);

    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_c906_conv2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_GROUP_CONV2D, shl_c906_conv2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D,
                        shl_c906_depthwise_conv2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_FULLYCONNECTED, shl_c906_fullyconnected_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV1D, shl_c906_conv1d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV1D,
                        shl_c906_depthwise_conv1d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_MAXPOOL2D, shl_c906_maxpool2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_AVGPOOL2D, shl_c906_avgpool2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_DIV, shl_c906_div_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_ABS, shl_c906_abs_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_ADD, shl_c906_add_cap);
    /* skip cache_matmul and cache_conv1d */
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_CLIP, shl_c906_clip_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_CONCAT, shl_c906_concat_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_AVGPOOL2D,
                        shl_c906_global_avgpool2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_MAXPOOL2D,
                        shl_c906_global_maxpool2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_LEAKY_RELU, shl_c906_leaky_relu_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_LRN, shl_c906_lrn_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_MATMUL, shl_c906_matmul_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_MINIMUM, shl_c906_minimum_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_MUL, shl_c906_mul_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_PRELU, shl_c906_prelu_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU, shl_c906_relu_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU1, shl_c906_relu1_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU6, shl_c906_relu6_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_RESHAPE, shl_c906_reshape_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_SPLIT, shl_c906_split_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_SUB, shl_c906_sub_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_SUM, shl_c906_sum_stride_cap);
}
