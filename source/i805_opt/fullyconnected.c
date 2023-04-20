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

#include "i805_function.h"
#include "shl_i805.h"

// contraints: input->dim[0] = 1
int shl_i805_fullyconnected_q7(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *weights, struct csinn_tensor *bias,
                               struct csinn_fc_params *params)
{
    q7_t *input_data = (q7_t *)input->data;
    q7_t *weight_data = (q7_t *)weights->data;
    q7_t *bias_data = (q7_t *)bias->data;
    q7_t *output_data = (q7_t *)output->data;

    csky_vdsp2_fully_connected_q7(input_data, weight_data, input->dim[1], weights->dim[0],
                                  bias->qinfo->shift, output->qinfo->shift, bias_data, output_data);
    return CSINN_TRUE;
}

int shl_i805_fullyconnected_q15(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *weights, struct csinn_tensor *bias,
                                struct csinn_fc_params *params)
{
    q15_t *input_data = (q15_t *)input->data;
    q15_t *weight_data = (q15_t *)weights->data;
    q15_t *bias_data = (q15_t *)bias->data;
    q15_t *output_data = (q15_t *)output->data;

    csky_vdsp2_fully_connected_q15(input_data, weight_data, input->dim[1], weights->dim[0],
                                   bias->qinfo->shift, output->qinfo->shift, bias_data,
                                   output_data);
    return CSINN_TRUE;
}

int shl_i805_fullyconnected_init_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_tensor *weights, struct csinn_tensor *bias,
                                    struct csinn_fc_params *params)
{
    float real_scale = input->qinfo->scale * weights->qinfo->scale / output->qinfo->scale;
    shl_quantize_multiplier(real_scale, &output->qinfo->multiplier, &output->qinfo->shift);
    struct csinn_callback *cb = params->base.cb;
    cb->exec = shl_i805_fullyconnected_u8;
    return CSINN_TRUE;
}

int shl_i805_fullyconnected_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *weights, struct csinn_tensor *bias,
                               struct csinn_fc_params *params)
{
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *weights_data = (uint8_t *)weights->data;
    int32_t *bias_data = (int32_t *)bias->data;
    uint8_t *output_data = (uint8_t *)output->data;

    int32_t in_nodes = input->dim[1];  // i.e. in_nodes = weights->dim[1]
    int32_t out_nodes = weights->dim[0];

    shl_i805_fullyconnected_opt_u8(input_data, weights_data, bias_data, output_data, in_nodes,
                                   out_nodes, input->qinfo->zero_point, weights->qinfo->zero_point,
                                   output->qinfo->zero_point, output->qinfo->multiplier,
                                   -output->qinfo->shift);

    return CSINN_FALSE;
}
