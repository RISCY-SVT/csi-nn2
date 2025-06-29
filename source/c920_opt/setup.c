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

#include "c920/c920.h"
#include "c920/cap.h"
#include "c920/perf.h"

#define C920_OP_PATTERN_MAX 40
static struct shl_cb_table shl_c920_cb_table[C920_OP_PATTERN_MAX];

void shl_c920_reg_op(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *init,
                     void *exec, void *est, void *cap, void *perf)
{
    static int i = 0;
    if (i >= C920_OP_PATTERN_MAX) {
        shl_debug_error("C920 callback length is greater than C920_OP_PATTERN_MAX!\n");
    }
    shl_c920_cb_table[i].shl_cb_key = op_name * CSINN_DTYPE_SIZE + dtype;
    shl_c920_cb_table[i].shl_cb_value.init = init;
    shl_c920_cb_table[i].shl_cb_value.exec = exec;
    shl_c920_cb_table[i].shl_cb_value.est = est;
    shl_c920_cb_table[i].shl_cb_value.caps = cap;
    shl_c920_cb_table[i].shl_cb_value.perf = perf;
    i++;
}

struct csinn_callback *shl_cb_map_rvv(int op, int dtype);
struct csinn_callback *shl_cb_map_c920(int op, int dtype)
{
    struct csinn_callback *cb = NULL;
    for (int i = 0; i < C920_OP_PATTERN_MAX; i++) {
        if (shl_c920_cb_table[i].shl_cb_key == (op * CSINN_DTYPE_SIZE + dtype)) {
            cb = &(shl_c920_cb_table[i].shl_cb_value);
            break;
        }
    }
    if ((cb == NULL) || (cb->est == NULL && (cb->init == NULL || cb->exec == NULL))) {
        cb = shl_cb_map_rvv(op, dtype);
    }
    return cb;
}

int shl_c920_set_packn_layout(struct csinn_session *sess, bool packn_layout)
{
    struct shl_gref_target_data *gref_td = sess->td;
    struct shl_c920_option *c920_option = gref_td->cpu_option;
    c920_option->base.use_packn_layout = packn_layout;
    return CSINN_TRUE;
}

struct shl_c920_option *shl_c920_get_graph_option(struct csinn_session *sess)
{
    struct shl_gref_target_data *gref_td = sess->td;
    if (gref_td) {
        return (struct shl_c920_option *)(gref_td->cpu_option);
    } else {
        return NULL;
    }
}

void shl_c920_session_init(struct csinn_session *sess)
{
    struct shl_c920_option *c920_option = shl_mem_alloc(sizeof(struct shl_c920_option));
    struct shl_ref_graph *graph = shl_mem_alloc(sizeof(struct shl_ref_graph));
    struct shl_gref_target_data *target_data = shl_mem_alloc(sizeof(struct shl_gref_target_data));
    target_data->graph = graph;
    c920_option->base.use_packn_layout = 1;  // c920 set use_packn_layout true default
    target_data->cpu_option = c920_option;
    sess->td = target_data;
    shl_c920_set_binary_model_op_init(sess, false);
    sess->base_layout = CSINN_LAYOUT_NCHW;
}

void shl_c920_session_deinit(struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    shl_mem_free(graph->input);
    shl_mem_free(graph->output);
    shl_mem_free(graph->layer);
    struct shl_c920_option *c920_option = shl_c920_get_graph_option(sess);
    if (c920_option) {
        shl_mem_free(c920_option);
    }
    shl_mem_free(graph);
    shl_mem_free(sess->td);
    shl_mem_free(sess->input);
    shl_mem_free(sess->output);
}

static int pre_init(struct shl_node *node)
{
    /* base has same address with params */
    struct csinn_params_base *params = node->data;

    int (*func)();

    int org_rm = params->sess->base_run_mode;
    params->sess->base_run_mode = CSINN_RM_LAYER;
    struct csinn_callback *cb = shl_gref_best_callback(node);

    params->sess->base_run_mode = org_rm;

    return CSINN_TRUE;
}

static int init_op(struct shl_node *node)
{
    /* base has same address with params */
    struct csinn_params_base *params = node->data;
    struct csinn_callback *cb = params->cb;

    if (cb->init != NULL) {
        if (shl_gref_call_layer_func(cb->init, node) != CSINN_TRUE) {
            return CSINN_FALSE;
        }
    }

    return CSINN_TRUE;
}

static void sess_op_init(struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);

    // pre init, find best callback
    for (int i = 0; i < graph->layer_index; i++) {
        struct shl_node *n = graph->layer[i];
        if (n->type >= 0 && n->type < CSINN_OP_SIZE) {
            pre_init(n);
        } else {
            shl_debug_error("Unknown layer\n");
            return;
        }
    }

    // different layout
    bool use_packn = true;
    for (int i = 0; i < graph->layer_index; i++) {
        struct csinn_params_base *curr_params = graph->layer[i]->data;
        if (curr_params->api == CSINN_TVMGEN) {
            use_packn = false;
            break;
        }
    }
    shl_c920_set_packn_layout(sess, use_packn);

    // call init
    for (int i = 0; i < graph->layer_index; i++) {
        struct shl_node *n = graph->layer[i];
        if (n->type >= 0 && n->type < CSINN_OP_SIZE) {
            init_op(n);
        } else {
            shl_debug_error("Unknown layer\n");
            return;
        }
    }
}

/**
 * @brief Sets up a CSINN session for the C920 backend, initializing graph nodes and optionally saving a binary model.
 *
 * This function performs the following operations:
 * - Retrieves the reference graph associated with the session.
 * - Initializes session operations.
 * - Iterates through all layers in the graph to update the reference count for input and output tensors.
 * - Increments the reference count for all output tensors in the graph.
 * - If the session is configured to save the model (in float16 or float32), it:
 *   - Determines the binary model file path.
 *   - Opens the file and writes the binary model header.
 *   - Dumps the graph structure and graph info sections into the file at aligned offsets.
 *   - Saves section information metadata.
 *   - Closes the file.
 *
 * @param[in,out] sess Pointer to the CSINN session structure to be set up.
 */
void shl_c920_session_setup(struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    struct shl_node *n;
    FILE *b;
    char *path;
    int bm_offset = 8192;
    struct shl_binary_model_section_info *sinfo;
    bool save_binary_model = false;

    if (sess->model.save_mode == CSINN_SAVE_AND_RUN || sess->model.save_mode == CSINN_SAVE_ONLY) {
        if (sess->base_dtype == CSINN_DTYPE_FLOAT16 || sess->base_dtype == CSINN_DTYPE_FLOAT32) {
            save_binary_model = true;
        } else {
            shl_debug_warning("Unsupport to save this dtype binary model yet\n");
        }
    }

    struct shl_ref_graph *ggraph = graph;

    sess_op_init(sess);

    for (int i = 0; i < ggraph->layer_index; i++) {
        n = ggraph->layer[i];
        for (int j = 0; j < n->in_num; j++) {
            if (n->in[j]->ref_count_init > 0) {
                n->in[j]->ref_count_init++;
            }
        }
        if (n->type != CSINN_SUBGRAPH) {
            for (int k = 0; k < n->out_num; k++) {
                n->out[k]->ref_count_init++;
            }
        }
    }

    for (int i = 0; i < ggraph->output_num; i++) {
        ggraph->output[i]->ref_count_init++;
    }

    if (save_binary_model) {
        if (sess->model.bm_path == NULL) {
            path = "shl.hhb.bm";
        } else {
            path = sess->model.bm_path;
        }
        b = fopen(path, "wb");
        shl_dump_bm_header(b);

        /* TODO: start from more */
        bm_offset = 8192;
        fseek(b, bm_offset, SEEK_SET);
        sinfo = shl_mem_alloc(sizeof(struct shl_binary_model_section_info));

        /* only dump top(global) graph, unsupport subgraph */
        fseek(b, bm_offset, SEEK_SET);
        int ggraph_size = shl_dump_bm_graph_struct_section(b, ggraph);
        sinfo->sections[0].graph_offset = bm_offset / 4096;
        sinfo->sections[0].graph_size = ggraph_size;
        bm_offset = shl_gref_size_align(bm_offset + ggraph_size, 4096);

        fseek(b, bm_offset, SEEK_SET);
        int info_size = shl_dump_bm_graph_info_section(b, sess);
        sinfo->sections[0].info_offset = bm_offset / 4096;
        sinfo->sections[0].info_size = info_size;
        bm_offset = shl_gref_size_align(bm_offset + info_size, 4096);

        /* save section info */
        sinfo->section_num = 2;
        fseek(b, 4096, SEEK_SET);
        shl_dump_bm_section_info(b, sinfo);
        fclose(b);
    }
}

/* use tensor name to match same */
static void merge_output(struct shl_ref_graph *graph, struct csinn_session *sess)
{
    /* match graph output */
    for (int i = 0; i < graph->output_num; i++) {
        struct shl_node *gnode = graph->output[i];
        char *sname = gnode->name;
        for (int j = 0; j < graph->layer_index; j++) {
            struct shl_node *node = graph->layer[j];
            for (int m = 0; m < node->out_num; m++) {
                if (strcmp(node->name, sname) == 0) {
                    /* TODO: free graph output node */
                    graph->output[i] = node->out[m];
                    break;
                }
            }
        }
    }

    for (int i = 0; i < sess->input_num; i++) {
        /* TODO: free sess output node */
        sess->input[i] = graph->input[i]->data;
    }

    for (int i = 0; i < sess->output_num; i++) {
        /* TODO: free sess output node */
        sess->output[i] = graph->output[i]->data;
    }
}

static void graph_match_session(struct shl_ref_graph *graph, struct csinn_session *sess)
{
    struct shl_gref_target_data *td = sess->td;
    td->graph = graph;

    for (int i = 0; i < graph->layer_index; i++) {
        struct shl_node *n = graph->layer[i];
        /* fix op callback, skip subgraph */
        if (n->type < CSINN_OP_SIZE) {
            struct csinn_params_base *base = n->data;
            base->sess = sess;
            struct csinn_tensor *input = n->in[0]->data;

            int org_rm = base->sess->base_run_mode;
            base->sess->base_run_mode = CSINN_RM_LAYER;
            shl_op_callback_map(base, n->type, input->dtype);
            base->sess->base_run_mode = org_rm;
        }
    }
}

/**
 * @brief Loads a binary model into the given CSINN session for the C920 backend.
 *
 * This function initializes the session with a binary model by:
 *   - Retrieving the base address of the binary model from the session.
 *   - Accessing the section information structure within the binary model.
 *   - Allocating and loading the reference graph structure from the binary model.
 *   - Matching the loaded graph with the session.
 *   - Merging output information between the graph and the session.
 *   - Initializing operation handlers for the binary model.
 *   - Performing additional session operation initialization.
 *
 * @param[in,out] sess Pointer to the CSINN session structure to be initialized.
 * @return CSINN_TRUE on successful loading and initialization.
 */
int shl_c920_load_binary_model(struct csinn_session *sess)
{
    char *bm_base = sess->model.bm_addr;
    struct shl_binary_model_section_info *sinfo =
        (struct shl_binary_model_section_info *)(bm_base + 4096);
    struct shl_ref_graph *ggraph = shl_mem_alloc(sizeof(struct shl_ref_graph));
    shl_bm_graph_struct_load(
        ggraph, (struct shl_ref_graph *)(bm_base + sinfo->sections[0].graph_offset * 4096));
    graph_match_session(ggraph, sess);
    merge_output(ggraph, sess);
    shl_c920_set_binary_model_op_init(sess, true);
    sess_op_init(sess);

    return CSINN_TRUE;
}

/**
 * @brief Retrieves and processes the output tensor from a session for the C920 backend.
 *
 * This function fetches the output tensor at the specified index from the session and, if the
 * session is configured to use the packn layout, converts the tensor's layout from NC1xC0 format
 * (such as NC1DHWC0, NC1HWC0, NC1WC0, or NC1C0) to the standard NDArray layout in-place,
 * depending on the tensor's data type (float32, float16, or int8).
 *
 * After conversion, the function updates the output tensor's dimension and layout metadata
 * to reflect the new layout. If the data type is unsupported, an error is reported.
 *
 * @param index   The index of the output tensor to retrieve.
 * @param output  Pointer to the output tensor structure to be processed.
 * @param sess    Pointer to the session structure containing the output tensors and options.
 * @return        CSINN_TRUE on success, or CSINN_UNSUPPORT_DTYPE if the data type is unsupported.
 */
int shl_c920_get_output(int index, struct csinn_tensor *output, struct csinn_session *sess)
{
    struct csinn_tensor *sess_output = sess->output[index];
    struct shl_c920_option *option = shl_c920_get_graph_option(sess);
    if (option && option->base.use_packn_layout) {
        if (output->layout == CSINN_LAYOUT_NC1DHWC0 || output->layout == CSINN_LAYOUT_NC1HWC0 ||
            output->layout == CSINN_LAYOUT_NC1WC0 || output->layout == CSINN_LAYOUT_NC1C0) {
            if (output->dtype == CSINN_DTYPE_FLOAT32) {
                shl_rvv_tensor_nc1xc0_to_ndarray_inplace_fp32(output);
            } else if (output->dtype == CSINN_DTYPE_FLOAT16) {
                shl_rvv_tensor_nc1xc0_to_ndarray_inplace_fp16(output);
            } else if (output->dtype == CSINN_DTYPE_INT8) {
                shl_rvv_tensor_nc1xc0_to_ndarray_inplace_int8(output);
            } else {
                shl_debug_error("c920 get output unsupported dtype: %d\n", output->dtype);
                return CSINN_UNSUPPORT_DTYPE;
            }

            /* TODO: unset sess_output, alloc another data space and copy to output */
            sess_output->dim[1] =
                sess_output->dim[1] * sess_output->dim[sess_output->dim_count - 1];
            sess_output->dim[sess_output->dim_count - 1] = 0;
            sess_output->dim_count = sess_output->dim_count - 1;
            if (sess_output->layout == CSINN_LAYOUT_NC1DHWC0) {
                sess_output->layout = CSINN_LAYOUT_NCDHW;
            } else if (sess_output->layout == CSINN_LAYOUT_NC1HWC0) {
                sess_output->layout = CSINN_LAYOUT_NCHW;
            } else if (sess_output->layout == CSINN_LAYOUT_NC1WC0) {
                sess_output->layout = CSINN_LAYOUT_NCW;
            } else if (sess_output->layout == CSINN_LAYOUT_NC1C0) {
                sess_output->layout = CSINN_LAYOUT_NC;
            }
        }
    }

    return CSINN_TRUE;
}

/**
 * @brief Returns the appropriate runtime callback function pointer for the given API identifier.
 *
 * This function maps a given API identifier to its corresponding runtime callback function
 * for the C920 backend. For certain API values, it delegates the callback resolution to the
 * generic reference implementation.
 *
 * @param api The API identifier (e.g., CSINN_SESSION_INIT, CSINN_SESSION_DEINIT, etc.).
 * @return Pointer to the corresponding callback function, or NULL if not found.
 *
 * The following API identifiers are handled:
 * - CSINN_SESSION_INIT:        Returns pointer to shl_c920_session_init.
 * - CSINN_SESSION_DEINIT:      Returns pointer to shl_c920_session_deinit.
 * - CSINN_SESSION_SETUP:       Returns pointer to shl_c920_session_setup.
 * - CSINN_LOAD_BG:             Returns pointer to shl_c920_load_binary_model.
 * - CSINN_GET_OUTPUT:          Returns pointer to shl_c920_get_output.
 * - CSINN_SESSION_RUN, CSINN_UPDATE_INPUT, CSINN_UPDATE_OUTPUT,
 *   CSINN_SET_INPUT_NUMBER, CSINN_SET_OUTPUT_NUMBER, CSINN_SET_INPUT,
 *   CSINN_SET_OUTPUT, CSINN_GET_INPUT, CSINN_TENSOR_ENTRY:
 *      Delegates to shl_gref_runtime_callback(api).
 *
 * If the API identifier is not recognized, logs a debug message and returns NULL.
 */
void *shl_c920_runtime_callback(int api)
{
    switch (api) {
        case CSINN_SESSION_INIT:                // 0
            return shl_c920_session_init;
        case CSINN_SESSION_DEINIT:              // 1  
            return shl_c920_session_deinit;
        case CSINN_SESSION_SETUP:               // 2
            return shl_c920_session_setup;
        case CSINN_LOAD_BG:                     // 15
            return shl_c920_load_binary_model;
        case CSINN_GET_OUTPUT:                  // 13
            return shl_c920_get_output;
        case CSINN_SESSION_RUN:                 // 3
        case CSINN_UPDATE_INPUT:                // 4
        case CSINN_UPDATE_OUTPUT:               // 5
        case CSINN_SET_INPUT_NUMBER:            // 6
        case CSINN_SET_OUTPUT_NUMBER:           // 7
        case CSINN_GET_INPUT_NUMBER:            // 8
        case CSINN_GET_OUTPUT_NUMBER:           // 9
        // These APIs are used to manage session inputs and outputs.
        // They are not specific to C920 and can be handled by the generic reference implementation.
        case CSINN_SET_INPUT:                   // 10
        case CSINN_SET_OUTPUT:                  // 11
        case CSINN_GET_INPUT:                   // 12
            // For these APIs, we delegate to the generic reference implementation
            // as they are not specific to C920.
            // This allows for a consistent interface across different backends.
            // The actual implementation will be handled by shl_gref_runtime_callback.
            // This is useful for operations that are common across different backends.
        case CSINN_TENSOR_ENTRY:                // 14   
            return shl_gref_runtime_callback(api);
        default:
            shl_debug_error("%s: Invalid API Num: %d. Returning NULL.\n", __func__, api);
            return NULL;
    }
}

void shl_target_init_c920()
{
#ifndef CONFIG_C920_CONVOLUTION_FP32_DISABLED
    shl_c920_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D, shl_c920_conv2d_init_fp32, NULL,
                    shl_gref_conv2d, shl_c920_conv2d_cap, shl_c920_conv2d_perf);
    shl_c920_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GROUP_CONV2D, shl_c920_conv2d_init_fp32, NULL,
                    shl_gref_group_conv2d, shl_c920_conv2d_cap, shl_c920_conv2d_perf);
#endif
#ifndef CONFIG_C920_CONVOLUTION_FP16_DISABLED
    shl_c920_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_c920_conv2d_init_fp16, NULL,
                    shl_gref_conv2d, shl_c920_conv2d_cap, shl_c920_conv2d_perf);
    shl_c920_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GROUP_CONV2D, shl_c920_conv2d_init_fp16, NULL,
                    shl_gref_group_conv2d, shl_c920_conv2d_cap, shl_c920_conv2d_perf);
#endif
#ifndef CONFIG_C920_FULLYCONNECTED_FP32_DISABLED
    shl_c920_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_FULLYCONNECTED, shl_c920_fullyconnected_init_fp32,
                    NULL, shl_gref_fullyconnected, shl_c920_fullyconnected_cap,
                    shl_c920_fullyconnected_perf);
#endif
#ifndef CONFIG_C920_FULLYCONNECTED_FP16_DISABLED
    shl_c920_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_FULLYCONNECTED, shl_c920_fullyconnected_init_fp16,
                    NULL, shl_gref_fullyconnected, shl_c920_fullyconnected_cap,
                    shl_c920_fullyconnected_perf);
#endif
#ifndef CONFIG_C920_MATMUL_FP32_DISABLED
    shl_c920_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MATMUL, shl_c920_matmul_init_fp32, NULL,
                    shl_gref_matmul, shl_c920_matmul_cap, shl_c920_matmul_perf);
#endif
#ifndef CONFIG_C920_MATMUL_FP16_DISABLED
    shl_c920_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MATMUL, shl_c920_matmul_init_fp16, NULL,
                    shl_gref_matmul, shl_c920_matmul_cap, shl_c920_matmul_perf);
#endif
    shl_register_op_callback(CSINN_C920, shl_cb_map_c920);
    shl_register_runtime_callback(CSINN_C920, shl_c920_runtime_callback);
}
