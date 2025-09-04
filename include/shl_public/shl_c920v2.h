// csi-nn2/include/shl_public/shl_c920v2.h

#ifndef INCLUDE_SHL_C920V2_H_
#define INCLUDE_SHL_C920V2_H_

#include "csi_nn.h"
#include "shl_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

void *shl_c920v2_f32_to_input_dtype(uint32_t index, float *data, struct csinn_session *sess);
float *shl_c920v2_output_to_f32_dtype(uint32_t index, void *data, struct csinn_session *sess);

int shl_c920v2_detect_yolov5_postprocess(struct csinn_tensor **input_tensors,
                                         struct shl_yolov5_box *out,
                                         struct shl_yolov5_params *params);
int shl_c920v2_yolox_preprocess(struct csinn_tensor *input, struct csinn_tensor *output);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_SHL_C920V2_H_
