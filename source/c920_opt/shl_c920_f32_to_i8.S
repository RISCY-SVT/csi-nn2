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

/**************************************************************************************************

    void shl_c920_f32_to_i8(const float *input,
                            int8_t *output,
                            int32_t offset,
                            float *scale,
                            uint32_t length)

    Algorithm works as follows:
        (1) f32 *= 1/scale, convert to i32
        (2) i32 += offset
        (3) i32 narrow to i16 (with saturation)
        (4) i16 narrow to i8 (with saturation)

    register definition:
        a0:         input addr
        a1:         output addr
        a2:         zero_point (offset)
        a3:         scale
        a4:         element length
        fa0:        1/scale
        ft0:        1.0f
        v0-v7:      f32
        v8-v15:     f32,u32/i32
        v16-v19:    u16/i16
        v20-v21:    u8/i8
        v28-v31:    1/scale

    note: vector extension 0.7.1 [support flexible vlen]

 *************************************************************************************************/

    .file           "shl_c920_f32_to_i8.S"
    .section        .text.shl_c920_f32_to_i8, "ax", @progbits
    .align          5
    .global         shl_c920_f32_to_i8
    .type           shl_c920_f32_to_i8, @function

shl_c920_f32_to_i8:
    csrr            t0, vlenb   // t0 = vlen/8
    slli            t1, t0, 1   // t1 = 2 * vlen/8 = vlen/4
    slli            t2, t0, 2   // t2 = 4 * vlen/8 = vlen/2
    flw             fa0, (a3)   // load scale value
    li              t3, 1
    fcvt.s.w        ft0, t3     // ft0 = 1.0f
    fdiv.s          fa0, ft0, fa0  // fa0 = 1/scale
    vsetvli         zero, zero, e32, m4
    vfmv.v.f        v28, fa0    // broadcast 1/scale to v28

.L2:
    bgt             t1, a4, .L1 // if remaining < vlen/4, go to tail processing
    vsetvli         zero, zero, e32, m4
    vle.v           v0, (a0)    // load first vlen/4 floats
    add             a0, a0, t2
    vle.v           v4, (a0)    // load second vlen/4 floats
    add             a0, a0, t2

    sub             a4, a4, t1  // decrement remaining count
    bgt             t1, a4, .L2End // if next iteration would exceed, finish this one

.L2Loop:
    // Process current loaded data
    vfmul.vv        v8, v0, v28    // multiply by 1/scale
    vfmul.vv        v12, v4, v28
    vfcvt.x.f.v     v8, v8         // convert to i32 (round to nearest)
    vfcvt.x.f.v     v12, v12
    
    // Prefetch next data
    vle.v           v0, (a0)
    add             a0, a0, t2
    vle.v           v4, (a0)
    add             a0, a0, t2
    
    // Continue processing
    vadd.vx         v8, v8, a2     // add offset
    vadd.vx         v12, v12, a2
    
    // Narrow to i16 then i8 with saturation (signed)
    vsetvli         zero, zero, e16, m2
    vnclip.vi       v16, v8, 0     // narrow i32->i16 with saturation
    vnclip.vi       v18, v12, 0
    vsetvli         zero, zero, e8, m1
    vnclip.vi       v20, v16, 0    // narrow i16->i8 with saturation
    vnclip.vi       v21, v18, 0
    
    // Store results
    vse.v           v20, (a1)
    add             a1, a1, t0
    vse.v           v21, (a1)
    add             a1, a1, t0

    vsetvli         zero, zero, e32, m4
    sub             a4, a4, t1
    bgt             a4, t1, .L2Loop // continue if remaining > vlen/4

.L2End:
    // Process last prefetched data
    vfmul.vv        v8, v0, v28
    vfmul.vv        v12, v4, v28
    vfcvt.x.f.v     v8, v8
    vfcvt.x.f.v     v12, v12
    vadd.vx         v8, v8, a2
    vadd.vx         v12, v12, a2
    
    vsetvli         zero, zero, e16, m2
    vnclip.vi       v16, v8, 0
    vnclip.vi       v18, v12, 0
    vsetvli         zero, zero, e8, m1
    vnclip.vi       v20, v16, 0
    vnclip.vi       v21, v18, 0
    
    vse.v           v20, (a1)
    add             a1, a1, t0
    vse.v           v21, (a1)
    add             a1, a1, t0

.L1:
    beqz            a4, .End     // exit if no remaining elements

.L1Loop:
    // Process remaining elements (< vlen/4)
    vsetvli         t0, a4, e32, m4
    slli            t2, t0, 2    // t2 = t0 * 4 (bytes for float input)
    vle.v           v0, (a0)
    add             a0, a0, t2
    
    vfmul.vv        v8, v0, v28
    vfcvt.x.f.v     v8, v8
    vadd.vx         v8, v8, a2
    
    vsetvli         t0, a4, e16, m2
    vnclip.vi       v16, v8, 0
    vsetvli         t0, a4, e8, m1
    vnclip.vi       v20, v16, 0
    
    vse.v           v20, (a1)
    add             a1, a1, t0

    sub             a4, a4, t0
    bgtz            a4, .L1Loop

.End:
    ret
    .end
