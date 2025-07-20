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

    void shl_c920_i8_to_f32(const int8_t *input,
                            float *output,
                            int32_t offset,
                            float *scale,
                            uint32_t length)

    Algorithm works as follows:
        (1) i8 sign-extend to i16
        (2) i16 vwsub(offset) to i32
        (3) i32 convert to f32
        (4) f32 *= scale

    register definition:
        a0: input addr
        a1: output addr
        a2: offset
        a3: scale point
        a4: element length

    note: vector extension 0.7.1 [support flexible vlen]

 *************************************************************************************************/

    .file           "shl_c920_i8_to_f32.S"
    .section        .text.shl_c920_i8_to_f32, "ax", @progbits
    .align          5
    .global         shl_c920_i8_to_f32
    .type           shl_c920_i8_to_f32, @function

shl_c920_i8_to_f32:
    csrr            t0, vlenb   // t0 = vlen/8
    slli            t1, t0, 1   // t1 = 2 * vlen/8 = vlen/4
    flw             fa0, (a3)   // load scale value
    slli            t2, t0, 2   // t2 = 4 * vlen/8 = vlen/2
    vsetvli         zero, zero, e32, m4
    vfmv.v.f        v28, fa0    // broadcast scale to v28

.L2:
    bgt             t1, a4, .L1 // if remaining < vlen/4, go to tail processing
    vsetvli         zero, zero, e8, m1
    vle.v           v0, (a0)    // load first vlen/8 bytes
    add             a0, a0, t0
    vle.v           v1, (a0)    // load second vlen/8 bytes
    add             a0, a0, t0

    sub             a4, a4, t1  // decrement remaining count
    bgt             t1, a4, .L2End // if next iteration would exceed, finish this one

.L2Loop:
    // Process current loaded data (sign-extend for int8_t)
    vwadd.vx        v2, v0, zero // sign-extend i8 to i16
    vwadd.vx        v4, v1, zero

    // Prefetch next data
    vle.v           v0, (a0)
    add             a0, a0, t0
    vle.v           v1, (a0)
    add             a0, a0, t0

    // Continue processing
    vsetvli         zero, zero, e16, m2
    vwsub.vx        v8, v2, a2   // subtract offset, widen to i32
    vwsub.vx        v12, v4, a2
    vsetvli         zero, zero, e32, m4
    vfcvt.f.x.v     v16, v8     // convert i32 to f32
    vfcvt.f.x.v     v20, v12
    vfmul.vv        v8, v16, v28 // multiply by scale
    vfmul.vv        v12, v20, v28
    vse.v           v8, (a1)     // store results
    add             a1, a1, t2
    vse.v           v12, (a1)
    add             a1, a1, t2

    vsetvli         zero, zero, e8, m1
    sub             a4, a4, t1
    bgt             a4, t1, .L2Loop // continue if remaining > vlen/4

.L2End:
    // Process last prefetched data
    vwadd.vx        v2, v0, zero
    vwadd.vx        v4, v1, zero

    vsetvli         zero, zero, e16, m2
    vwsub.vx        v8, v2, a2
    vwsub.vx        v12, v4, a2
    vsetvli         zero, zero, e32, m4
    vfcvt.f.x.v     v16, v8
    vfcvt.f.x.v     v20, v12

    vfmul.vv        v8, v16, v28
    vfmul.vv        v12, v20, v28

    vse.v           v8, (a1)
    add             a1, a1, t2
    vse.v           v12, (a1)
    add             a1, a1, t2

.L1:
    beqz            a4, .End     // exit if no remaining elements

.L1Loop:
    // Process remaining elements (< vlen/4)
    vsetvli         t0, a4, e8, m1
    slli            t1, t0, 2    // t1 = t0 * 4 (bytes for float output)
    vle.v           v0, (a0)
    add             a0, a0, t0
    vwadd.vx        v2, v0, zero // sign-extend i8 to i16

    vsetvli         t0, a4, e16, m2
    vwsub.vx        v4, v2, a2
    vsetvli         t0, a4, e32, m4
    vfcvt.f.x.v     v8, v4
    vfmul.vv        v4, v8, v28
    vse.v           v4, (a1)
    add             a1, a1, t1

    sub             a4, a4, t0
    bgtz            a4, .L1Loop

.End:
    ret
    .end
