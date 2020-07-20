/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.

 */

#ifndef AVX2NEON_H
#error Never use <immintrin.h> directly; include " avx2neon.h" instead.
#endif

#include <arm_neon.h>
#include "typedefs.h"

# define RROTATE(a,n)     (((a)<<(n))|(((a)&0xffffffff)>>(32-(n))))
# define sigma_0(x)       (RROTATE((x),25) ^ RROTATE((x),14) ^ ((x)>>3))
# define sigma_1(x)       (RROTATE((x),15) ^ RROTATE((x),13) ^ ((x)>>10))
# define Sigma_0(x)       (RROTATE((x),30) ^ RROTATE((x),19) ^ RROTATE((x),10))
# define Sigma_1(x)       (RROTATE((x),26) ^ RROTATE((x),21) ^ RROTATE((x),7))

# define Ch(x,y,z)       (((x) & (y)) ^ ((~(x)) & (z)))
# define Maj(x,y,z)      (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

FORCE_INLINE __m128i _mm_sha256rnds2_epu32(__m128i a, __m128i b, __m128i k)
{
    __m128i res;
    uint32_t A[3];
    uint32_t B[3];
    uint32_t C[3];
    uint32_t D[3];
    uint32_t E[3];
    uint32_t F[3];
    uint32_t G[3];
    uint32_t H[3];
    uint32_t K[2];

    A[0] = vgetq_lane_u32(b.vect_u32, 3);
    B[0] = vgetq_lane_u32(b.vect_u32, 2);
    C[0] = vgetq_lane_u32(a.vect_u32, 3);
    D[0] = vgetq_lane_u32(a.vect_u32, 2);
    E[0] = vgetq_lane_u32(b.vect_u32, 1);
    F[0] = vgetq_lane_u32(b.vect_u32, 0);
    G[0] = vgetq_lane_u32(a.vect_u32, 1);
    H[0] = vgetq_lane_u32(a.vect_u32, 0);

    K[0] = vgetq_lane_u32(k.vect_u32, 0);
    K[1] = vgetq_lane_u32(k.vect_u32, 1);

    for (int i = 0; i < 2; i ++) {
        uint32_t T0 = Ch(E[i], F[i], G[i]) ;
        uint32_t T1 = Sigma_1(E[i]) + K[i] + H[i];
        uint32_t T2 = Maj(A[i], B[i], C[i]);

        A[i + 1] = T0 + T1 + T2 + Sigma_0(A[i]);
        B[i + 1] = A[i];
        C[i + 1] = B[i];
        D[i + 1] = C[i];
        E[i + 1] = T0 + T1 + D[i];
        F[i + 1] = E[i];
        G[i + 1] = F[i];
        H[i + 1] = G[i];
    }

    res.vect_u32 = vsetq_lane_u32(F[2], res.vect_u32, 0);
    res.vect_u32 = vsetq_lane_u32(E[2], res.vect_u32, 1);
    res.vect_u32 = vsetq_lane_u32(B[2], res.vect_u32, 2);
    res.vect_u32 = vsetq_lane_u32(A[2], res.vect_u32, 3);

    return res;
}

FORCE_INLINE __m128i _mm_sha256msg1_epu32(__m128i a, __m128i b)
{
    __asm__ __volatile__(
        "sha256su0 %[dst].4S, %[src].4S  \n\t"
        : [dst] "+w" (a)
        : [src] "w" (b)
    );
    return a;
}

FORCE_INLINE __m128i _mm_sha256msg2_epu32(__m128i a, __m128i b)
{
    __m128i res;
    uint32_t A = vgetq_lane_u32(b.vect_u32, 2);
    uint32_t B = vgetq_lane_u32(b.vect_u32, 3);

    uint32_t C = vgetq_lane_u32(a.vect_u32, 0) + sigma_1(A);
    uint32_t D = vgetq_lane_u32(a.vect_u32, 1) + sigma_1(B);
    uint32_t E = vgetq_lane_u32(a.vect_u32, 2) + sigma_1(C);
    uint32_t F = vgetq_lane_u32(a.vect_u32, 3) + sigma_1(D);

    res.vect_u32 = vsetq_lane_u32(C, res.vect_u32, 0);
    res.vect_u32 = vsetq_lane_u32(D, res.vect_u32, 1);
    res.vect_u32 = vsetq_lane_u32(E, res.vect_u32, 2);
    res.vect_u32 = vsetq_lane_u32(F, res.vect_u32, 3);

    return res;
}

typedef enum {
    _MM_SWIZ_REG_NONE = 0,
    _MM_SWIZ_REG_DCBA = 1,
    _MM_SWIZ_REG_CDAB = 2,
    _MM_SWIZ_REG_BADC = 3,
    _MM_SWIZ_REG_AAAA = 4,
    _MM_SWIZ_REG_BBBB = 5,
    _MM_SWIZ_REG_CCCC = 6,
    _MM_SWIZ_REG_DDDD = 7,
    _MM_SWIZ_REG_DACB = 8,
} _MM_SWIZZLE_ENUM;

FORCE_INLINE __m512i _mm512_mask_swizzle_epi32 (__m512i src, __mmask16 k, __m512i v, _MM_SWIZZLE_ENUM s)
{
    __m512i res;
    int32_t tmp_src[16],tmp_v[16],tmp_dst[16];
    int i;
    if(s==_MM_SWIZ_REG_NONE||s==_MM_SWIZ_REG_DCBA){
        return v;
    }

    vst1q_s32(tmp_src,src.vect_s32[0]);
    vst1q_s32(tmp_src+4,src.vect_s32[1]);
    vst1q_s32(tmp_src+8,src.vect_s32[2]);
    vst1q_s32(tmp_src+12,src.vect_s32[3]);
    vst1q_s32(tmp_v,v.vect_s32[0]);
    vst1q_s32(tmp_v+4,v.vect_s32[1]);
    vst1q_s32(tmp_v+8,v.vect_s32[2]);
    vst1q_s32(tmp_v+12,v.vect_s32[3]);

    switch (s)
    {
    case _MM_SWIZ_REG_CDAB:
        for(int j=0;j<8;j++){
            i=j*2;//i := j*64
            if((k>>(j*2))&&0x0001){
                tmp_dst[i] = tmp_v[i+1];//v[i+63:i+32]
            }else{
                tmp_dst[i] = tmp_src[i];
            }
            if((k>>(j*2+1))&&0x0001){
                tmp_dst[i+1] = tmp_v[i];
            }else{
                tmp_dst[i+1] = tmp_v[i+1];
            }
        }
        break;
    case _MM_SWIZ_REG_BADC:
        for(int j=0;j<4;j++){
            i=j*4;//i :=j*128;
            if((k>>(j*4))&&0x0001){
                tmp_dst[i]=tmp_v[i+2];
            }else{
                tmp_dst[i]=tmp_src[i];
            }
            if((k>>(j*4+1))&&0x0001){
                tmp_dst[i+1] = tmp_v[i+1];
            }else{
                tmp_dst[i+i] = tmp_src[i+1];
            }
            if((k>>(j*4+2))&&0x0001){
                tmp_dst[i+2] = tmp_v[i];
            }else{
                tmp_dst[i+2] = tmp_src[i+2];
            }
            if((k>>(j*4+3))&&0x0001){
                tmp_dst[i+3] = tmp_v[i+1];
            }else{
                tmp_dst[i+3] = tmp_src[i+3];
            }
        }
        break;
    case _MM_SWIZ_REG_AAAA:
        for(int j=0;j<4;j++){
            i=j*4;
            if((k>>(j*4))&&0x0001){
                tmp_dst[i] = tmp_v[i];
            }else{
                tmp_dst[i] = tmp_src[i];
            }
            if((k>>(j*4+1))&&0x0001){
                tmp_dst[i+1] = tmp_v[i];
            }else{
                tmp_dst[i+1] = tmp_src[i+1];
            }
            if((k>>(j*4+2))&&0x0001){
                tmp_dst[i+2] = tmp_v[i];
            }else{
                tmp_dst[i+2] = tmp_src[i+2];
            }
            if((k>>(j*4+3))&&0x0001){
                tmp_dst[i+3] = tmp_v[i];
            }else{
                tmp_dst[i+3] = tmp_src[i+3];
            }
        }
        break;
    case _MM_SWIZ_REG_BBBB:
        for(int j=0;j<4;j++){
            i=j*4;
            if((k>>(j*4))&&0x0001){
                tmp_dst[i] = tmp_v[i+1];
            }else{
                tmp_dst[i] = tmp_src[i];
            }
            if((k>>(j*4+1))&&0x0001){
                tmp_dst[i+1+1] = tmp_v[i+1];
            }else{
                tmp_dst[i+1] = tmp_src[i+1];
            }
            if((k>>(j*4+2))&&0x0001){
                tmp_dst[i+2] = tmp_v[i+1];
            }else{
                tmp_dst[i+2] = tmp_src[i+2];
            }
            if((k>>(j*4+3))&&0x0001){
                tmp_dst[i+3] = tmp_v[i+1];
            }else{
                tmp_dst[i+3] = tmp_src[i+3];
            }
        }
        break;
    case _MM_SWIZ_REG_CCCC:
        for(int j=0;j<4;j++){
            i=j*4;
            if((k>>(j*4))&&0x0001){
                tmp_dst[i] = tmp_v[i+2];
            }else{
                tmp_dst[i] = tmp_src[i];
            }
            if((k>>(j*4+1))&&0x0001){
                tmp_dst[i+1] = tmp_v[i+2];
            }else{
                tmp_dst[i+1] = tmp_src[i+1];
            }
            if((k>>(j*4+2))&&0x0001){
                tmp_dst[i+2] = tmp_v[i]+2;
            }else{
                tmp_dst[i+2] = tmp_src[i+2];
            }
            if((k>>(j*4+3))&&0x0001){
                tmp_dst[i+3] = tmp_v[i+2];
            }else{
                tmp_dst[i+3] = tmp_src[i+3];
            }
        }
        break;
    case _MM_SWIZ_REG_DDDD:
        for(int j=0;j<4;j++){
            i=j*4;
            if((k>>(j*4))&&0x0001){
                tmp_dst[i] = tmp_v[i+3];
            }else{
                tmp_dst[i] = tmp_src[i];
            }
            if((k>>(j*4+1))&&0x0001){
                tmp_dst[i+1] = tmp_v[i+3];
            }else{
                tmp_dst[i+1] = tmp_src[i+1];
            }
            if((k>>(j*4+2))&&0x0001){
                tmp_dst[i+2] = tmp_v[i+3];
            }else{
                tmp_dst[i+2] = tmp_src[i+2];
            }
            if((k>>(j*4+3))&&0x0001){
                tmp_dst[i+3] = tmp_v[i+3];
            }else{
                tmp_dst[i+3] = tmp_src[i+3];
            }
        }
        break;
    case _MM_SWIZ_REG_DACB:
        for(int j=0;j<4;j++){
            i=j*4;
            if((k>>(j*4))&&0x0001){
                tmp_dst[i] = tmp_v[i+1];
            }else{
                tmp_dst[i] = tmp_src[i];
            }
            if((k>>(j*4+1))&&0x0001){
                tmp_dst[i+1] = tmp_v[i+2];
            }else{
                tmp_dst[i+1] = tmp_src[i+1];
            }
            if((k>>(j*4+2))&&0x0001){
                tmp_dst[i+2] = tmp_v[i];
            }else{
                tmp_dst[i+2] = tmp_src[i+2];
            }
            if((k>>(j*4+3))&&0x0001){
                tmp_dst[i+3] = tmp_v[i+3];
            }else{
                tmp_dst[i+3] = tmp_src[i+3];
            }
        }
        break;
    default:
        break;
    }
    res.vect_s32[0] = vld1q_s32(tmp_dst);
    res.vect_s32[1] = vld1q_s32(tmp_dst+4);
    res.vect_s32[2] = vld1q_s32(tmp_dst+8);
    res.vect_s32[3] = vld1q_s32(tmp_dst+12);
    return res;
}

FORCE_INLINE __m512i _mm512_mask_shufflelo_epi16 (__m512i src,__mmask32 k,__m512i a,int imm8){
    __m512i dst;
    int16_t tmp_dst[32];
    int i,j,j1,j2;
    for(j=0;j<4;j++){
        switch((imm8>>(j*2))&0x03){
            case 0:
                tmp_dst[j] = vgetq_lane_s16(a.vect_s16[0],0);
                tmp_dst[j+8] = vgetq_lane_s16(a.vect_s16[1],0);
                tmp_dst[j+16] = vgetq_lane_s16(a.vect_s16[2],0);
                tmp_dst[j+24] = vgetq_lane_s16(a.vect_s16[3],0);
            break;
            case 1:
                tmp_dst[j] = vgetq_lane_s16(a.vect_s16[0],1);
                tmp_dst[j+8] = vgetq_lane_s16(a.vect_s16[1],1);
                tmp_dst[j+16] = vgetq_lane_s16(a.vect_s16[2],1);
                tmp_dst[j+24] = vgetq_lane_s16(a.vect_s16[3],1);
            break;
            case 2:
                tmp_dst[j] = vgetq_lane_s16(a.vect_s16[0],2);
                tmp_dst[j+8] = vgetq_lane_s16(a.vect_s16[1],2);
                tmp_dst[j+16] = vgetq_lane_s16(a.vect_s16[2],2);
                tmp_dst[j+24] = vgetq_lane_s16(a.vect_s16[3],2);
            break;
            case 3:
                tmp_dst[j] = vgetq_lane_s16(a.vect_s16[0],3);
                tmp_dst[j+8] = vgetq_lane_s16(a.vect_s16[1],3);
                tmp_dst[j+16] = vgetq_lane_s16(a.vect_s16[2],3);
                tmp_dst[j+24] = vgetq_lane_s16(a.vect_s16[3],3);
            break;
        }
    }
    for(j=4;j<8;j++){
        switch(j){
            case 4:
                tmp_dst[j] = vgetq_lane_s16(a.vect_s16[0],4);
                tmp_dst[j+8] = vgetq_lane_s16(a.vect_s16[1],4);
                tmp_dst[j+16] = vgetq_lane_s16(a.vect_s16[2],4);
                tmp_dst[j+24] = vgetq_lane_s16(a.vect_s16[3],4);
            break;
            case 5:
                tmp_dst[j] = vgetq_lane_s16(a.vect_s16[0],5);
                tmp_dst[j+8] = vgetq_lane_s16(a.vect_s16[1],5);
                tmp_dst[j+16] = vgetq_lane_s16(a.vect_s16[2],5);
                tmp_dst[j+24] = vgetq_lane_s16(a.vect_s16[3],5);
            break;
            case 6:
                tmp_dst[j] = vgetq_lane_s16(a.vect_s16[0],6);
                tmp_dst[j+8] = vgetq_lane_s16(a.vect_s16[1],6);
                tmp_dst[j+16] = vgetq_lane_s16(a.vect_s16[2],6);
                tmp_dst[j+24] = vgetq_lane_s16(a.vect_s16[3],6);
            break;
            case 7:
                tmp_dst[j] = vgetq_lane_s16(a.vect_s16[0],7);
                tmp_dst[j+8] = vgetq_lane_s16(a.vect_s16[1],7);
                tmp_dst[j+16] = vgetq_lane_s16(a.vect_s16[2],7);
                tmp_dst[j+24] = vgetq_lane_s16(a.vect_s16[3],7);
            break;
        }
        
    }

    //dst.vect_s16[0]=vdupq_n_s16(0x00);
    //dst.vect_s16[1]=vdupq_n_s16(0x00);
    //dst.vect_s16[2]=vdupq_n_s16(0x00);
    //dst.vect_s16[3]=vdupq_n_s16(0x00);

    for(j=0;j<32;j++){
        i=j;
        j1=j/8;
        j2=j%8;
        if((k>>j)&0x01){
            dst.vect_s16[j1] = vsetq_lane_s16(tmp_dst[i],dst.vect_s16[j1],j2);
        }else{

            dst.vect_s16[j1] = vsetq_lane_s16(vgetq_lane_s16(src.vect_s16[j1],j2),dst.vect_s16[j1],j2);
        }
    }
    return dst;
}