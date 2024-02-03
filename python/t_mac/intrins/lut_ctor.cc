#include <arm_neon.h>
#include <algorithm>


#define vaddvq_f16(v) \
    ((v)[0] + (v)[1] + (v)[2] + (v)[3] + (v)[4] + (v)[5] + (v)[6] + (v)[7])

template <int K>
struct mylog2 {
    enum {
        value = 1 + mylog2<K / 2>::value
    };
};

template <>
struct mylog2<1> {
    enum {
        value = 0
    };
};

// Current implementation requires (K * 4) == act_group_size and K >= 8
// s0 = -1, s1 = 1
template <int K, bool FastAggregation = true>
int32_t lut_ctor_g4_int8_impl(int8_t* qlut, float16_t* b, float16_t* lut_scales, float16_t* lut_biases) {
    float16x8_t vec_lut[K / 8][16];
    float16_t scales = 0.0;
    float16_t biases = 0.0;
#pragma unroll
    for (int k = 0; k < K / 8; ++k) {
        float16x8x4_t vec_bs = vld4q_f16(b + k * 32);

#pragma unroll
        for (int g = 1; g < 16; g += 2) {
            vec_lut[k][g] = vec_bs.val[0];
            if (g & 0b0010) {
                vec_lut[k][g] = vec_lut[k][g] + vec_bs.val[1];
            } else {
                vec_lut[k][g] = vec_lut[k][g] - vec_bs.val[1];
            }
            if (g & 0b0100) {
                vec_lut[k][g] = vec_lut[k][g] + vec_bs.val[2];
            } else {
                vec_lut[k][g] = vec_lut[k][g] - vec_bs.val[2];
            }
            if (g & 0b1000) {
                vec_lut[k][g] = vec_lut[k][g] + vec_bs.val[3];
            } else {
                vec_lut[k][g] = vec_lut[k][g] - vec_bs.val[3];
            }
        }
#pragma unroll
        for (int g = 0; g < 16; g += 2) {
            vec_lut[k][g] = -vec_lut[k][15 - g];
        }

        // Current implementation only works for bits=4
        // TODO: set biases for bits=1, bits=2 and bits=3
        biases += vaddvq_f16(vec_lut[k][0]);

        float16x8_t abssum = vabsq_f16(vec_bs.val[0]) + vabsq_f16(vec_bs.val[1]) + vabsq_f16(vec_bs.val[2]) + vabsq_f16(vec_bs.val[3]);
        scales = std::max(scales, vmaxvq_f16(abssum));
    }

    scales = scales / 127;
    float16_t t_scales = scales ? 1.0 / scales : 0.0;

#pragma unroll
    for (int k = 0; k < K / 8; ++k) {
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            vec_lut[k][g] = vmulq_n_f16(vec_lut[k][g], t_scales);
        }

        int8x8_t vec_qlut[16];
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            vec_qlut[g] = vqmovn_s16(vcvtnq_s16_f16(vec_lut[k][g]));
        }

#pragma unroll
        for (int g = 0; g < 16; ++g) {
            vst1_lane_s8(qlut + k * 8 * 16          + g, vec_qlut[g], 0);
        }
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            vst1_lane_s8(qlut + k * 8 * 16 + 16     + g, vec_qlut[g], 1);
        }
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            vst1_lane_s8(qlut + k * 8 * 16 + 16 * 2 + g, vec_qlut[g], 2);
        }
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            vst1_lane_s8(qlut + k * 8 * 16 + 16 * 3 + g, vec_qlut[g], 3);
        }
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            vst1_lane_s8(qlut + k * 8 * 16 + 16 * 4 + g, vec_qlut[g], 4);
        }
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            vst1_lane_s8(qlut + k * 8 * 16 + 16 * 5 + g, vec_qlut[g], 5);
        }
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            vst1_lane_s8(qlut + k * 8 * 16 + 16 * 6 + g, vec_qlut[g], 6);
        }
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            vst1_lane_s8(qlut + k * 8 * 16 + 16 * 7 + g, vec_qlut[g], 7);
        }
    }

    // https://arxiv.org/pdf/2106.10860.pdf
    // Fast aggregation bias: -ActK * log2(ActK) / 4
    // 15 = (1/2 + 1 + 2 + 4) / (1/2)
    // TODO: set fast aggregation biase for bits=1, bits=2 and bits=3
    if (FastAggregation) {
        biases -= scales * (mylog2<K>::value / 4 * K * 15);
        scales = scales * K;
    }

    *lut_scales = scales;
    *lut_biases = biases;
    return 0;
}

// TODO: Add API to toggle FastAggregation
#define lut_ctor(k)                                                                                           \
    int32_t lut_ctor_g4_int8_##k(int8_t* qlut, float16_t* b, float16_t* lut_scales, float16_t* lut_biases) {  \
        return lut_ctor_g4_int8_impl<k>(qlut, b, lut_scales, lut_biases);                                     \
    }

#ifdef __cplusplus
extern "C" {
#endif

lut_ctor(8)
lut_ctor(16)

#ifdef __cplusplus
}
#endif
