#include <arm_neon.h>
#include <iostream>

template <int N>
struct SignedHalvingAdder {
    SignedHalvingAdder<N / 2> adder;
    int8x16_t lhs;

    inline void push(int8x16_t v, int k) {
        if (k < N / 2) {
            adder.push(v, k);
            if (k == N / 2 - 1) {
                lhs = adder.get();
            }
        } else {
            adder.push(v, k - N / 2);
            if (k == N - 1) {
                lhs = vrhaddq_s8(lhs, adder.get());
            }
        }
    }

    inline int8x16_t get() {
        return lhs;
    }

    inline int16x8_t get_low() {
        return vmovl_s8(vget_low_s8(lhs));
    }

    inline int16x8_t get_high() {
        return vmovl_high_s8(lhs);
    }
};

template <>
struct SignedHalvingAdder<2> {
    int8x16_t lhs;

    inline void push(int8x16_t v, int k) {
        if (k == 0) {
            lhs = v;
        } else {
            lhs = vrhaddq_s8(lhs, v);
        }
    }

    inline int8x16_t get() {
        return lhs;
    }

    inline int16x8_t get_low() {
        return vmovl_s8(vget_low_s8(lhs));
    }

    inline int16x8_t get_high() {
        return vmovl_high_s8(lhs);
    }
};

struct SignedLongAdder {
    int16x8_t lhs_low;
    int16x8_t lhs_high;
    int8x16_t lhs;

    inline void push(int8x16_t v, int k) {
        if (k == 0) {
            lhs = v;
        } else {
            lhs_low = vaddl_s8(vget_low_s8(lhs), vget_low_s8(v));
            lhs_high = vaddl_high_s8(lhs, v);
        }
    }

    inline int16x8_t get_low() {
        return lhs_low;
    }

    inline int16x8_t get_high() {
        return lhs_high;
    }
};

template <int N>
struct SignedWideningAdder {
    SignedLongAdder adder;
    int16x8_t lhs_low;
    int16x8_t lhs_high;

    inline void push(int8x16_t v, int k) {
        if (k % 2 == 0) {
            adder.push(v, 0);
        } else {
            adder.push(v, 1);
            if (k == 1) {
                lhs_low = adder.get_low();
                lhs_high = adder.get_high();
            } else {
                lhs_low += adder.get_low();
                lhs_high += adder.get_high();
            }
        }
    }

    inline int16x8_t get_low() {
        return lhs_low;
    }

    inline int16x8_t get_high() {
        return lhs_high;
    }
};

int main() {
    SignedHalvingAdder<16> adder;
    SignedWideningAdder<16> adder_ref;

    int8x16_t a[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        a[i] = vcombine_s8(vcreate_s8(std::rand()), vcreate_s8(std::rand()));
    }

#pragma unroll
    for (int i = 0; i < 16; i++) {
        adder.push(a[i], i);
    }
#pragma unroll
    for (int i = 0; i < 16; i++) {
        adder_ref.push(a[i], i);
    }
    int16x8_t res_low = adder.get_low() - 1;
    int16x8_t res_ref = adder_ref.get_low();

    std::cout << 16 * (int)vgetq_lane_s16(res_low, 0) << std::endl;
    std::cout << 16 * (int)vgetq_lane_s16(res_low, 1) << std::endl;
    std::cout << 16 * (int)vgetq_lane_s16(res_low, 2) << std::endl;
    std::cout << 16 * (int)vgetq_lane_s16(res_low, 3) << std::endl;
    std::cout << 16 * (int)vgetq_lane_s16(res_low, 4) << std::endl;
    std::cout << 16 * (int)vgetq_lane_s16(res_low, 5) << std::endl;
    std::cout << 16 * (int)vgetq_lane_s16(res_low, 6) << std::endl;
    std::cout << 16 * (int)vgetq_lane_s16(res_low, 7) << std::endl;

    std::cout << std::endl;

    std::cout << vgetq_lane_s16(res_ref, 0) << std::endl;
    std::cout << vgetq_lane_s16(res_ref, 1) << std::endl;
    std::cout << vgetq_lane_s16(res_ref, 2) << std::endl;
    std::cout << vgetq_lane_s16(res_ref, 3) << std::endl;
    std::cout << vgetq_lane_s16(res_ref, 4) << std::endl;
    std::cout << vgetq_lane_s16(res_ref, 5) << std::endl;
    std::cout << vgetq_lane_s16(res_ref, 6) << std::endl;
    std::cout << vgetq_lane_s16(res_ref, 7) << std::endl;
}