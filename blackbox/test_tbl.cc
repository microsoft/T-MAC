#include <arm_neon.h>
#include <chrono>
#include <iostream>

#define UNROLL 10000
#define ITER 1000000
#define K 8


void test_tbl() {
    std::cout << __FUNCTION__ << std::endl;
    uint8_t result[K * 16];
    uint8x16_t vec_d[K];

    for (int i = 0; i < K; i++) {
        vec_d[i] = vcombine_u8(vcreate_u8(std::rand()), vcreate_u8(std::rand()));
    }

    uint8x16_t vec_m = vcombine_u8(vcreate_u8(std::rand()), vcreate_u8(std::rand()));

    for (int i = 0; i < ITER; i++) {
        for (int j = 0; j < UNROLL; j++) {
            for (int k = 0; k < K; k++) {
                vec_d[k] = vqtbl1q_u8(vec_m, vec_d[k]);
                vec_d[k] = vaddq_u8(vec_d[k], vec_m);
            }
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITER; i++) {
#pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#pragma unroll
            for (int k = 0; k < K; k++) {
                vec_d[k] = vqtbl1q_u8(vec_m, vec_d[k]);
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < K; k++) {
        vst1q_u8(result + k * 16, vec_d[k]);
        std::cout << (int)result[k * 16] << std::endl;
    }

    std::cout << (((int64_t)UNROLL * ITER * K) / (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())) << std::endl;
}


void test_add() {
    std::cout << __FUNCTION__ << std::endl;
    uint8_t result[K * 16];
    uint8x16_t vec_d[K];

    for (int i = 0; i < K; i++) {
        vec_d[i] = vcombine_u8(vcreate_u8(std::rand()), vcreate_u8(std::rand()));
    }

    uint8x16_t vec_m = vcombine_u8(vcreate_u8(std::rand()), vcreate_u8(std::rand()));

    for (int i = 0; i < ITER; i++) {
        for (int j = 0; j < UNROLL; j++) {
            for (int k = 0; k < K; k++) {
                vec_d[k] = vaddq_u8(vec_d[k], vec_m);
            }
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < (int64_t)ITER * UNROLL; i++) {
#pragma unroll
        for (int k = 0; k < K; k++) {
            vec_d[k] = vaddq_u8(vec_d[k], vec_m);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < K; k++) {
        vst1q_u8(result + k * 16, vec_d[k]);
        std::cout << (int)result[k * 16] << std::endl;
    }

    std::cout << (((int64_t)UNROLL * ITER * K) / (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())) << std::endl;
}

void test_and() {
    std::cout << __FUNCTION__ << std::endl;
    uint8_t result[K * 16];
    uint8x16_t vec_d[K];

    for (int i = 0; i < K; i++) {
        vec_d[i] = vcombine_u8(vcreate_u8(std::rand()), vcreate_u8(std::rand()));
    }

    uint8x16_t vec_m = vcombine_u8(vcreate_u8(std::rand()), vcreate_u8(std::rand()));

    for (int i = 0; i < ITER; i++) {
        for (int j = 0; j < UNROLL; j++) {
            for (int k = 0; k < K; k++) {
                vec_d[k] = vandq_u8(vec_d[k], vec_m);
            }
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < (int64_t)ITER * UNROLL; i++) {
#pragma unroll
        for (int k = 0; k < K; k++) {
            vec_d[k] = veorq_u8(vec_d[k], vec_m);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < K; k++) {
        vst1q_u8(result + k * 16, vec_d[k]);
        std::cout << (int)result[k * 16] << std::endl;
    }

    std::cout << (((int64_t)UNROLL * ITER * K) / (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())) << std::endl;
}


void test_zip() {
    std::cout << __FUNCTION__ << std::endl;
    uint8_t result[K * 16];
    uint8x16_t vec_d[K];

    for (int i = 0; i < K; i++) {
        vec_d[i] = vcombine_u8(vcreate_u8(std::rand()), vcreate_u8(std::rand()));
    }

    uint8x16_t vec_m = vcombine_u8(vcreate_u8(std::rand()), vcreate_u8(std::rand()));

    for (int i = 0; i < ITER; i++) {
        for (int j = 0; j < UNROLL; j++) {
            for (int k = 0; k < K; k++) {
                vec_d[k] = vzip1q_u8(vec_d[k], vec_m);
            }
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < (int64_t)ITER * UNROLL; i++) {
#pragma unroll
        for (int k = 0; k < K; k++) {
            vec_d[k] = vzip1q_u8(vec_d[k], vec_m);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < K; k++) {
        vst1q_u8(result + k * 16, vec_d[k]);
        std::cout << (int)result[k * 16] << std::endl;
    }

    std::cout << (((int64_t)UNROLL * ITER * K) / (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())) << std::endl;
}


void test_fadd() {
    std::cout << __FUNCTION__ << std::endl;
    uint8_t result[K * 16];
    uint8x16_t vec_d[K];

    for (int i = 0; i < K; i++) {
        vec_d[i] = vcombine_u8(vcreate_u8(std::rand()), vcreate_u8(std::rand()));
    }

    uint8x16_t vec_m = vcombine_u8(vcreate_u8(std::rand()), vcreate_u8(std::rand()));

    for (int i = 0; i < ITER; i++) {
        for (int j = 0; j < UNROLL; j++) {
            for (int k = 0; k < K; k++) {
                vec_d[k] = vreinterpretq_u8_f16(
                    vreinterpretq_f16_u8(vec_d[k]) +
                    vreinterpretq_f16_u8(vec_m)
                );
            }
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITER; i++) {
#pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#pragma unroll
            for (int k = 0; k < K; k++) {
                vec_d[k] = vreinterpretq_u8_f16(
                    vreinterpretq_f16_u8(vec_d[k]) +
                    vreinterpretq_f16_u8(vec_m)
                );
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < K; k++) {
        vst1q_u8(result + k * 16, vec_d[k]);
        std::cout << (int)result[k * 16] << std::endl;
    }

    std::cout << (((int64_t)UNROLL * ITER * K) / (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())) << std::endl;
}


void test_fma() {
    std::cout << __FUNCTION__ << std::endl;
    uint8_t result[K * 16];
    uint8x16_t vec_d[K];

    for (int i = 0; i < K; i++) {
        vec_d[i] = vcombine_u8(vcreate_u8(std::rand()), vcreate_u8(std::rand()));
    }

    uint8x16_t vec_m = vcombine_u8(vcreate_u8(std::rand()), vcreate_u8(std::rand()));

    for (int i = 0; i < ITER; i++) {
        for (int j = 0; j < UNROLL; j++) {
            for (int k = 0; k < K; k++) {
                vec_d[k] = vreinterpretq_u8_f16(
                    vfmaq_f16(
                        vreinterpretq_f16_u8(vec_d[k]),
                        vreinterpretq_f16_u8(vec_m),
                        vreinterpretq_f16_u8(vec_d[k])
                    )
                );
            }
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITER; i++) {
#pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#pragma unroll
            for (int k = 0; k < K; k++) {
                vec_d[k] = vreinterpretq_u8_f16(
                    vfmaq_f16(
                        vreinterpretq_f16_u8(vec_d[k]),
                        vreinterpretq_f16_u8(vec_m),
                        vreinterpretq_f16_u8(vec_d[k])
                    )
                );
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < K; k++) {
        vst1q_u8(result + k * 16, vec_d[k]);
        std::cout << (int)result[k * 16] << std::endl;
    }

    std::cout << (((int64_t)UNROLL * ITER * K) / (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())) << std::endl;
}


int main() {
    test_tbl();
    test_add();
    test_and();
    test_fadd();
    test_fma();
    test_zip();
}