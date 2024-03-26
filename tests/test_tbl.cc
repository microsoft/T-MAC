#include "tbl.cc"
#include <stdio.h>
#include <random>

extern "C" {
tbl_g4_int8_float_update(true, 16, 2, 16, false)
}

int main() {
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    float cbits[256];
    tbl_float_reset(256, cbits);
    int8_t lut[32][16];
    std::uniform_int_distribution<int16_t> uni1(-127, 127);
    printf("lut\n");
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 16; j++) {
            lut[i][j] = uni1(gen);
            printf("\t%d", lut[i][j]);
        }
        printf("\n");
    }
    uint8_t A[4096];
    std::uniform_int_distribution<uint16_t> uni2(0U, 255U);
    printf("A\n");
    for (int i = 0; i < 4096; i++) {
        A[i] = uni2(gen);
        printf("%u ", A[i]);
    }
    printf("\nscales\n");
    float scales[128];
    std::normal_distribution<float> dis(0, 1);
    for (int i = 0; i < 128; i++) {
        scales[i] = dis(gen);
        printf("%f ", scales[i]);
    }
    printf("\nlut_scales\n");
    float lut_scales[2];
    float lut_biases[2];
    for (int i = 0; i < 2; i++) {
        lut_scales[i] = dis(gen);
        printf("%f ", lut_scales[i]);
    }
    printf("\nlut_biases\n");
    for (int i = 0; i < 2; i++) {
        lut_biases[i] = dis(gen);
        printf("%f ", lut_biases[i]);
    }
    printf("\n");

    tbl_g4_int8_float_update_strue_k16_b2_ak16_fafalse(256, cbits, lut[0],  A,        scales, lut_scales    , lut_biases    );
    for (int i = 0; i < 256; i++) {
        printf("%f ", cbits[i]);
    }
    printf("\n");
    tbl_g4_int8_float_update_strue_k16_b2_ak16_fafalse(256, cbits, lut[16], A + 2048, scales, lut_scales + 1, lut_biases + 1);
    for (int i = 0; i < 256; i++) {
        printf("%f ", cbits[i]);
    }
}
