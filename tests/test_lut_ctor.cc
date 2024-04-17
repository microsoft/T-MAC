#include "lut_ctor.cc"

lut_ctor(0, 4)

int main() {
    int8_t qlut[8][16];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 16; j++) {
            qlut[i][j] = 0;
        }
    }
    float_type b[32];
    for (int i = 0; i < 32; i++) {
        b[i] = i;
    }
    float_type lut_scales = 0.0;
    float_type lut_biases = 0.0;

    partial_max_reset(&lut_scales);
    partial_max_g4_int8_k8(&lut_scales, b);

    printf("lut_scales: %f\n", lut_scales);

    lut_ctor_g4_int8_k0_b4(32, qlut[0], b, &lut_scales, &lut_biases);
    printf("lut_scales: %f\n", lut_scales);
    printf("lut_biases: %f\n", lut_biases);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 16; j++) {
            printf("%d ", qlut[i][j]);
        }
        printf("\n");
    }
}
