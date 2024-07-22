#include "stdint.h"
#ifdef __cplusplus
extern "C"
#endif
 int32_t qgemm_lut_t1_int8_m1024_k4096_n1_b4(void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C);
#ifdef __cplusplus
extern "C"
#endif
 int32_t preprocessor_t1_int8_m16384_k4096_n1_b4(void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT);
#ifdef __cplusplus
extern "C"
#endif
 int32_t qgemm_lut_t1_int8_m256_k4096_n1_b4(void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C);
#ifdef __cplusplus
extern "C"
#endif
 int32_t preprocessor_t1_int8_m44032_k4096_n1_b4(void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT);
#ifdef __cplusplus
extern "C"
#endif
 int32_t qgemm_lut_t1_int8_m256_k11008_n1_b4(void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C);
#ifdef __cplusplus
extern "C"
#endif
 int32_t preprocessor_t1_int8_m16384_k11008_n1_b4(void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT);inline int qgemm_lut_int8(int m, int k, int n, int b, void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C) {

    if (m == 1024 && k == 4096 && n == 1 && b == 4) return qgemm_lut_t1_int8_m1024_k4096_n1_b4(A, LUT, Scales, LUT_Scales, LUT_Biases, C);

    if (m == 256 && k == 4096 && n == 1 && b == 4) return qgemm_lut_t1_int8_m256_k4096_n1_b4(A, LUT, Scales, LUT_Scales, LUT_Biases, C);

    if (m == 256 && k == 11008 && n == 1 && b == 4) return qgemm_lut_t1_int8_m256_k11008_n1_b4(A, LUT, Scales, LUT_Scales, LUT_Biases, C);

    return -1;
}
inline int preprocessor_int8(int m, int k, int n, int b, void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT) {

    if (m == 16384 && k == 4096 && n == 1 && b == 4) return preprocessor_t1_int8_m16384_k4096_n1_b4(B, LUT_Scales, LUT_Biases, QLUT);

    if (m == 44032 && k == 4096 && n == 1 && b == 4) return preprocessor_t1_int8_m44032_k4096_n1_b4(B, LUT_Scales, LUT_Biases, QLUT);

    if (m == 16384 && k == 11008 && n == 1 && b == 4) return preprocessor_t1_int8_m16384_k11008_n1_b4(B, LUT_Scales, LUT_Biases, QLUT);

    return -1;
}
