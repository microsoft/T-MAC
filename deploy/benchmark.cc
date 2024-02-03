#include "t-mac/tmac_gemm_wrapper.h"
#include "arm_neon.h"

int main(int argc, char** argv)
{
  assert(argc == 8 && "Usage: ./benchmark <M:int> <K:int> <N:int> <warmup:int> <repeat:int> <n_threads:int> <bm:int = 1024>");

  int M = std::atoi(argv[1]);
  int K = std::atoi(argv[2]);
  int N = std::atoi(argv[3]);
  int warmup = std::atoi(argv[4]);
  int repeat = std::atoi(argv[5]);
  int n_threads = std::atoi(argv[6]);
  int bm = std::atoi(argv[7]);
  int g = 4;
  int group_size = 128;
  int act_group_size = 64;
  int bits = 4;

  TMAC::TMACGeMMWrapper<float16_t> gemm(n_threads, act_group_size);
  gemm.set_workspace(K, N);
  DLTensor* A;
  DLTensor* scales;
  DLTensor* B;
  DLTensor* C;

  int64_t A_shape[3] = {M * bits / bm, K / g, bm / 2};
  int64_t scales_shape[3] = {M * bits / bm, K / group_size, bm / bits};
  int64_t B_shape[3] = {N, K};
  int64_t C_shape[3] = {N, M};
  TVMArrayAlloc(B_shape, 2, kDLFloat, 16, 1, kDLCPU, 0, &B);
  TVMArrayAlloc(A_shape, 3, kDLUInt, 8, 1, kDLCPU, 0, &A);
  TVMArrayAlloc(scales_shape, 3, kDLFloat, 16, 1, kDLCPU, 0, &scales);
  TVMArrayAlloc(C_shape, 2, kDLFloat, 16, 1, kDLCPU, 0, &C);

  for (int i = 0; i < warmup; i++) {
    gemm.run(A, scales, B, C, M, K, N, bits);
  }

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < repeat; i++) {
    gemm.run(A, scales, B, C, M, K, N, bits);
  }
  auto end = std::chrono::system_clock::now();
  double lat = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / repeat / 1000;

  // double lat = 10000;
  // for (int r = 0; r < 100; r++) {
  //   auto start = std::chrono::system_clock::now();
  //   for (int i = 0; i < repeat; i++) {
  //     gemm.run(A, scales, B, C, M, K, N, bits);
  //   }
  //   auto end = std::chrono::system_clock::now();
  //   lat = std::min(lat, static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / repeat / 1000);
  // }

  LOG(INFO) << "Avg: " << lat << " ms";

  TVMArrayFree(A);
  TVMArrayFree(scales);
  TVMArrayFree(B);
  TVMArrayFree(C);
}
