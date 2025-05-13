#include <iostream>                                           
#include "cutlass/gemm/device/gemm.h"                         // 引入cutlass头文件
// #include <typeinfo>
// #include <cxxabi.h>


using ColumnMajor = cutlass::layout::ColumnMajor;             
using RowMajor    = cutlass::layout::RowMajor;                

using ElementOutput = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;




  using CutlassGemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 512/8, 32>,
      cutlass::gemm::GemmShape<64, 256//8, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2
      >;


 
void generate_tensor_2D(cutlass::half_t *ptr, int i_M, int i_N){        // 二维矩阵填充函数（此处全部填充1）
    for(int i = 0; i < i_M; i++){
        for(int j = 0; j < i_N; j++){
            *(ptr + i*i_N + j ) = cutlass::half_t(1.0);
        }
    }
}

 
int main(int argc, const char *arg[]) {

    cudaSetDevice(0);  // 使用第X个GPU
 


    int M = 4096;           
    int N = 27648/8;           
    int K = 5120;            
 
    int lda = K;
    int ldb = K;
    int ldc = N;
 
    cutlass::half_t alpha = cutlass::half_t(1.0);      //alpha
    cutlass::half_t beta = cutlass::half_t(0.0);       //beta
 
    cutlass::half_t *A;               
    cutlass::half_t *B;               
    cutlass::half_t *C;               
 
    size_t A_mem_size = sizeof(cutlass::half_t) * M * K; 
    size_t B_mem_size = sizeof(cutlass::half_t) * K * N; 
    size_t C_mem_size = sizeof(cutlass::half_t) * M * N; 
 
    A = (cutlass::half_t*)malloc(A_mem_size);  
    B = (cutlass::half_t*)malloc(B_mem_size);  
    C = (cutlass::half_t*)malloc(C_mem_size);  
 
    generate_tensor_2D(A, M, K);     
    generate_tensor_2D(B, K, N);     
    generate_tensor_2D(C, M, N);     

 
    cutlass::half_t *d_A;            
    cutlass::half_t *d_B;            
    cutlass::half_t *d_C;            
 
    cudaMalloc((void**)&d_A, A_mem_size); 
    cudaMalloc((void**)&d_B, B_mem_size); 
    cudaMalloc((void**)&d_C, C_mem_size); 
 
    cudaMemcpy(d_A, A, A_mem_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, B, B_mem_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_C, C, C_mem_size, cudaMemcpyHostToDevice); 
 
    CutlassGemm gemm_operator;                  



    // int status;
    // char *realname = abi::__cxa_demangle(typeid(gemm_operator).name(), 0, 0, &status);
    // std::cout << realname << std::endl;
    // free(realname);  

    CutlassGemm::Arguments args({M, N, K},      // Gemm Problem dimensions
                                {d_A, lda},     // source matrix A
                                {d_B, ldb},     // source matrix B
                                {d_C, ldc},     // source matrix C
                                {d_C, ldc},     // destination matrix D
                                {alpha, beta}); // alpha & beta
    gemm_operator(args); //运行Gemm
 
    // cudaMemcpy(C, d_C, C_mem_size, cudaMemcpyDeviceToHost);  
    // std::cout << C[0] << std::endl;                          
    // std::cout << C[M * N - 1] << std::endl;                  
 
    return 0;
}   