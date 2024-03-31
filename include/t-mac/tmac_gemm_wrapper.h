#pragma once

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <assert.h>
#include <chrono>
#include <cstdio>
#include <map>
#include <tuple>
#include <mutex>

namespace TMAC {

constexpr size_t kAllocAlignment = 64;
constexpr DLDevice dev = {
  .device_type = kDLCPU,
  .device_id = 0,
};
constexpr DLDataType int_dtype = {
  .code = kDLInt,
  .bits = 8,
  .lanes = 1,
};
constexpr DLDataType float_dtype = {
  .code = kDLFloat,
  .bits = sizeof(T) * 8,
  .lanes = 1,
};

template <typename T, int g = 4>
class TMACGeMMWrapper {
public:
  TMACGeMMWrapper(int n_threads, int act_group_size)
      : _n_threads(n_threads),
        _act_group_size(act_group_size),
        _mod_syslib((*tvm::runtime::Registry::Get("runtime.SystemLib"))()),
        _config_threadpool(tvm::runtime::Registry::Get("runtime.config_threadpool")),
        _allocated(false)
  {
    (*_config_threadpool)(1, _n_threads);
    int num_threads = (*tvm::runtime::Registry::Get("runtime.NumThreads"))();
    LOG(INFO) << "NUM_THREADS: " << num_threads;
  }

  void run(DLTensor* A, DLTensor* scales, DLTensor* B, DLTensor* C, int M, int K, int N, int bits)
  {
    assert(_allocated);

    int64_t qlut_shape[3] = {N, K / g, (1 << g)};
    int64_t luts_shape[3] = {N, K / _act_group_size};

    DLTensor QLUTt = {
      .data = _qlut,
      .device = dev,
      .ndim = 3,
      .dtype = int_dtype,
      .shape = qlut_shape,
    };
    DLTensor LUTSt = {
      .data = _lut_scales,
      .device = dev,
      .ndim = 2,
      .dtype = float_dtype,
      .shape = luts_shape,
    };
    DLTensor LUTBt = {
      .data = _lut_biases,
      .device = dev,
      .ndim = 2,
      .dtype = float_dtype,
      .shape = luts_shape,
    };

    tvm::runtime::PackedFunc pf = get_function({M, K, N, bits, 0});
    tvm::runtime::PackedFunc qf = get_function({M, K, N, bits, 1});

    // Currently the parallelism of preprocessor is disabled due to large thread communication overhead in `benchmark.cc`.
    // But according to profiled results of python side, the overhead is not that large and the best NUM_THREADS should be 4.
    // TODO: Find out the reason for the high communication overhead in C++ side.
    pf(B, &LUTSt, &LUTBt, &QLUTt);
    qf(A, &QLUTt, scales, &LUTSt, &LUTBt, C);
  }

  // Activation (B): NxK
  // Parallelism is disabled for preprocessor
  // Should only be called in main thread
  void llama_cpp_init(void* B, int M, int K, int N, int bits)
  {
    assert(_allocated);

    DLTensor Bt = {
      .data = B,
    };
    DLTensor QLUTt = {
      .data = _qlut,
    };
    DLTensor LUTSt = {
      .data = _lut_scales,
    };
    DLTensor LUTBt = {
      .data = _lut_biases,
    };

    tvm::runtime::PackedFunc pf = get_function({M, K, N, bits, 0});
    pf(B, &LUTSt, &LUTBt, &QLUTt);
  }

  // Activation (B): NxK, Weights (A): MxK
  // This is the task of only one thread for GeMM: N x K x (M x num_threads)
  // Please split the blocks in llama.cpp and pass the right ptr for scales, A and C
  void llama_cpp_compute(void* A, void* scales, void* C, int M, int K, int N, int bits)
  {
    assert(_allocated);

    DLTensor At = {
      .data = A,
    };
    DLTensor St = {
      .data = scales,
    };
    DLTensor Ct = {
      .data = C,
    };
    DLTensor QLUTt = {
      .data = _qlut,
    };
    DLTensor LUTSt = {
      .data = _lut_scales,
    };
    DLTensor LUTBt = {
      .data = _lut_biases,
    };

    tvm::runtime::PackedFunc qf = get_function({M, K, N, bits});
    qf(&At, &QLUTt, &St, &LUTSt, &LUTBt, &Ct);
  }

  // Should only be called in main thread
  void set_workspace(int maxK, int maxN)
  {
    posix_memalign(&_qlut, kAllocAlignment, maxN * maxK / g * (1 << g) * sizeof(int8_t));
    posix_memalign(&_lut_scales, kAllocAlignment, maxN * maxK / _act_group_size * sizeof(T));
    posix_memalign(&_lut_biases, kAllocAlignment, maxN * maxK / _act_group_size * sizeof(T));
    _allocated = true;
  }

  ~TMACGeMMWrapper()
  {
    free(_qlut);
    free(_lut_scales);
    free(_lut_biases);
  }

  using _fkey = std::tuple<int, int, int, int, int>;

  tvm::runtime::PackedFunc get_function(_fkey key)
  {
    std::lock_guard<std::mutex> lock(_m);
    auto iter = _fcache.find(key);
    tvm::runtime::PackedFunc f;
    if (iter == _fcache.end()) {
      if (std::get<4>(key) != 0) {
        f = _mod_syslib.GetFunction(
          "qgemm_lut_"
            + "t" + std::to_string(_n_threads) + "_"
            + "int8_"
            + "m" + std::to_string(std::get<0>(key) * std::get<3>(key)) + "_"
            + "k" + std::to_string(std::get<1>(key)) + "_"
            + "n" + std::to_string(std::get<2>(key)) + "_"
            + "b" + std::to_string(std::get<3>(key))
        );
      } else {
        f = _mod_syslib.GetFunction(
          "preprocessor_"
            + "t" + std::to_string(_n_threads) + "_"
            + "int8_"
            + "m" + std::to_string(std::get<0>(key) * std::get<3>(key)) + "_"
            + "k" + std::to_string(std::get<1>(key)) + "_"
            + "n" + std::to_string(std::get<2>(key)) + "_"
            + "b" + std::to_string(std::get<3>(key))
        );
      }
      _fcache[key] = f;
      return f;
    } else {
      return iter->second;
    }
  }

private:
  tvm::runtime::Module _mod_syslib;
  std::map<_fkey, tvm::runtime::PackedFunc> _fcache;
  const tvm::runtime::PackedFunc* _config_threadpool;

  int _n_threads;
  int _act_group_size;

  // workspace ptrs
  void* _qlut;
  void* _lut_scales;
  void* _lut_biases;

  bool _allocated;
  std::mutex _m;
};

} // namespace TMAC
