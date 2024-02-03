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

namespace TMAC {

static constexpr size_t kAllocAlignment = 64;

template <typename T, int g = 4>
class TMACGeMMWrapper {
public:
  TMACGeMMWrapper(int n_threads, int act_group_size)
      : _n_threads(n_threads),
        _act_group_size(act_group_size),
        _mod_syslib((*tvm::runtime::Registry::Get("runtime.SystemLib"))()),
        _config_threadpool(tvm::runtime::Registry::Get("runtime.config_threadpool"))
  {
    (*_config_threadpool)(1, _n_threads);
    int num_threads = (*tvm::runtime::Registry::Get("runtime.NumThreads"))();
    LOG(INFO) << "NUM_THREADS: " << num_threads;
  }

  void run(DLTensor* A, DLTensor* scales, DLTensor* B, DLTensor* C, int M, int K, int N, int bits)
  {
    int64_t qlut_shape[3] = {N, K / g, (1 << g)};
    int64_t luts_shape[3] = {N, K / _act_group_size};
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

    tvm::runtime::PackedFunc pf = get_function({0, K, N, 0});
    tvm::runtime::PackedFunc qf = get_function({M, K, N, bits});

    // Currently the parallelism of preprocessor is disabled due to large thread communication overhead in `benchmark.cc`.
    // But according to profiled results of python side, the overhead is not that large and the best NUM_THREADS should be 4.
    // TODO: Find out the reason for the high communication overhead in C++ side.
    pf(B, &LUTSt, &LUTBt, &QLUTt);
    qf(A, &QLUTt, scales, &LUTSt, &LUTBt, C);
  }

  void set_workspace(int maxK, int maxN)
  {
    posix_memalign(&_qlut, kAllocAlignment, maxN * maxK / g * (1 << g) * sizeof(int8_t));
    posix_memalign(&_lut_scales, kAllocAlignment, maxN * maxK / _act_group_size * sizeof(T));
    posix_memalign(&_lut_biases, kAllocAlignment, maxN * maxK / _act_group_size * sizeof(T));
  }

  ~TMACGeMMWrapper()
  {
    free(_qlut);
    free(_lut_scales);
    free(_lut_biases);
  }

  using _fkey = std::tuple<int, int, int, int>;

  tvm::runtime::PackedFunc get_function(_fkey key)
  {
    auto iter = _fcache.find(key);
    tvm::runtime::PackedFunc f;
    if (iter == _fcache.end()) {
      if (std::get<0>(key) != 0) {
        f = _mod_syslib.GetFunction(
          "qgemm_lut_"
            + std::to_string(std::get<0>(key) * std::get<3>(key)) + "_"
            + std::to_string(std::get<1>(key)) + "_"
            + std::to_string(std::get<2>(key)) + "_"
            + std::to_string(_n_threads) + "_"
            + "int8_"
            + std::to_string(std::get<3>(key))
        );
      } else {
        f = _mod_syslib.GetFunction(
          "preprocessor_"
            + std::to_string(std::get<1>(key)) + "_"
            + std::to_string(std::get<2>(key)) + "_"
            + std::to_string(_n_threads) + "_"
            + "int8"
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
};

} // namespace TMAC
