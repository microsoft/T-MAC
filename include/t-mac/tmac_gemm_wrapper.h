#pragma once

#ifdef TMAC_USE_TVM_THREADPOOL
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#else
#include "t-mac/kernels.h"
#include "dmlc/logging.h"
#endif

#include <assert.h>
#include <chrono>
#include <cstdio>
#include <map>
#include <tuple>
#include <mutex>

#include "t-mac/INIReader.h"

namespace TMAC {

constexpr size_t kAllocAlignment = 64;

struct TMACGeMMConfig {
  int bm;
  int simd_n_in;
  int simd_n_out;
  int kfactor;
  int group_size;
  int lut_scales_size;
  int scales_size;
  int n_tile_num;
};

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

inline std::string get_kcfg_file(const std::string& kcfg_file)
{
  if (kcfg_file.empty()) {
    if (const char* kcfg_file_cstr = getenv("TMAC_KCFG_FILE")) {
      return kcfg_file_cstr;
    } else {
#ifdef TMAC_KCFG_FILE
      return STR(TMAC_KCFG_FILE);
#else
      LOG(FATAL) << "Please set TMAC_KCFG_FILE environment variable";
      return "";
#endif
    }
  } else {
    return kcfg_file;
  }
}

inline std::string get_library_file(const std::string& library_file)
{
  if (library_file.empty()) {
    if (const char* library_file_cstr = getenv("TMAC_KERNELS_LIBRARY")) {
      return library_file_cstr;
    } else {
#ifdef TMAC_KERNELS_LIBRARY
      return STR(TMAC_KERNELS_LIBRARY);
#else
      LOG(FATAL) << "Please set TMAC_KERNELS_LIBRARY environment variable";
      return "";
#endif
    }
  } else {
    return library_file;
  }
}

#undef STR
#undef QUOTE

template <typename T, int g = 4>
class TMACGeMMWrapper {
public:
  TMACGeMMWrapper(int n_threads, int act_group_size, const std::string& kcfg_file, const std::string& library_file)
      : _n_threads(0),
        _act_group_size(act_group_size),
        _allocated(false),
        _reader(get_kcfg_file(kcfg_file))
  {
#ifdef TMAC_USE_TVM_THREADPOOL
#ifdef TMAC_USE_SYSLIB
    _mod_lib = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
#else
    LOG(INFO) << "Loading kernels from: " << get_library_file(library_file);
    _mod_lib = tvm::runtime::Module::LoadFromFile(get_library_file(library_file));
#endif
    _config_threadpool = tvm::runtime::Registry::Get("runtime.config_threadpool");
    set_num_threads(n_threads);
#endif
  }

  TMACGeMMWrapper() : TMACGeMMWrapper(1, 32, "", "") {}

  void set_num_threads(int n_threads)
  {
#ifdef TMAC_USE_TVM_THREADPOOL
    if (n_threads != _n_threads) {
      _n_threads = n_threads;
      (*_config_threadpool)(1, _n_threads);
      int num_threads = (*tvm::runtime::Registry::Get("runtime.NumThreads"))();
      LOG(INFO) << "NUM_THREADS: " << num_threads;
    }
#endif
  }

#ifdef TMAC_USE_TVM_THREADPOOL
  void run(DLTensor* A, DLTensor* scales, DLTensor* B, DLTensor* C, int M, int K, int N, int bits)
  {
    assert(_allocated);

    int64_t qlut_shape[3] = {N, K / g, (1 << g)};
    int64_t luts_shape[3] = {N, K / _act_group_size};

    const DLDevice cpu_dev = {
      /* .device_type = */ kDLCPU,
      /* .device_id   = */ 0,
    };
    const DLDataType int_dtype = {
      /* .code  = */ kDLInt,
      /* .bits  = */ 8,
      /* .lanes = */ 1,
    };
    const DLDataType float_dtype = {
      /* .code  = */ kDLFloat,
      /* .bits  = */ sizeof(T) * 8,
      /* .lanes = */ 1,
    };

    DLTensor QLUTt = {
      /* .data   = */ _qlut,
      /* .device = */ cpu_dev,
      /* .ndim   = */ 3,
      /* .dtype  = */ int_dtype,
      /* .shape  = */ qlut_shape,
    };
    DLTensor LUTSt = {
      /* .data   = */ _lut_scales,
      /* .device = */ cpu_dev,
      /* .ndim   = */ 2,
      /* .dtype  = */ float_dtype,
      /* .shape  = */ luts_shape,
    };
    DLTensor LUTBt = {
      /* .data   = */ _lut_biases,
      /* .device = */ cpu_dev,
      /* .ndim   = */ 2,
      /* .dtype  = */ float_dtype,
      /* .shape  = */ luts_shape,
    };

    tvm::runtime::PackedFunc pf = get_function({M, K, N, bits, 0});
    tvm::runtime::PackedFunc qf = get_function({M, K, N, bits, 1});

    // Currently the parallelism of preprocessor is disabled due to large thread communication overhead in `benchmark.cc`.
    // But according to profiled results of python side, the overhead is not that large and the best NUM_THREADS should be 4.
    // TODO: Find out the reason for the high communication overhead in C++ side.
    pf(B, &LUTSt, &LUTBt, &QLUTt);
    qf(A, &QLUTt, scales, &LUTSt, &LUTBt, C);
  }
#endif

  // Activation (B): NxK
  // Parallelism is disabled for preprocessor
  // Should only be called in main thread
  void llama_cpp_init(void* B, void* qlut, void* lut_scales, void* lut_biases, int M, int K, int N, int bits)
  {
#ifdef TMAC_USE_TVM_THREADPOOL
    DLTensor Bt = {
      /* .data = */ B,
    };
    DLTensor QLUTt = {
      /* .data = */ qlut,
    };
    DLTensor LUTSt = {
      /* .data = */ lut_scales,
    };
    DLTensor LUTBt = {
      /* .data = */ lut_biases,
    };

    tvm::runtime::PackedFunc pf = get_function({M, K, N, bits, 0});
    pf(&Bt, &LUTSt, &LUTBt, &QLUTt);
#else
    int ret = preprocessor_int8(M * bits, K, N, bits, B, lut_scales, lut_biases, qlut);
    DCHECK(ret == 0) << "error calling preprocessor (m=" << M << ", k=" << K << ", n=" << N << ", b=" << bits << ")";
#endif
  }

  // Activation (B): NxK, Weights (A): MxK
  // This is the task of only one thread for GeMM: N x K x (M x num_threads)
  // Please split the blocks in llama.cpp and pass the right ptr for scales, A and C
  void llama_cpp_compute(void* A, void* scales, void* qlut, void* lut_scales, void* lut_biases, void* C, int M, int K, int N, int bits)
  {
#ifdef TMAC_USE_TVM_THREADPOOL
    DLTensor At = {
      /* .data = */ A,
    };
    DLTensor St = {
      /* .data = */ scales,
    };
    DLTensor Ct = {
      /* .data = */ C,
    };
    DLTensor QLUTt = {
      /* .data = */ qlut,
    };
    DLTensor LUTSt = {
      /* .data = */ lut_scales,
    };
    DLTensor LUTBt = {
      /* .data = */ lut_biases,
    };

    tvm::runtime::PackedFunc qf = get_function({M, K, N, bits, 1});
    qf(&At, &QLUTt, &St, &LUTSt, &LUTBt, &Ct);
#else
    int ret = qgemm_lut_int8(M * bits, K, N, bits, A, qlut, scales, lut_scales, lut_biases, C);
    DCHECK(ret == 0) << "error calling qgemm_lut (m=" << M << ", k=" << K << ", n=" << N << ", b=" << bits << ")";
#endif
  }

  TMACGeMMConfig get_kcfg(int M, int K, int N, int bits)
  {
    // TODO: find a better way to find kcfg when _n_threads is unknown
    const std::vector<int> n_threads_hints = {1, 4, 8, 16};
    std::string section;
    int old_n_threads = _n_threads;
    for (int n_threads : n_threads_hints) {
      _n_threads = n_threads;
      section = get_template_name({M, K, N, bits, 1});
      if (_reader.Sections().count(section) > 0) {
        break;
      }
    }
    _n_threads = old_n_threads;

    return {
      /* .bm              = */ (int)_reader.GetInteger(section, "bm", 0),
      /* .simd_n_in       = */ (int)_reader.GetInteger(section, "simd_n_in", 0),
      /* .simd_n_out      = */ (int)_reader.GetInteger(section, "simd_n_out", 0),
      /* .kfactor         = */ (int)_reader.GetInteger(section, "kfactor", 0),
      /* .group_size      = */ (int)_reader.GetInteger(section, "group_size", 0),
      /* .lut_scales_size = */ (int)_reader.GetInteger(section, "lut_scales_size", 0),
      /* .scales_size     = */ (int)_reader.GetInteger(section, "scales_size", 0),
      /* .n_tile_num      = */ (int)_reader.GetInteger(section, "n_tile_num", 0),
    };
  }

  // Should only be called in main thread
  void set_workspace(int maxK, int maxN)
  {
#if defined(_WIN32)
    _qlut = _aligned_malloc(maxN * maxK / g * (1 << g) * sizeof(int8_t), kAllocAlignment);
    _lut_scales = _aligned_malloc(maxN * maxK / _act_group_size * sizeof(T), kAllocAlignment);
    _lut_biases = _aligned_malloc(maxN * maxK / _act_group_size * sizeof(T), kAllocAlignment);
#else
    posix_memalign(&_qlut, kAllocAlignment, maxN * maxK / g * (1 << g) * sizeof(int8_t));
    posix_memalign(&_lut_scales, kAllocAlignment, maxN * maxK / _act_group_size * sizeof(T));
    posix_memalign(&_lut_biases, kAllocAlignment, maxN * maxK / _act_group_size * sizeof(T));
#endif
    _allocated = true;
  }

  ~TMACGeMMWrapper()
  {
    if (_allocated) {
#if defined(_WIN32)
      _aligned_free(_qlut);
      _aligned_free(_lut_scales);
      _aligned_free(_lut_biases);
#else
      free(_qlut);
      free(_lut_scales);
      free(_lut_biases);
#endif
    }
  }

private:
  using _fkey = std::tuple<int, int, int, int, int>;

#ifdef TMAC_USE_TVM_THREADPOOL
  tvm::runtime::Module _mod_lib;
  std::map<_fkey, tvm::runtime::PackedFunc> _fcache;
  const tvm::runtime::PackedFunc* _config_threadpool;

  tvm::runtime::PackedFunc get_function(_fkey key)
  {
    std::lock_guard<std::mutex> lock(_m);
    auto iter = _fcache.find(key);
    tvm::runtime::PackedFunc f;
    if (iter == _fcache.end()) {
      std::string func_name = get_template_name(key);
      f = _mod_lib.GetFunction(func_name);
      ICHECK(f != nullptr) << func_name;
      _fcache[key] = f;
      return f;
    } else {
      return iter->second;
    }
  }
#endif

  int _n_threads;
  int _act_group_size;

  // workspace ptrs
  void* _qlut;
  void* _lut_scales;
  void* _lut_biases;

  bool _allocated;
  std::mutex _m;

  INIReader _reader;

  std::string get_template_name(_fkey key)
  {
    if (std::get<4>(key) != 0) {
      return
        std::string("qgemm_lut")
          + "_t" + std::to_string(_n_threads)
          + "_int8"
          + "_m" + std::to_string(std::get<0>(key) * std::get<3>(key))
          + "_k" + std::to_string(std::get<1>(key))
          + "_n" + std::to_string(std::get<2>(key))
          + "_b" + std::to_string(std::get<3>(key));
    } else {
      return
        std::string("preprocessor")
          + "_t" + std::to_string(_n_threads)
          + "_int8"
          + "_m" + std::to_string(std::get<0>(key) * std::get<3>(key))
          + "_k" + std::to_string(std::get<1>(key))
          + "_n" + std::to_string(std::get<2>(key))
          + "_b" + std::to_string(std::get<3>(key));
    }
  }
};

} // namespace TMAC
