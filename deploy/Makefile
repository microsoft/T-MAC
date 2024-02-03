ROOT_DIR:=$(shell dirname $(dir $(realpath $(firstword $(MAKEFILE_LIST)))))

PKG_CFLAGS = -std=c++17 -O3 -fPIC\
	-I${TVM_ROOT}/include\
	-I${TVM_ROOT}/3rdparty/dmlc-core/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-I${ROOT_DIR}/include\
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

PKG_LDFLAGS = -L${TVM_ROOT}/build -ldl -pthread

.PHONY: clean all

all: build/benchmark

build/kernels.o: compile.py
	@mkdir -p $(@D)
	python compile.py -t -r -o ./tuned
	@mv ./tuned/kernels.o $@

build/benchmark: benchmark.cc build/kernels.o
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ -ltvm_runtime $(PKG_LDFLAGS)

clean:
	rm -rf build
