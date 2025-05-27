## Conventions

- `EXECUTORCH_ROOT` refers to the root directory of executorch. Replace it or prepare the environment variable before execution.
- `HEXAGON_SDK_ROOT` refers to the root directory of Hexagon NPU SDK.
- `QNN_SDK_ROOT` refers to the root directory of downloaded QNN SDK

## Build Executorch

Follow [backends-qualcomm](https://github.com/pytorch/executorch/blob/main/docs/source/backends-qualcomm.md) to build Executorch with QNN.

## Android LlamaDemo

Follow [delegates-qualcomm](https://github.com/pytorch/executorch/blob/main/examples/demo-apps/android/LlamaDemo/docs/delegates/qualcomm_README.md) to build the Android APP with QNN.

## Build OP Package

Follow [this document](https://docs.qualcomm.com/bundle/publicresource/topics/80-77512-1/hexagon-dsp-sdk-collection-landing-page.html?product=1601111740010422) to download and setup Hexagon NPU SDK.

In `${EXECUTORCH_ROOT}/backends/qualcomm/runtime/op_packages/TMANOpPackage`, run `make htp_x86 htp_v75 htp_v79`, the file will be generated under `build/{arch}/libQnnTMANOpPackage.so`.

## Convert Models

Prepare environments:

```
cd ${HEXAGON_SDK_ROOT}
source setup_sdk_env.source
cd ${QNN_SDK_ROOT}/bin
source envsetup.sh
export QNN_OP_PACKAGE_PATHS="${EXECUTORCH_ROOT}/backends/qualcomm/runtime/op_packages/TMANOpPackage/build/x86_64-linux-clang/libQnnTMANOpPackage.so:TMANOpPackageInterfaceProvider:CPU"
```

Convert from GPTQ models:

```
cd ${EXECUTORCH_ROOT}
python examples/qualcomm/oss_scripts/llama3/llama3.py -b build-android -s emulator-5554 -m SM8650 --ptq 16a4w --model_dir ${MODEL_DIR} --tokenizer_model ${MODEL_DIR}/tokenizer.json --model_mode kv --max_seq_len 128 --prompt "User: Are you conscious?\nAssistant: " --llama_model llama3_2 --artifact ${MODEL_DIR}/llama_qnn --use_tman --compile_only --num_sharding 1 --kv_updater shift_pointer
```

- `-m`: `soc_model`
    - SM8550(Snapdragon 8 Gen 2): htp_v73
    - SM8650(Snapdragon 8 Gen 3): htp_v75
    - SM8750(Snapdragon 8 Elite): htp_v79
