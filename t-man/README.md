<h1 align="center">
T-MAN
</h1>

<h3 align="center">
Efficient Low-Bit LLM Inference on NPU
</h3>

<div align="center">
<video src="https://github.com/user-attachments/assets/4da57932-5942-4a69-bf96-2ad158a3640c"></video>
</div>


T-MAN extends the idea of [utilizing table lookup for low-bit inference](https://dl.acm.org/doi/10.1145/3689031.3696099) to Qualcomm NPU. It is the first NPU inference framework to support 1/1.58-bit [BitNet](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T), and 2-bit QAT models including [Qwen-3](https://huggingface.co/BitDistiller/Qwen-8B-w2g64-gptq) and [Llama-3](https://huggingface.co/BitDistiller/Llama-3.1-8B-Instruct-w2g64-gptq) from [BitDistiller](https://github.com/DD-DuDa/BitDistiller).

By achieving up to 50 t/s token generation for [BitNet-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) on Snapdragon 8G3, T-MAN is 2x faster than T-MAC. T-MAN is even 1.4x faster for Llama-3.1-8B compared to Qualcomm [QNN](https://aihub.qualcomm.com/mobile/models/llama_v3_1_8b_instruct). As only NPU is required, T-MAN also does not impact the performance of your commonly used apps that rely on CPU and GPU.

### Use the Android App

- Get the apk from the [release page](https://github.com/microsoft/T-MAC/releases).
- Select a model (e.g., Qwen3-8B) in the settings. The model files will be downloaded automatically (requires internet access).
- Load the model.
- Enjoy your conversation!

### :warning: Notice

- The demo APK has been tested only on Snapdragon 8 GEN 3 devices.
- Models are exported using [KV Cache Mode](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/oss_scripts/llama/README.md). Prefill performance is currently slow, but hybrid models will be released soon.
- The building process is complicated as QNN is required. We are preparing a docker container. Currently we recommend testing through the prebuilt apk.

### Building the Project

For instructions on building the project, refer to [build.md](docs/build.md).
