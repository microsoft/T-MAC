# End-2-End Inference Through llama.cpp (legacy)

> The following guide use BitNet-3B. We will add instructions how to use GPTQ/GGUF/BitDistiller models or even your customized models.

First, download the model `huggingface-cli download 1bitLLM/bitnet_b1_58-3B --local-dir ${model_dir}`.

Then, compile kernels for the model. There are two options:

- Use prebuilt kernels for ARM CPUs:
    ```bash
    cd deploy
    cp tuned/arm-hf-bitnet-3b/* tuned/
    ```
- Compile the kernels yourself:
    ```bash
    cd deploy
    python compile.py -o tuned -da -nt 4 -tb -gc -ags -1 -t -m hf-bitnet-3b
    ```
    > Specify `-ags 64` on ARM CPUs for better performance.

Build T-MAC C++ source:

```bash
cd ..  # back to project root directory
export TMAC_ROOT_DIR=$(pwd)
cd build
cmake -DCMAKE_INSTALL_PREFIX=${TMAC_ROOT_DIR}/install ..
cmake --build . --target install --config Release
```

Convert the huggingface model to gguf:

```bash
cd ../3rdparty/llama.cpp/gguf-py
pip install .
cd ..
python convert-hf-to-gguf-bitnet.py ${model_dir} --outtype i2 --outfile ${model_dir}/hf-bitnet-3B.i2.gguf --kcfg ${TMAC_ROOT_DIR}/install/lib/kcfg.ini
```

Build llama.cpp:

```bash
# in 3rdparty/llama.cpp
mkdir build
cd build
cmake .. -DLLAMA_TMAC=ON -DCMAKE_PREFIX_PATH=${TMAC_ROOT_DIR}/install/lib/cmake/t-mac -DCMAKE_BUILD_TYPE=Release -DLLAMA_LLAMAFILE_DEFAULT=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
# In Windows Visual Studio PowerShell:
# cmake .. -DLLAMA_TMAC=ON -DCMAKE_PREFIX_PATH=${TMAC_ROOT_DIR}/install/lib/cmake/t-mac -DCMAKE_BUILD_TYPE=Release -DLLAMA_LLAMAFILE_DEFAULT=OFF -T ClangCL
cmake --build . --target main llama-bench --config Release
```

Run inference through:

```bash
./bin/main -m ~/Downloads/test_models/hf-bitnet-3B.i2.gguf -n 128 -t 1 -p "Microsoft Corporation is an American multinational corporation and technology company headquartered in Redmond, Washington." -b 1 -ngl 0 -c 2048
```
