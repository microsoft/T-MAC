cd $HOME/T-MAC
source $HOME/tmac/bin/activate
source build/t-mac-envs.sh
mkdir $MODEL_DIR
huggingface-cli download ChenMnZ/Llama-2-7b-EfficientQAT-w4g128-GPTQ --local-dir $MODEL_DIR/llama-2-7b-4bit
python tools/run_pipeline.py -o $MODEL_DIR/llama-2-7b-4bit -m llama-2-7b-4bit -nt 4