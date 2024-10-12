FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

## Install Python-3.9
RUN apt-get update \
 && apt-get install -y software-properties-common \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get install -y python3.9 python3.9-distutils python3.9-venv python3-pip \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
 && python -m pip install --upgrade pip \
 && python -V

## Install TVM build dependencies and other required packages.
RUN apt-get install -y \
                    build-essential libtinfo-dev zlib1g-dev libzstd-dev libxml2-dev \
                    wget \
                    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*


## Set HOME path
ENV HOME /root
ENV MODEL_DIR $HOME/models

## Install cmake-3.29
RUN cd $HOME \
 && wget https://github.com/Kitware/CMake/releases/download/v3.30.4/cmake-3.30.4-linux-x86_64.sh \
 && chmod +x ./cmake-3.30.4-linux-x86_64.sh \
 && ./cmake-3.30.4-linux-x86_64.sh --skip-license --prefix=/usr/local \
 && cmake --version

## Set virtualenv
RUN python -m venv $HOME/tmac

## Download T-MAC
RUN cd $HOME \
 && git clone --recursive https://github.com/microsoft/T-MAC.git -b main \
 && cd T-MAC

## Install T-MAC
RUN . $HOME/tmac/bin/activate \
 && cd $HOME/T-MAC \
 && pip install -e . -v \
 && . build/t-mac-envs.sh \
 && pip install 3rdparty/llama.cpp/gguf-py

## Download model and run pipeline
# RUN . $HOME/tmac/bin/activate \
#  && . build/t-mac-envs.sh \
#  && cd $HOME/T-MAC \
#  && mkdir $MODEL_DIR \
#  && huggingface-cli download ChenMnZ/Llama-2-7b-EfficientQAT-w4g128-GPTQ --local-dir $MODEL_DIR/llama-2-7b-4bit \
#  && python tools/run_pipeline.py -o $MODEL_DIR/llama-2-7b-4bit -m llama-2-7b-4bit -nt 4