set -e

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <model_path> <kernel_name> <model_type> [--rechunk] [--convert-model] [--run-only] [--disable-t-mac]"
    echo "       model_path: path to the model directory"
    echo "       kernel_name: name of the kernel for compiler, e.g., llama-2-7b-4bit, hf-bitnet-3b, hf-bitnet-large-intn, hf-bitnet-large-tq, trilm-3.9b"
    echo "       model_type: type of the model, e.g., f16, int_n, tq1_0, tq2_0, q4_0"
    echo "       --rechunk: optional. Rechunk the model if set."
    echo "       --convert-model: optional. Convert the model to gguf format if set."
    echo "       --run-only: optional. Skip the compilation and only run the inference and benchmark if set."
    echo "       --disable-t-mac: optional. Disable T-MAC if set."
    exit 1
fi


if [[ "$3" == "q4_0" ]]; then
    export EXTRA_COMPILE_ARGS=("-gs=32" "-ags=32")
elif [[ "$3" == "tq1_0" || "$3" == "tq2_0" ]]; then
    export EXTRA_COMPILE_ARGS=("-gs=256" "-ags=64")
else
    export EXTRA_COMPILE_ARGS=()
fi


RECHUNK=false
for arg in "$@"; do
    case $arg in
        --rechunk)
            RECHUNK=true
            ;;
        *)
            ;;
    esac
done


CONVERT_MODEL=false
for arg in "$@"; do
    case $arg in
        --convert-model)
            CONVERT_MODEL=true
            ;;
        *)
            ;;
    esac
done

RUN_ONLY=false
for arg in "$@"; do
    case $arg in
        --run-only)
            RUN_ONLY=true
            ;;
        *)
            ;;
    esac
done

DISABLE_T_MAC=false
for arg in "$@"; do
    case $arg in
        --disable-t-mac)
            DISABLE_T_MAC=true
            ;;
        *)
            ;;
    esac
done

export MODEL_DIR=$(readlink -f "$1")
export KERNEL_NAME=$2
export MODEL_DTYPE=$3

echo "MODEL_DIR: $MODEL_DIR"
echo "KERNEL_NAME: $KERNEL_NAME"
echo "MODEL_DTYPE: $MODEL_DTYPE"
echo "RECHUNK: $RECHUNK"
echo "CONVERT_MODEL: $CONVERT_MODEL"
echo "RUN_ONLY: $RUN_ONLY"
echo "DISABLE_T_MAC: $DISABLE_T_MAC"


if [ "$RUN_ONLY" != true ]; then
    if [ "$DISABLE_T_MAC" == true ]; then
        echo "===  python tools/run_pipeline.py -o $MODEL_DIR -m $KERNEL_NAME -nt 4 -s 4,5 "${EXTRA_COMPILE_ARGS[@]}" --disable-t-mac  ==="
        python tools/run_pipeline.py -o $MODEL_DIR -m $KERNEL_NAME -nt 4 -s 4,5 ${EXTRA_COMPILE_ARGS[@]} --disable-t-mac
    else
        echo "===  python tools/run_pipeline.py -o $MODEL_DIR -m $KERNEL_NAME -nt 4 -s 0,1,2,4,5 "${EXTRA_COMPILE_ARGS[@]}" -q $MODEL_DTYPE  ==="
        python tools/run_pipeline.py -o $MODEL_DIR -m $KERNEL_NAME -nt 4 -s 0,1,2,4,5 ${EXTRA_COMPILE_ARGS[@]} -q $MODEL_DTYPE
        if $CONVERT_MODEL; then
            echo "===  python tools/run_pipeline.py -o $MODEL_DIR -m $KERNEL_NAME -nt 4 -s 3 "${EXTRA_COMPILE_ARGS[@]}" -q $MODEL_DTYPE  ==="
            python tools/run_pipeline.py -o $MODEL_DIR -m $KERNEL_NAME -nt 4 -s 3 ${EXTRA_COMPILE_ARGS[@]} -q $MODEL_DTYPE
        fi
    fi
fi

echo "===  python tools/run_pipeline.py -o "$MODEL_DIR" -it "$MODEL_DTYPE" -s 6  ==="
python tools/run_pipeline.py -o "$MODEL_DIR" -it $MODEL_DTYPE -s 6
for threads in $(seq 1 4); do
    echo "===  Running with $threads threads, 1 batch  ==="
    python tools/run_pipeline.py -o "$MODEL_DIR" -it $MODEL_DTYPE -nt $threads -s 7
done

