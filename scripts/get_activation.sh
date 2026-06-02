set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

OLD_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
trap 'cd "$OLD_DIR"' EXIT
echo "current dir: $(pwd)"

MODEL_PATH=output/OLMoE-1B-7B-0125/code/full_bs-32_lr-2e-5-linear_epochs-2_liger_z2
DATA_DIR=data/sft
DATA_FILE=code_val
OUTPUT_DIR=output/OLMoE-1B-7B-0125/activation

WORLD_SIZE=1
RANK=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --world_size)   WORLD_SIZE="$2"; shift 2 ;;
    --rank)         RANK="$2"; shift 2 ;;
    *)              echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done


# Run extraction
mkdir -p $OUTPUT_DIR
python src/get_activation.py run \
    --model "$MODEL_PATH" \
    --data_files "$DATA_FILE" \
    --input_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_len 4096 \
    --dtype bf16 \
    --max_tokens_per_batch 131072 \
    --world_size "$WORLD_SIZE" \
    --rank "$RANK"