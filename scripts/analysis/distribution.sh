set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

OLD_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
trap 'cd "$OLD_DIR"' EXIT
echo "current dir: $(pwd)"

SOURCE_MODEL_PATH=output/OLMoE-1B-7B-0125/code/full_bs-32_lr-2e-5-linear_epochs-2_liger_z2
MERGED_MODEL_PATH=output/OLMoE-1B-7B-0125-merge/math-code/wudi-300-base-none
DATA_PATH=data/sft/code_val.jsonl
OUTPUT_PATH=output/analysis/distribution/code_val
TMP_OUTPUT_PATH=$(mktemp -d /dev/shm/eval-XXXXXX)

# Run extraction
mkdir -p "$TMP_OUTPUT_PATH"
python src/analysis/distribution.py \
    --input_file $DATA_PATH \
    --output_dir $TMP_OUTPUT_PATH \
    --model1 $MERGED_MODEL_PATH \
    --model2 $SOURCE_MODEL_PATH \
    --max_length 4096

sleep 5
echo "Copying results to $OUTPUT_PATH"
mkdir -p "$OUTPUT_PATH"
cp -r "$TMP_OUTPUT_PATH"/* "$OUTPUT_PATH"/
rm -r "$TMP_OUTPUT_PATH"
echo "Finished."
