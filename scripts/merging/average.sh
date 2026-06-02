set -e
set -o pipefail

OLD_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
trap 'cd "$OLD_DIR"' EXIT
echo "current dir: $(pwd)"

export CUDA_VISIBLE_DEVICES=0

METHOD=average
MODEL_DIR=${HDFS_OUTPUT}/moe-merge/OLMoE-1B-7B-0125
BASE_MODEL_PATH=${HDFS_MODELS}/allenai/OLMoE-1B-7B-0125
SOURCE_MODELS="[
    '$MODEL_DIR/math/full_bs-32_lr-2e-5-linear_epochs-2_liger_z2',
    '$MODEL_DIR/code/full_bs-32_lr-2e-5-linear_epochs-2_liger_z2',
]"
OUTPUT_PATH=${MODEL_DIR}-merge/math-code/${METHOD}
TMP_PATH=/dev/shm/moe-merge-$(echo -n "$OUTPUT_PATH" | sha256sum | awk '{print $1}')
echo "Using temporary local path $TMP_PATH fast output."

echo "Merging models..."
python src/merge.py \
    --method $METHOD \
    --output_path "$TMP_PATH" \
    --source_models "$SOURCE_MODELS" \
    --base_model $BASE_MODEL_PATH \
    --device cuda \
    --target_dtype bfloat16

echo "Copying merged model to: $OUTPUT_PATH (background)"
mkdir -p "$OUTPUT_PATH"
(time cp -r "$TMP_PATH"/* "$OUTPUT_PATH"/) &
COPY_PID=$!

echo "Evaluating merged model (background)..."
bash scripts/eval/math_code.sh "$TMP_PATH" "$OUTPUT_PATH" &
EVAL_PID=$!

set +e
wait $COPY_PID
COPY_RC=$?
wait $EVAL_PID
EVAL_RC=$?
set -e

if [ $COPY_RC -ne 0 ] || [ $EVAL_RC -ne 0 ]; then
    echo "ERROR: copy rc=$COPY_RC, eval rc=$EVAL_RC"
    exit 1
fi

echo "Merged model saved to: $OUTPUT_PATH"

echo "Cleaning up temporary files..."
rm -rf $TMP_PATH
