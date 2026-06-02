set -e
set -o pipefail

OLD_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
trap 'cd "$OLD_DIR"' EXIT
echo "current dir: $(pwd)"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

METHOD=router_calibration
MODEL_DIR=output/OLMoE-1B-7B-0125
BASE_MODEL_PATH=output/OLMoE-1B-7B-0125-merge/math-code/wudi-300-base-none
SOURCE_MODELS="[
    '$MODEL_DIR/math/full_bs-32_lr-2e-5-linear_epochs-2_liger_z2_packing',
    '$MODEL_DIR/code/full_bs-32_lr-2e-5-linear_epochs-2_liger_z2_packing',
]"
MAX_SAMPLES_PER_DOMAIN=1000
REDUCE_NON_DIAG_A=0.9
OUTPUT_PATH=${BASE_MODEL_PATH}/${METHOD}-${REDUCE_NON_DIAG_A}-OpenMathInstruct2_SelfOSSInstructSC2_correct-${MAX_SAMPLES_PER_DOMAIN}
TMP_PATH=$(mktemp -d /dev/shm/merge-XXXXXX)
echo "Using temporary local path $TMP_PATH for fast output."

echo "Merging models..."
python src/merge.py \
    --method $METHOD \
    --output_path "$TMP_PATH" \
    --source_models "$SOURCE_MODELS" \
    --base_model $BASE_MODEL_PATH \
    --device cuda \
    --target_dtype bfloat16 \
    --datasets "{'math': ['output/OLMoE-1B-7B-0125/data/OpenMathInstruct2_correct.jsonl'], 'code': ['output/OLMoE-1B-7B-0125/data/SelfOSSInstructSC2_correct.jsonl']}" \
    --max_samples_per_domain $MAX_SAMPLES_PER_DOMAIN \
    --batch_size 4 \
    --reduce_non_diag_a $REDUCE_NON_DIAG_A


echo "Copying merged model to: $OUTPUT_PATH (background)"
mkdir -p "$OUTPUT_PATH"
(time cp -r "$TMP_PATH"/* "$OUTPUT_PATH"/) &
COPY_PID=$!

echo "Evaluating merged model (background)..."
bash scripts/test.sh \
    --domains math,code \
    --model_path "$TMP_PATH" \
    --output_path "$OUTPUT_PATH/test-math,code-4" \
    --repeats 4 \
    --tp 1 &
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
