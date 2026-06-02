set -euo pipefail

OLD_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
trap 'cd "$OLD_DIR"' EXIT
echo "current dir: $(pwd)"

export CUDA_VISIBLE_DEVICES=0,1,2,3

METHOD=router_calibration_cg
PRECONDITIONER=none
REGULARIZATION=0.9

while [[ $# -gt 0 ]]; do
  case $1 in
    --regularization)      REGULARIZATION="$2"; shift 2 ;;
    *)                     echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

W_INIT=origin
MAX_ITER=1000
CG_TOL=1e-6
MODEL_DIR=output/OLMoE-1B-7B-0125
BASE_MODEL_PATH=output/OLMoE-1B-7B-0125-merge/if-math-code/wudi-300-base-none
SOURCE_MODELS="[
    '$MODEL_DIR/if/full_bs-32_lr-2e-5-linear_epochs-2_liger_z2',
    '$MODEL_DIR/math/full_bs-32_lr-2e-5-linear_epochs-2_liger_z2',
    '$MODEL_DIR/code/full_bs-32_lr-2e-5-linear_epochs-2_liger_z2',
]"
OUTPUT_PATH=$BASE_MODEL_PATH/${METHOD}-${REGULARIZATION}-${W_INIT}-${PRECONDITIONER}-${MAX_ITER}-${CG_TOL}-UltraFeedback_OpenMathInstruct2_SelfOSSInstructSC2_correct_7000
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
    --work_dtype float32 \
    --datasets "{'if': ['output/OLMoE-1B-7B-0125/data/UltraFeedback_correct.jsonl'], 'math': ['output/OLMoE-1B-7B-0125/data/OpenMathInstruct2_correct.jsonl'], 'code': ['output/OLMoE-1B-7B-0125/data/SelfOSSInstructSC2_correct.jsonl']}" \
    --max_samples_per_domain 7000 \
    --batch_size 4 \
    --regularization $REGULARIZATION \
    --w_init $W_INIT \
    --preconditioner_type $PRECONDITIONER \
    --cg_max_iter $MAX_ITER \
    --cg_tol $CG_TOL \
    | tee $TMP_PATH/merging.log

echo "Copying merged model to: $OUTPUT_PATH (background)"
mkdir -p "$OUTPUT_PATH"
(time cp -r "$TMP_PATH"/* "$OUTPUT_PATH"/) &
COPY_PID=$!

echo "Evaluating merged model (background)..."
bash scripts/test.sh \
    --domains "['if','math','code']" \
    --model_path "$TMP_PATH" \
    --output_path "$OUTPUT_PATH" \
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
