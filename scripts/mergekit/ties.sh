set -e
set -o pipefail

OLD_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
trap 'cd "$OLD_DIR"' EXIT
echo "current dir: $(pwd)"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DENSITY=0.2
while [[ $# -gt 0 ]]; do
  case $1 in
    --density)      DENSITY="$2"; shift 2 ;;
    *)              echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

METHOD=ties
DTYPE=bfloat16
MODEL_DIR=output/OLMoE-1B-7B-0125
BASE_MODEL_PATH=models/allenai/OLMoE-1B-7B-0125
MODEL1_PATH=${MODEL_DIR}/if/full_bs-32_lr-2e-5-linear_epochs-2_liger_z2
MODEL2_PATH=${MODEL_DIR}/math/full_bs-32_lr-2e-5-linear_epochs-2_liger_z2
MODEL3_PATH=${MODEL_DIR}/code/full_bs-32_lr-2e-5-linear_epochs-2_liger_z2
MODEL3_PATH=${MODEL_DIR}/code/full_bs-32_lr-2e-5-linear_epochs-2_liger_z2
OUTPUT_PATH="${MODEL_DIR}-mergekit/if-math-code/${METHOD}-${DTYPE}"
TMP_PATH=$(mktemp -d /dev/shm/mergekit-XXXXXX)
CONFIG="\
models:
  - model: $MODEL1_PATH
    parameters:
      density: $DENSITY
      weight: 0.3333
  - model: $MODEL2_PATH
    parameters:
      density: $DENSITY
      weight: 0.3333
  - model: $MODEL3_PATH
    parameters:
      density: $DENSITY
      weight: 0.3333
merge_method: $METHOD
base_model: $BASE_MODEL_PATH
parameters:
  normalize: true
  int8_mask: true
dtype: $DTYPE
tokenizer:
  source: $MODEL1_PATH
chat_template: auto
"
mkdir -p config/mergekit
echo "$CONFIG" > config/mergekit/${METHOD}.yml

echo "Using temporary local path $TMP_PATH for fast output."
echo "Merging models with MergeKit..."
mergekit-yaml config/mergekit/${METHOD}.yml \
    $TMP_PATH \
    --trust-remote-code \
    --cuda \
    --lazy-unpickle \
    --read-to-gpu \
    --multi-gpu \
    --num-threads 64 \
    --copy-tokenizer \
    --random-seed 42

echo "Copying merged model to: $OUTPUT_PATH (background)"
mkdir -p "$OUTPUT_PATH"
(time cp -r "$TMP_PATH"/* "$OUTPUT_PATH"/) &
COPY_PID=$!

# echo "Validating merged model (background)..."
# bash scripts/validate.sh \
#     --model_path "$TMP_PATH" \
#     --data_dir data/calibration \
#     --datasets "{'OpenMathInstruct2_val': 'math', 'SelfOSSInstructSC2_val': 'code'}" \
#     --output_dir "$OUTPUT_PATH/val-math,code" &
# EVAL_PID=$!

echo "Evaluating merged model (background)..."
bash scripts/test.sh \
    --domains '["if","math","code"]' \
    --model_path "$TMP_PATH" \
    --output_path "$OUTPUT_PATH/test-["if","math","code"]-4" \
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
