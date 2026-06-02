# set -euo pipefail

OLD_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
trap 'cd "$OLD_DIR"' EXIT
echo "current dir: $(pwd)"

export HF_ALLOW_CODE_EVAL=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

DOMAINS='["math","code"]'
MODEL_PATH=output/OLMoE-1B-7B-0125-merge/math-code/wudi-300-base-none/router_calibration_cg-0.9-origin-none-1000-1e-6-OpenMathInstruct2_SelfOSSInstructSC2_correct/0.6
OUTPUT_PATH=$MODEL_PATH
REPEATS=4
TENSOR_PARALLEL_SIZE=1
while [[ $# -gt 0 ]]; do
  case $1 in
    --domains)      DOMAINS="$2"; shift 2 ;;
    --model_path)   MODEL_PATH="$2"; shift 2 ;;
    --output_path)  OUTPUT_PATH="$2"; shift 2 ;;
    --repeats)      REPEATS="$2"; shift 2 ;;
    --tp)           TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    *)              echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done
DOMAINS_CLEAN="${DOMAINS//[\[\]\"]/}"  # 去掉 [ ] "
OUTPUT_PATH="${OUTPUT_PATH}/test-${DOMAINS_CLEAN}-${REPEATS}"
TMP_OUTPUT_PATH=$(mktemp -d /dev/shm/eval-XXXXXX)

echo "Running Evaluation"
python src/evaluate.py \
    --model_path "$MODEL_PATH" \
    --domains "$DOMAINS" \
    --output_path "$TMP_OUTPUT_PATH" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --repeats "$REPEATS" \
    --bashrc_path "$SCRIPT_DIR/../../.mybashrc" \
    --gen_env_name eval \
    --eval_env_name bcb_text

# Copy results to HDFS
sleep 5
echo "Copying results to $OUTPUT_PATH"
mkdir -p "$OUTPUT_PATH"
cp -r "$TMP_OUTPUT_PATH"/* "$OUTPUT_PATH"/
rm -r "$TMP_OUTPUT_PATH"
echo "Finished."
