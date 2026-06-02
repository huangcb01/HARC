set -euo pipefail

OLD_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
trap 'cd "$OLD_DIR"' EXIT
echo "current dir: $(pwd)"


MODEL_PATH=output/OLMoE-1B-7B-0125/code/full_bs-32_lr-2e-5-linear_epochs-2_liger_z2
DATA_DIR=data/calibration
DATASETS="{'OpenMathInstruct2_val': 'math', 'SelfOSSInstructSC2_val': 'code'}"
OUTPUT_DIR=output/OLMoE-1B-7B-0125/data
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)   MODEL_PATH="$2"; shift 2 ;;
    --data_dir)     DATA_DIR="$2"; shift 2 ;;
    --datasets)     DATASETS="$2"; shift 2 ;;
    --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
    *)              echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done


mkdir -p $OUTPUT_DIR
python src/validate.py full \
    --model_name_or_path $MODEL_PATH \
    --dataset_dir $DATA_DIR \
    --datasets "$DATASETS" \
    --output_dir $OUTPUT_DIR \
    --temperature 0 \
    --seed 42
