set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

OLD_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
trap 'cd "$OLD_DIR"' EXIT
echo "current dir: $(pwd)"

SOURCE_MODEL_PATH=output/OLMoE-1B-7B-0125/code/full_bs-32_lr-2e-5-linear_epochs-2_liger_z2
MERGED_MODEL_PATH=output/OLMoE-1B-7B-0125-merge/math-code/wudi-300-base-none
DATA_PATH=output/OLMoE-1B-7B-0125-merge/math-code/wudi-300-base-none/data/SelfOSSInstructSC2_correct.jsonl
OUTPUT_PATH=output/analysis/relation/SelfOSSInstructSC2_correct.json

# Run extraction
mkdir -p $(dirname "$OUTPUT_PATH")
python src/analysis/relation.py \
    --source_model_path "$SOURCE_MODEL_PATH" \
    --merged_model_path "$MERGED_MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH"
