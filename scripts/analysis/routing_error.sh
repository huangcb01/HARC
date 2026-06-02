set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

OLD_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
trap 'cd "$OLD_DIR"' EXIT
echo "current dir: $(pwd)"

for DOMAIN in math code; do
    SOURCE_MODEL_PATH=output/OLMoE-1B-7B-0125/${DOMAIN}/full_bs-32_lr-2e-5-linear_epochs-2_liger_z2
    MERGED_MODEL_PATH=output/OLMoE-1B-7B-0125-mergekit/math-code/linear-bfloat16
    DATA_DIR=data/sft
    DATA_FILE=${DOMAIN}_val
    OUTPUT_DIR=output/analysis/routing_error/router_calibration_cg_${DATA_FILE}

    # Run extraction
    mkdir -p $OUTPUT_DIR
    python src/analysis/routing_error.py \
        --source_model_path "$SOURCE_MODEL_PATH" \
        --merged_model_path "$MERGED_MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --dataset_name "$DATA_FILE" \
        --output_dir "$OUTPUT_DIR" \
        | tee "$OUTPUT_DIR/routing_error.log"
done
