set -e
set -o pipefail

OLD_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
trap 'cd "$OLD_DIR"' EXIT
echo "current dir: $(pwd)"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LLAMAFACTORY_ASYNC_OUTPUT_DIR=1
export LLAMAFACTORY_ASYNC_OUTPUT_DIR_CKPT_STABLE_SEC=10
export LLAMAFACTORY_ASYNC_OUTPUT_DIR_BACKLOG=5
export LLAMAFACTORY_ASYNC_OUTPUT_DIR_EXIT_TIMEOUT=800
export CUDA_LAUNCH_BLOCKING=1

source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/huangcanbin02/.mybashrc
conda activate lf

MODEL_PATH=$HDFS_MODELS/allenai/OLMoE-1B-7B-0125
TEMPLATE=olmo
DATA_DIR=$HDFS_DATA/merge/sft
DATASET_TYPE=math

PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=2e-5
LR_SCHEDULER_TYPE=linear
NUM_TRAIN_EPOCHS=2
ADDITIONAL_ARGS="--deepspeed config/deepspeed/ds_z2_config.json"
ADDITIONAL_ARGS="$ADDITIONAL_ARGS --enable_liger_kernel"
ADDITIONAL_ARGS="$ADDITIONAL_ARGS --packing true --neat_packing"

declare -A DATASET_MAP
DATASET_MAP["if"]="oasst1_converted,flan_v2_converted,no_robots_converted,personahub_ifdata_manual_seed_v3_29980"
DATASET_MAP["ml"]="tulu_v3.9_wildchat_100k,tulu_v3.9_aya_100k"
DATASET_MAP["safety"]="coconot_converted,tulu_v3.9_wildjailbreak_decontaminated_50k,tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k"
DATASET_MAP["math"]="personahub_math_v5_regen_149960,tulu-3-sft-personas-math-grade,tulu_v3.9_open_math_2_gsm8k_50k,tulu_v3.9_personahub_math_interm_algebra_20k"
DATASET_MAP["mathpython"]="numinamath_tir_math_decontaminated"
DATASET_MAP["code"]="personahub_code_v2_34999,evol_codealpaca_heval_decontaminated"
DATASET_MAP["science"]="tulu_v3.9_sciriff_10k"
DATASET_MAP["table"]="tulu_v3.9_table_gpt_5k"
DATASET_MAP["math-code"]="${DATASET_MAP["math"]},${DATASET_MAP["code"]}"
DATASET_MAP["all"]="${DATASET_MAP["if"]},${DATASET_MAP["ml"]},${DATASET_MAP["safety"]},${DATASET_MAP["math"]},${DATASET_MAP["mathpython"]},${DATASET_MAP["code"]},${DATASET_MAP["science"]},${DATASET_MAP["table"]}"

DATASET_MAP["new_ml_without_en"]="Russian,Chinese,Spanish,Telugu,French,Japanese,German,Bengali,Thai,Swahili"
DATASET_MAP["new_ml"]="English,${DATASET_MAP["new_ml_without_en"]}"
DATASET_MAP["math-new_ml_without_en"]="${DATASET_MAP["math"]},${DATASET_MAP["new_ml_without_en"]}"
DATASET_MAP["math-new_ml"]="${DATASET_MAP["math"]},${DATASET_MAP["new_ml"]}"

SCRIPT_NAME=$(basename "$0")
MODEL_NAME=$(basename $MODEL_PATH)
CUDAS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
TOTAL_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * CUDAS))
OUTPUT_DIR=$HDFS_OUTPUT/moe-merge/${MODEL_NAME}/${DATASET_TYPE}/full_bs-${TOTAL_BATCH_SIZE}_lr-${LEARNING_RATE}-${LR_SCHEDULER_TYPE}_epochs-${NUM_TRAIN_EPOCHS}_liger_z2_packing
OUTPUT_DIR="${OUTPUT_DIR//,/_}"  # replace , to _

# ============================================== Debug ================================================
# export CUDA_VISIBLE_DEVICES=0
# PER_DEVICE_TRAIN_BATCH_SIZE=2
# GRADIENT_ACCUMULATION_STEPS=2
# ADDITIONAL_ARGS="$ADDITIONAL_ARGS --max_samples 400"
# OUTPUT_DIR=output/tmp
# =====================================================================================================

mkdir -p $OUTPUT_DIR
cat "${SCRIPT_DIR}/$(basename "$0")" > $OUTPUT_DIR/train.sh
llamafactory-cli train \
    --template $TEMPLATE \
    --dataset ${DATASET_MAP[$DATASET_TYPE]} \
    --dataset_dir $DATA_DIR \
    --cutoff_len 4096 \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 1 \
    --model_name_or_path $MODEL_PATH \
    --moe_aux_loss_coef 0 \
    --trust_remote_code \
    --stage sft \
    --finetuning_type full \
    --plot_loss \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --save_strategy steps \
    --save_steps 50 \
    --logging_steps 10 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --bf16 \
    $ADDITIONAL_ARGS \
    | tee $OUTPUT_DIR/train.log


conda activate eval
bash scripts/eval/math_code.sh $OUTPUT_DIR
