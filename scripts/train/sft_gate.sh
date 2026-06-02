set -e
set -o pipefail

OLD_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
trap 'cd "$OLD_DIR"' EXIT
echo "current dir: $(pwd)"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TIMEOUT=1200


MODEL_PATH=output/OLMoE-1B-7B-0125-merge/math-code/wudi-300-base-none
TEMPLATE=olmo
DATA_DIR=output/OLMoE-1B-7B-0125/data
DATASETS=(OpenMathInstruct2_correct SelfOSSInstructSC2_correct)

MOE_ROUTER_LOSS_TYPE=forward_kl
MOE_ROUTER_LOSS_WEIGHT=100
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=5e-5
LR_SCHEDULER_TYPE=linear
NUM_TRAIN_EPOCHS=2
SAVE_STEPS=50
ADDITIONAL_ARGS="--deepspeed config/deepspeed/ds_z2_config.json"
ADDITIONAL_ARGS="$ADDITIONAL_ARGS --enable_liger_kernel"
ADDITIONAL_ARGS="$ADDITIONAL_ARGS --disable_gradient_checkpointing"
ADDITIONAL_ARGS="$ADDITIONAL_ARGS --moe_router_loss_type ${MOE_ROUTER_LOSS_TYPE} \
    --moe_router_loss_weight ${MOE_ROUTER_LOSS_WEIGHT} \
    --gold_router_output_dir output/OLMoE-1B-7B-0125/activation/router_output \
"

TRAIN_DATASETS=$(IFS=,; echo "${DATASETS[*]/%/_train}")
VAL_DATASETS=$(IFS=,; echo "${DATASETS[*]/%/_val}")
MODEL_NAME=$(basename $MODEL_PATH)
CUDAS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
TOTAL_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * CUDAS))
OUTPUT_DIR=$MODEL_PATH/$(IFS=-; echo "${DATASETS[*]}")/freeze_${MOE_ROUTER_LOSS_TYPE}-${MOE_ROUTER_LOSS_WEIGHT}_bs-${TOTAL_BATCH_SIZE}_lr-${LEARNING_RATE}-${LR_SCHEDULER_TYPE}_epochs-${NUM_TRAIN_EPOCHS}_liger_z2
OUTPUT_DIR="${OUTPUT_DIR//,/_}"  # replace , to _
TMP_DIR=$(mktemp -d /dev/shm/llamafactory-XXXXXX)

# ============================================== Debug ================================================
# export CUDA_VISIBLE_DEVICES=0
# PER_DEVICE_TRAIN_BATCH_SIZE=2
# GRADIENT_ACCUMULATION_STEPS=2
# ADDITIONAL_ARGS="$ADDITIONAL_ARGS --max_samples 400"
# OUTPUT_DIR=output/tmp
# rm -rf $OUTPUT_DIR
# =====================================================================================================

mkdir -p $OUTPUT_DIR
cat "${SCRIPT_DIR}/$(basename "$0")" > $OUTPUT_DIR/train.sh
training_command="
set -e && \
set -o pipefail && \
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/huangcanbin02/.mybashrc && \
conda activate lf && \
llamafactory-cli train \
    --template $TEMPLATE \
    --dataset $TRAIN_DATASETS \
    --eval_dataset $VAL_DATASETS \
    --dataset_dir $DATA_DIR \
    --cutoff_len 4096 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 1 \
    --model_name_or_path $MODEL_PATH \
    --moe_aux_loss_coef 0 \
    --trust_remote_code \
    --stage sft \
    --finetuning_type freeze \
    --freeze_trainable_layers 100000 \
    --freeze_trainable_modules mlp.gate \
    --plot_loss \
    --output_dir $TMP_DIR \
    --do_train \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --save_total_limit 5 \
    --logging_steps 10 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --bf16 \
    --do_eval \
    --per_device_eval_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --eval_strategy steps \
    --eval_steps $SAVE_STEPS \
    --compute_accuracy \
    --load_best_model_at_end \
    --metric_for_best_model eval_accuracy \
    $ADDITIONAL_ARGS \
    | tee $OUTPUT_DIR/trainer.log && \
conda activate eval && \
bash scripts/test.sh \
    --domains math,code \
    --model_path "$TMP_DIR" \
    --output_path "$TMP_DIR/test-math,code-4" \
    --repeats 4 \
    --tp 1
"

python src/async_write.py \
    --command "$training_command" \
    --disk_dir $OUTPUT_DIR \
    --mem_dir $TMP_DIR \
    --poll_interval_sec 2 \
    --checkpoint_stable_sec 6 \
    --backlog_dir_limit 30 \
    --sync_concurrency 3 \
    | tee $OUTPUT_DIR/writer.log

rm -r $TMP_DIR
# rm -r $OUTPUT_DIR/checkpoint-*
