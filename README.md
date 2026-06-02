# HARC: Hessian-Aware Router Calibration for MoE Model Merging


**HARC** is a training-free framework for calibrating routers when merging Mixture-of-Experts (MoE) models. It is the official implementation of our ICML 2026 paper "**When Model Merging Breaks Routing: Training-Free Calibration for MoE**".

## Quick Start

### 1. Environment

This project is built on top of [HuggingFace Transformers](https://github.com/huggingface/transformers), [vLLM](https://github.com/vllm-project/vllm), and [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory). We recommend using Python ≥ 3.10 and PyTorch ≥ 2.1.

```bash
# Clone the repository
git clone https://github.com/huangcb01/HARC.git
cd HARC

# Install dependencies
pip install torch transformers accelerate datasets fire safetensors numpy pydantic psutil rich tqdm vllm deepspeed liger-kernel llamafactory
```

### 2. Fine-Tune Source MoE Models

Use the provided scripts to fine-tune domain-specific MoE models (e.g., math, code, IF) from a base MoE model such as [OLMoE-1B-7B-0125](https://huggingface.co/allenai/OLMoE-1B-7B-0125):

```bash
# Fine-tune on math
bash scripts/train/sft_olmoe.sh   # edit DATASET_TYPE=math inside

# Fine-tune on code
bash scripts/train/sft_olmoe.sh   # edit DATASET_TYPE=code inside
```

These scripts use DeepSpeed ZeRO-2 and Liger Kernel for efficient training.

### 3. Merge Models (Two-Stage Pipeline)

HARC follows a **two-stage** merging pipeline:

#### Stage 1: Merge Non-Router Parameters

First, merge the source models using any existing dense merging method, such as:

```bash
bash scripts/merging/wudi.sh
```

#### Stage 2: Calibrate the Router with HARC

Then, apply HARC to calibrate the router weights using a small set of calibration data:

```bash
bash scripts/merging/router_calibration_cg.sh
```

## Evaluation

We evaluate on **mathematical reasoning** and **code generation** benchmarks:

- **Math:** GSM8K, MATH-500
- **Code:** HumanEval+, MBPP+

```bash
# Evaluate a merged model
bash scripts/test.sh \
    --domains '["math","code"]' \
    --model_path <path-to-model> \
    --output_path <output-dir> \
    --repeats 4 \
    --tp 1
```

Evaluation configs are in `config/eval_tasks/`.
