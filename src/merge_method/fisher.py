import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from typing import Any, cast
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

from file import LazyTensorLoader, TensorWriter
from utils import get_logger

logger = get_logger(__name__)


def compute_fisher(
    source_model_loaders: list[LazyTensorLoader],
    datasets: dict[str, list[str]],
    max_samples_per_domain: int,
    device: str,
    batch_size: int = 64,
):
    all_fisher = []

    for model_idx, loader in enumerate(source_model_loaders):
        # 检查是否有对应的数据集
        model_key = str(model_idx)
        if model_key not in datasets:
            raise ValueError(f"No dataset found for model {model_idx} (key '{model_key}' not in datasets)")

        model_path = loader.index.base_path
        logger.info(f"Loading model {model_idx} from {model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        fisher_dict = defaultdict(lambda: torch.tensor(0.0))
        counts = defaultdict(int)

        dataset = load_dataset("json", data_files=datasets[model_key], split="train")
        dataset = dataset.map(tokenize, fn_kwargs={"tokenizer": tokenizer}, num_proc=os.cpu_count() or 1)
        collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=4)

        for batch in tqdm(data_loader, desc=f"Model {model_idx}"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            model.zero_grad()
            outputs.loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad = param.grad.detach()
                    fisher = grad**2
                    fisher_dict[name] = (fisher_dict[name] * counts[name] + fisher) / (counts[name] + 1)
                    counts[name] += 1

        logger.info(f"Computed {len(fisher_dict)} Fisher matrices for model {model_idx}")
        all_fisher.append(dict(fisher_dict))

        del model
        torch.cuda.empty_cache() if device.startswith("cuda") else None

    return all_fisher


def fisher(
    source_model_loaders: list[LazyTensorLoader],
    base_model_loader: LazyTensorLoader,
    writer: TensorWriter,
    output_path: str,
    device: str,
    dtype: torch.dtype,
    datasets: dict[str, list[str]],
    max_samples_per_domain: int,
    smooth: float = 1e-10,
    fisher_normalize: str = None,
    model_coeffs: list[float] = None,
) -> dict[str, Any]:
    logger.info("Computing Fisher information for all source models...")
    all_fisher = compute_fisher(source_model_loaders, datasets, max_samples_per_domain, device)

    logger.info("Merging models using Fisher-Merging...")
    tensor_names = list(source_model_loaders[0].index.tensor_paths.keys())

    num_models = len(source_model_loaders)
    if model_coeffs is None:
        model_coeffs = [1.0 / num_models] * num_models

    for tensor_name in tqdm(tensor_names, desc="Merging"):
        has_fisher = [tensor_name in fisher_dict for fisher_dict in all_fisher]

        if not any(has_fisher):
            logger.warning(f"No Fisher information found for {tensor_name}, using simple average")
            source_tensors = [loader.get_tensor(tensor_name, device, dtype) for loader in source_model_loaders]
            stacked = torch.stack(source_tensors)
            new_tensor = stacked.mean(dim=0)
            writer.save_tensor(tensor_name, new_tensor)
            continue

        missing_indices = [i for i, has_f in enumerate(has_fisher) if not has_f]
        if missing_indices:
            raise ValueError(f"Fisher information missing for models {missing_indices} at tensor {tensor_name}")

        source_tensors = [loader.get_tensor(tensor_name, device, dtype) for loader in source_model_loaders]
        fisher_weights = [fisher_dict[tensor_name].to(device, dtype) for fisher_dict in all_fisher]

        params = torch.stack(source_tensors)  # [N, ...]
        fisher = torch.stack(fisher_weights)  # [N, ...]

        fisher = fisher + smooth

        coeffs_tensor = torch.tensor(model_coeffs, device=device, dtype=dtype)
        coeff_shape = [len(model_coeffs)] + [1] * (len(params.shape) - 1)
        coeff_tensor = coeffs_tensor.view(*coeff_shape)

        if fisher_normalize is not None:
            dims = list(range(1, len(params.shape)))

            if fisher_normalize == "param":
                fisher_norm = torch.norm(fisher, dim=dims, keepdim=True)  # [N, 1, 1, ...]
            elif fisher_normalize == "model":
                model_dims = list(range(1, len(fisher.shape)))
                fisher_norm = torch.norm(fisher, dim=model_dims)  # [N]
                for _ in range(len(fisher.shape) - 1):
                    fisher_norm = fisher_norm.unsqueeze(-1)
            else:
                raise ValueError(f"Invalid fisher_normalize value: {fisher_normalize}. Must be 'param' or 'model'")

            fisher_norm_coeff = 1.0 / (fisher_norm + smooth)
            coeff_tensor = coeff_tensor * fisher_norm_coeff

        # w_merged = Σ(c_i * F_i * w_i) / Σ(c_i * F_i)
        sum_p = (params * fisher * coeff_tensor).sum(0)
        denom = (fisher * coeff_tensor).sum(0)
        new_tensor = sum_p / denom

        writer.save_tensor(tensor_name, new_tensor)

    logger.info("Fisher-Merging completed!")
    return {}


def tokenize(item, tokenizer):
    input_ids = tokenizer.apply_chat_template(item["messages"])
    prompt_input_ids = tokenizer.apply_chat_template([item["messages"][0]], add_generation_prompt=True)
    prompt_len = len(prompt_input_ids)
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return {"input_ids": input_ids, "labels": labels}