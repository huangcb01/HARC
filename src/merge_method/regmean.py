import torch
import torch.nn as nn
import json
from typing import Any, cast
from collections import defaultdict
from tqdm import tqdm

from file import LazyTensorLoader, TensorWriter
from utils import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = get_logger(__name__)


def load_jsonl(file_path, max_samples=None):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if line.strip():
                data.append(json.loads(line))
    return data


def create_gram_hooks(model):
    grams = defaultdict(lambda: torch.tensor(0.0, dtype=torch.float32))
    counts = defaultdict(int)
    hooks = {}

    def get_gram_hook(name):
        def hook(module, input, output):
            x = input[0].detach()  # [b, t, h]
            x = x.view(-1, x.size(-1)).to(torch.float32)  # [b*t, h]
            if x.size(0) != 0:
                xtx = torch.matmul(x.transpose(0, 1), x)  # [h, h]
                grams[name] = (grams[name] * counts[name] + xtx) / (x.size(0) + counts[name])
                counts[name] += x.size(0)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = module.register_forward_hook(get_gram_hook(name))
            hooks[name] = hook

    return grams, counts, hooks


def remove_hooks(hooks):
    for hook in hooks.values():
        hook.remove()


def compute_grams(
    source_model_loaders: list[LazyTensorLoader],
    datasets: dict[str, list[str]],
    max_samples_per_domain: int,
    device: str
):
    all_grams = []

    for model_idx, loader in enumerate(source_model_loaders):
        # 检查是否有对应的数据集
        model_key = str(model_idx)
        if model_key not in datasets:
            raise ValueError(f"No dataset found for model {model_idx} (key '{model_key}' not in datasets)")

        data_files = datasets[model_key]

        # 使用 transformers 加载模型
        model_path = loader.index.base_path
        logger.info(f"Loading model {model_idx} from {model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        grams, counts, hooks = create_gram_hooks(model)

        for data_file in data_files:
            logger.info(f"Processing dataset: {data_file}")
            data = load_jsonl(data_file, max_samples_per_domain)

            for item in tqdm(data, desc=f"Model {model_idx}"):
                inputs = tokenizer.apply_chat_template(item["messages"], return_tensors="pt", return_dict=True).to(device)
                with torch.no_grad():
                    model(**inputs)

        remove_hooks(hooks)

        logger.info(f"Computed {len(grams)} Gram matrices for model {model_idx}")
        all_grams.append(dict(grams))

        del model
        torch.cuda.empty_cache() if device.startswith("cuda") else None

    return all_grams


def regmean(
    source_model_loaders: list[LazyTensorLoader],
    base_model_loader: LazyTensorLoader,
    writer: TensorWriter,
    output_path: str,
    device: str,
    dtype: torch.dtype,
    datasets: dict[str, list[str]],
    max_samples_per_domain: int,
    reduce_non_diag_a: float = 1.0,
) -> dict[str, Any]:
    logger.info("Computing Gram matrices for all source models...")
    all_grams = compute_grams(
        source_model_loaders,
        datasets,
        max_samples_per_domain,
        device,
    )

    logger.info("Merging models using RegMean...")
    tensor_names = list(source_model_loaders[0].index.tensor_paths.keys())

    for tensor_name in tqdm(tensor_names, desc="Merging"):
        if tensor_name.endswith(".weight"):
            module_name = tensor_name[: -len(".weight")]

            has_grams = [module_name in grams for grams in all_grams]

            if not any(has_grams):
                logger.warning(f"No Gram matrix found for {tensor_name}, using simple average")
                source_tensors = [loader.get_tensor(tensor_name, device, dtype) for loader in source_model_loaders]
                stacked = torch.stack(source_tensors)
                new_tensor = stacked.mean(dim=0)
                writer.save_tensor(tensor_name, new_tensor)
                continue

            missing_indices = [i for i, has_g in enumerate(has_grams) if not has_g]
            if missing_indices:
                raise ValueError(f"Gram matrix missing for models {missing_indices} at tensor {tensor_name}")

            source_tensors = [loader.get_tensor(tensor_name, device, dtype) for loader in source_model_loaders]
            grams = [grams_dict[module_name].to(device) for grams_dict in all_grams]

            if reduce_non_diag_a != 0.0:
                for i, gram in enumerate(grams):
                    grams[i] = _reduce_non_diag(gram, reduce_non_diag_a)

            sum_gram = sum(grams)
            sum_gram_m_ws = sum(
                [torch.matmul(gram, weight.transpose(0, 1).to(torch.float32)) for gram, weight in zip(grams, source_tensors)]
            )

            try:
                sum_gram_inv = torch.pinverse(sum_gram)
                wt = torch.matmul(sum_gram_inv, sum_gram_m_ws)
                new_tensor = wt.transpose(0, 1)
            except RuntimeError as e:
                print(sum_gram)
                logger.error(f"Failed to compute inverse for {tensor_name}: {e}")
                raise

            writer.save_tensor(tensor_name, new_tensor)
        else:
            source_tensors = [loader.get_tensor(tensor_name, device, dtype) for loader in source_model_loaders]
            stacked = torch.stack(source_tensors)
            new_tensor = stacked.mean(dim=0)
            writer.save_tensor(tensor_name, new_tensor)

    logger.info("RegMean merging completed!")
    return {}

def _reduce_non_diag(cov_mat: torch.Tensor, a: float):
    diag_weight = torch.diag(torch.ones(cov_mat.size(0)) - a).to(cov_mat.device)
    non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
    weight = diag_weight + non_diag_weight
    ret = cov_mat * weight
    return ret
