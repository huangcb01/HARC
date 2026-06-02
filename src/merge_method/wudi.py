import torch
from typing import Literal
from tqdm import tqdm
from torch import Tensor

from file import LazyTensorLoader, TensorWriter


def wudi(
    source_model_loaders: list[LazyTensorLoader],
    base_model_loader: LazyTensorLoader,
    writer: TensorWriter,
    output_path: str,
    device: str,
    dtype: torch.dtype,
    iter_num: int = 300,
    nonlinear_merge_method: Literal["base", "average", "sum"] = "base",
    nonlinear_layers: list[str] = ["lm_head", "embed_tokens", "norm", "bias"],
    balance_method: Literal["none", "pre_model"] = "none",
):
    """WUDI-Merging
    Whoever Started the Interference Should End It: Guiding Data-Free Model Merging via Task Vectors (https://arxiv.org/abs/2503.08099)

    Args:
        source_model_loaders: List of tensor loaders for the source models to be merged.
        base_model_loader: Tensor loader for the base model, used as reference in merging.
        writer: Writer object to save merged tensors to the output path.
        device: Device identifier (e.g., 'cpu', 'cuda') for tensor operations.
        dtype: Data type for tensors during merging.
        iter_num: Number of iterations for merging process. Default is 300.
        nonlinear_merge_method: Method for merging nonlinear layers.
            "base" uses the base model's parameters, "average" averages across models, "sum" sums across models. Default is "base".
        nonlinear_layers: List of regex patterns specifying which layers are considered nonlinear and merged differently.
            Default includes classifier, bias, LayerNorm, and embeddings layers.
        balance_method: Method for balancing task vectors during merging.
            "none" applies no balancing, "pre_model" balances before merging. Default is "none".
    """
    tensor_names = source_model_loaders[0].get_keys()
    n_models = len(source_model_loaders)

    # Balance task vectors
    if balance_method == "pre_model":
        tv_magnitude_sum = torch.zeros((n_models,), dtype=dtype, device=device)
        n_elements = 0
        for name in tqdm(tensor_names):
            base_tensor = base_model_loader.get_tensor(name, device)
            n_elements += base_tensor.numel()
            for i, loader in enumerate(source_model_loaders):
                tensor = loader.get_tensor(name, device)
                tv_magnitude_sum[i] += (tensor - base_tensor).abs().sum()
        tv_magnitude_mean = tv_magnitude_sum / n_elements
        tv_magnitude_mean_mean = tv_magnitude_mean.mean()
        balance_weights = tv_magnitude_mean_mean / tv_magnitude_mean
        print("Balance weights:", balance_weights)
    else:
        balance_weights = torch.ones((n_models,), dtype=dtype, device=device)

    # Merge
    for i, name in enumerate(tqdm(tensor_names, "Merging")):
        is_nonlinear = any(x in name for x in nonlinear_layers)
        base_tensor = base_model_loader.get_tensor(name, device, dtype)
        if is_nonlinear and nonlinear_merge_method == "base":
            writer.save_tensor(name, base_tensor)
            continue
        task_vectors = torch.stack(
            [loader.get_tensor(name, device, dtype) for loader in source_model_loaders]
        ) - base_model_loader.get_tensor(name, device, dtype).unsqueeze(0)
        task_vectors = task_vectors * balance_weights.view(-1, *(1,) * (task_vectors.dim() - 1))
        if is_nonlinear:
            if nonlinear_merge_method == "average":
                new_tensor = task_vectors.mean(dim=0)
            elif nonlinear_merge_method == "sum":
                new_tensor = task_vectors.sum(dim=0)
            else:
                raise ValueError(f"Invalid nonlinear merge method {nonlinear_merge_method}.")
        else:
            new_tensor = get_merge_vector(task_vectors, iter_num)
        writer.save_tensor(name, base_tensor + new_tensor)
    return {}


def get_merge_vector(vectors: Tensor, iter_num: int = 300):
    merged_vector = torch.nn.Parameter(vectors.sum(dim=0))
    optimizer = torch.optim.Adam([merged_vector], lr=2e-5)
    l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=1))
    for _ in range(iter_num):
        inner_product = torch.matmul(merged_vector.unsqueeze(0) - vectors, vectors.transpose(1, 2))
        loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return merged_vector.data.detach()
