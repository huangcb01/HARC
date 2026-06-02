import os
import numpy as np
import torch
from typing import Literal
from tqdm import tqdm
from torch import Tensor
from functools import partial

from file import LazyTensorLoader, TensorWriter
from visualize import create_param_heatmap
from .utils import (
    get_expert_num,
    get_layer_num,
    group_by_expert,
    group_by_expert_sublayer,
    group_by_sublayer,
    group_by_layer,
    get_sign_consensus_mask,
)


def gta(
    source_model_loaders: list[LazyTensorLoader],
    base_model_loader: LazyTensorLoader,
    writer: TensorWriter,
    output_path: str,
    device: str,
    dtype: torch.dtype,
    int8_mask: bool = False,
    method: Literal["task_arithmetic", "ties", "dare_ties", "sce"] = "task_arithmetic",
    group: Literal["tensor", "expert", "expert_sublayer", "sublayer", "layer", "model"] = "model",
    density: float = 1.0,
    rescale: bool | None = None,
    normalize: bool | None = None,
    binarize: bool = False,
    model_weights: Literal["equal", "auto"] | list[float] | Tensor = "equal",
    balance_method: Literal["none", "pre_model", "post_model", "component", "tensor"] = "none",
    sparsify_ignore: list[str] = [],
    sign_ignore: list[str] = [],
):
    if rescale is None:
        rescale = True if method in {"dare_ties", "dare_linear", "della", "della_linear"} else False
    if normalize is None:
        normalize = True if method in {"ties", "della", "sce"} else False
    sparsify_fn = partial(sparsify, method=method, density=density, rescale=rescale, int8_mask=int8_mask)

    # Divide tensors into groups
    tensor_names = source_model_loaders[0].get_keys()
    n_models = len(source_model_loaders)
    n_tensors = len(tensor_names)
    n_layers = get_layer_num(tensor_names)
    n_experts = get_expert_num(tensor_names)
    if group == "tensor":
        groups = [[name] for name in tensor_names]
    elif group == "expert":
        layers = group_by_layer(tensor_names)
        groups = []
        for layer in layers:
            groups.extend(group_by_expert(layer))
    elif group == "expert_sublayer":
        groups = group_by_expert_sublayer(tensor_names)
    elif group == "sublayer":
        groups = group_by_sublayer(tensor_names)
    elif group == "layer":
        groups = group_by_layer(tensor_names)
    elif group == "model":
        groups = [tensor_names]
    else:
        raise ValueError(f"`group` must be chosen from ['tensor', 'expert', 'sublayer', 'layer', 'model']")

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

    # Calculate merging weights
    if model_weights == "equal":
        model_weights = torch.ones((n_models, n_tensors), dtype=dtype, device=device) / n_models
    elif model_weights == "auto":
        model_weights = torch.zeros((n_models, n_tensors), dtype=dtype, device=device) / n_models
        attn_components = [
            "input_layernorm",
            "self_attn.q_proj",
            "self_attn.q_norm",
            "self_attn.k_proj",
            "self_attn.k_norm",
            "self_attn.v_proj",
            "post_attention_layernorm",
        ]
        mlp_components = ["mlp.gate"] + [f"mlp.experts.{i}." for i in range(n_experts)]
        all_components = attn_components + mlp_components
        n_components = len(all_components)
        tv_magnitude_sum = torch.zeros((2, n_layers, n_components), dtype=dtype, device=device)
        n_elements = torch.zeros((n_layers, n_components), dtype=torch.int32, device=device)
        for group_names in tqdm(groups, "Calculating weights"):
            task_vectors = [
                torch.stack([loader.get_tensor(name, device, dtype) for loader in source_model_loaders])
                - base_model_loader.get_tensor(name, device, dtype).unsqueeze(0)
                for name in group_names
            ]
            task_vectors = [
                task_vector if any(x in name for x in sparsify_ignore) else sparsify_fn(task_vector)
                for task_vector, name in zip(task_vectors, group_names)
            ]
            task_vectors = [
                task_vector * balance_weights.view(-1, *(1,) * (task_vector.dim() - 1))
                for task_vector in task_vectors
            ]
            group_weights = calculate_weights(task_vectors)  # shape=(n_models,)
            for i, name in enumerate(group_names):
                idx = tensor_names.index(name)
                model_weights[:, idx] = group_weights
                layer_idx = get_layer_num([name]) - 1
                component_idx = next((j for j, component in enumerate(all_components) if component in name), None)
                if component_idx is not None:
                    tv_magnitude_sum[:, layer_idx, component_idx] += (
                        task_vectors[i].abs().sum(dim=list(range(1, task_vectors[i].dim())))
                    )
                    n_elements[layer_idx][component_idx] += task_vectors[i][0].numel()
        tv_magnitude_mean = tv_magnitude_sum / n_elements[np.newaxis, :]
        tv_magnitude_mean = tv_magnitude_mean.float().numpy(force=True)

        # Balance weights if needed
        if balance_method == "post_model":
            model_weight_mean = model_weights.mean(dim=1, keepdim=True)
            model_weight_mean_mean = model_weight_mean.mean(dim=0, keepdim=True)
            model_weights = model_weights / model_weight_mean * model_weight_mean_mean
        else:
            NotImplementedError()

        # Binarize weights if needed
        if binarize:
            max_indices = torch.argmax(model_weights, dim=0)
            model_weights.zero_()
            model_weights.scatter_(0, max_indices.unsqueeze(0), 1)

        # Visualize
        model_weight_mean = np.zeros(shape=(n_models, n_layers, n_components), dtype=np.float32)
        for i in range(n_layers):
            for j in range(n_components):
                component_tensor_indices = [
                    k for k, name in enumerate(tensor_names) if f"layers.{i}." in name and all_components[j] in name
                ]
                model_weight_mean[:, i, j] = (
                    model_weights[:, component_tensor_indices].mean(dim=1).float().numpy(force=True)
                )
        tv_magnitude_mean_mean = tv_magnitude_mean.mean()
        tv_magnitude_mean_range = max(
            tv_magnitude_mean.max() - tv_magnitude_mean_mean, tv_magnitude_mean_mean - tv_magnitude_mean.min()
        )
        model_weight_center = 1 / n_models
        for i in range(n_models):
            create_param_heatmap(
                tv_magnitude_mean[i],
                os.path.join(output_path, "task_vector_magnitude", f"model{i}"),
                attn_components,
                mlp_components,
                f"Model {i} Task Vector Magnitude",
                vmin=tv_magnitude_mean_mean - tv_magnitude_mean_range,
                vmax=tv_magnitude_mean_mean + tv_magnitude_mean_range,
                center=tv_magnitude_mean_mean,
            )
            create_param_heatmap(
                model_weight_mean[i],
                os.path.join(output_path, "merging_weights", f"model{i}"),
                attn_components,
                mlp_components,
                f"Model {i} Merging Weights",
                vmin=0,
                vmax=model_weight_center * 2,
                center=model_weight_center,
            )
        create_param_heatmap(
            tv_magnitude_mean,
            os.path.join(output_path, "task_vector_magnitude", f"all_models"),
            attn_components,
            mlp_components,
            f"Model Task Vector Magnitude",
            figsize=(40, 20),
            vmin=tv_magnitude_mean_mean - tv_magnitude_mean_range,
            vmax=tv_magnitude_mean_mean + tv_magnitude_mean_range,
            center=tv_magnitude_mean_mean,
        )
    else:
        model_weights = torch.tensor(model_weights, dtype=dtype, device=device)

    # Get sign consensus and mix deltas
    for i, name in enumerate(tqdm(tensor_names, "Merging")):
        base_tensor = base_model_loader.get_tensor(name, device, dtype)
        task_vectors = torch.stack([loader.get_tensor(name, device, dtype) for loader in source_model_loaders]) - base_model_loader.get_tensor(name, device, dtype).unsqueeze(0)
        if not any(x in name for x in sparsify_ignore):
            task_vectors = sparsify_fn(task_vectors)
        task_vectors = task_vectors * balance_weights.view(-1, *(1,) * (task_vectors.dim() - 1))
        weights = model_weights[:, i].view(-1, *(1,) * (task_vectors.dim() - 1))
        if method in {"ties", "dare_ties", "breadcrumbs_ties", "della"} and not any(x in name for x in sign_ignore):
            weights = weights * get_sign_consensus_mask(task_vectors, "sum", int8_mask)
        merge_task_vector = (task_vectors * weights).sum(dim=0)
        if normalize:
            merge_task_vector /= weights.sum(dim=0).clamp(min=1e-8)
        new_tensor = base_tensor + merge_task_vector
        writer.save_tensor(name, new_tensor)
    return {}


def sparsify(
    task_vectors: torch.Tensor,
    method: str,
    density: float,
    rescale: bool,
    int8_mask: bool,
) -> torch.Tensor:
    mask_dtype = torch.int8 if int8_mask else task_vectors.dtype
    n_models = task_vectors.shape[0]
    k = int(density * task_vectors[0].numel())
    if k <= 0:
        return torch.zeros_like(task_vectors, dtype=task_vectors.dtype, device=task_vectors.device)
    if method == "task_arithmetic" or density >= 1:
        return task_vectors
    elif method == "ties":
        _, indices = task_vectors.abs().view(n_models, -1).topk(k)
        mask = torch.zeros_like(task_vectors, dtype=mask_dtype, device=task_vectors.device)
        mask.view(n_models, -1)[torch.arange(n_models).unsqueeze(1), indices] = 1
    elif method in {"dare_ties", "dare_linear"}:
        mask = torch.full_like(task_vectors, fill_value=density, dtype=task_vectors.dtype).bernoulli()
    elif method == "sce":
        var = task_vectors.var(dim=0, unbiased=False)
        _, indices = torch.topk(var.abs().view(-1), k=k, largest=True)
        mask = torch.zeros_like(var, dtype=mask_dtype, device=task_vectors.device)
        mask.view(-1)[indices] = 1
        mask = mask.unsqueeze(dim=0)
    else:
        raise ValueError(f"invalid method {method}")

    masked = task_vectors * mask
    if rescale:
        for i in range(n_models):
            before_scale = task_vectors[i].abs().sum()
            after_scale = masked[i].abs().sum()
            if before_scale > 1e-10 and after_scale > 1e-10:
                masked[i] *= before_scale / after_scale
    return masked


def calculate_weights(task_vectors: list[Tensor]) -> Tensor:  # type: ignore
    task_vectors: Tensor = torch.cat([tv.view(2, -1) for tv in task_vectors], dim=1)
    weights = torch.mean(task_vectors**2, dim=1)
    weight_sum = torch.sum(weights).item()
    if abs(weight_sum) < 1e-10:
        weights = torch.ones_like(weights, device=weights.device, dtype=task_vectors[0].dtype) / weights.shape[0]
    else:
        weights /= weight_sum
    return weights
