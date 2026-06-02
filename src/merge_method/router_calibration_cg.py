import os
import sys
import torch
import multiprocessing as mp
from torch import Tensor, nn
from typing import Any, Callable, Literal, cast
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeSdpaAttention,
    OlmoeRMSNorm,
    OlmoeRotaryEmbedding,
    OlmoeSparseMoeBlock,
)
from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
from tqdm import tqdm, trange

from file import LazyTensorLoader, TensorWriter
from utils import get_logger

logger = get_logger(__name__)


def router_calibration_cg(
    source_model_loaders: list[LazyTensorLoader],
    base_model_loader: LazyTensorLoader,
    writer: TensorWriter,
    output_path: str,
    device: str,
    dtype: torch.dtype,
    datasets: dict[str, list[str]],
    max_samples_per_domain: int | float,
    batch_size: int = 32,
    regularization: float = 1.0,
    w_init: Literal["zero", "origin", "average"] = "origin",
    preconditioner_type: Literal["none", "diagonal", "kronecker"] = "none",
    cg_max_iter: int = 100,
    cg_tol: float = 1e-6,
) -> dict[str, Any]:
    assert len(source_model_loaders) == len(datasets), "Number of source models must be the same of dataset domains."

    # Get configs
    model_path = base_model_loader.index.base_path
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    max_length = cast(int, model_config.max_position_embeddings)
    num_hidden_layers = cast(int, model_config.num_hidden_layers)
    num_experts = cast(int, model_config.num_experts)
    hidden_size = cast(int, model_config.hidden_size)
    num_domains = len(datasets)
    n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    tensor_keys = base_model_loader.get_keys()

    # Preprocess data
    logger.info("Preprocessing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    domain_datasets = {}
    total_tokens = []
    for domain, dataset_paths in tqdm(datasets.items()):
        dataset = load_dataset("json", data_files=dataset_paths, split="train")
        if isinstance(max_samples_per_domain, float) and 0 < max_samples_per_domain < 1:
            max_samples = int(len(dataset) * max_samples_per_domain)
        else:
            max_samples = int(max_samples_per_domain)
        dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))
        dataset = dataset.map(
            _tokenize,
            batched=False,
            remove_columns=dataset.column_names,
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
            num_proc=os.cpu_count(),
        )
        domain_datasets[domain] = dataset
        total_tokens.append(sum(dataset["length"]))
    del tokenizer

    # Setup worker context and pool
    pad_token_id = getattr(model_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model_config, "eos_token_id", 0) or 0
    worker_ctx = WorkerContext(
        cast(OlmoeConfig, model_config),
        dtype,
        batch_size=batch_size,
        pad_token_id=int(pad_token_id),
    )
    gpu_queue = mp.Queue()
    for i in range(n_devices):
        gpu_queue.put(i)
    pool = mp.Pool(processes=n_devices, initializer=_init_worker, initargs=(worker_ctx, gpu_queue))

    # Define helper functions.
    def is_router(tensor_name: str):
        return "gate.weight" in tensor_name

    def is_attn_layer(tensor_name: str):
        return any(key in tensor_name for key in ("self_attn", "input_layernorm", "post_attention_layernorm"))

    def is_mlp_layer(tensor_name: str):
        return "mlp" in tensor_name and not is_router(tensor_name)

    def merge_router_weights_with_calibration(layer_idx: int = 0):
        tensor_name = f"model.layers.{layer_idx}.mlp.gate.weight"
        original_weight = base_model_loader.get_tensor(tensor_name, device, dtype)
        source_weights = [
            source_model_loaders[domain_idx].get_tensor(tensor_name, device, dtype) for domain_idx in range(num_domains)
        ]

        def compute_sigma_w(W: Tensor | None = None, return_diagonal: bool = False):
            if W is not None and W.dim() == 1:  # CG 传入的是向量化的参数
                W = W.view(num_experts, hidden_size)

            result = torch.zeros((num_experts * hidden_size,), dtype=torch.float64, device=device)
            diagonal_accum = torch.zeros((num_experts * hidden_size,), dtype=torch.float64, device=device)

            for domain_idx in range(num_domains):
                Wi = source_weights[domain_idx]
                target_W = W if W is not None else Wi

                results = pool.starmap(_compute_sigma_w_worker, ((domain_idx, Wi, target_W) for _ in range(n_devices)))
                sigma_w_domain = sum(r[0] / total_tokens[domain_idx] for r in results)
                diagonal_domain = sum(r[1] / total_tokens[domain_idx] for r in results)

                if regularization > 1.0:  # Sigma_reg @ W = Sigma @ W + (gamma - 1) * diag(Sigma) * W
                    sigma_w_domain = sigma_w_domain + (regularization - 1.0) * diagonal_domain * target_W.view(-1)
                    diagonal_domain *= regularization
                elif regularization < 1.0:  # Sigma_reg @ W = gamma * Sigma @ W + (1 - gamma) * diag(Sigma) * W
                    sigma_w_domain = regularization * sigma_w_domain + (
                        1.0 - regularization
                    ) * diagonal_domain * target_W.view(-1)
                else:
                    pass
                result += sigma_w_domain
                diagonal_accum += diagonal_domain

            if return_diagonal:
                return result.to(dtype=dtype), diagonal_accum.to(dtype=dtype)
            return result.to(dtype=dtype)

        logger.info("Computing right-hand side and diagonal...")
        rhs, diagonal_for_precond = compute_sigma_w(W=None, return_diagonal=True)

        preconditioner: Callable[[Tensor], Tensor] | None = None

        if preconditioner_type == "diagonal":
            logger.info("Setting up diagonal preconditioner...")
            diagonal = diagonal_for_precond

            logger.info(f"Diagonal statistics before normalization: "
                       f"min={diagonal.min().item():.2e}, "
                       f"max={diagonal.max().item():.2e}, "
                       f"mean={diagonal.mean().item():.2e}, "
                       f"median={diagonal.median().item():.2e}, "
                       f"std={diagonal.std().item():.2e}")

            epsilon = 1e-10
            diagonal = torch.clamp(diagonal, min=epsilon)

            median_val = torch.median(diagonal)
            diagonal = diagonal / median_val

            logger.info(f"Diagonal statistics after normalization: "
                       f"min={diagonal.min().item():.2e}, "
                       f"max={diagonal.max().item():.2e}, "
                       f"mean={diagonal.mean().item():.2e}, "
                       f"median={diagonal.median().item():.2e}")

            preconditioner = lambda r: r / diagonal
        elif preconditioner_type == "kronecker":            # Kronecker 预处理
            logger.info("Computing Kronecker preconditioner...")
            H_bar_accum: Tensor = torch.zeros((num_experts, num_experts), dtype=dtype, device=device)
            C_bar_accum: Tensor = torch.zeros((hidden_size, hidden_size), dtype=dtype, device=device)

            for domain_idx in range(num_domains):
                results = pool.starmap(
                    _compute_kronecker_worker_optimized,
                    ((domain_idx, source_weights[domain_idx]) for _ in range(n_devices)),
                )
                domain_H_sum = sum([H_sum for H_sum, C_sum in results])
                domain_C_sum = sum([C_sum for H_sum, C_sum in results])
                H_bar_accum += domain_H_sum / total_tokens[domain_idx]
                C_bar_accum += domain_C_sum / total_tokens[domain_idx]
            H_bar = H_bar_accum
            C_bar = C_bar_accum

            eps = max(regularization, 1e-8)
            H_bar_reg = H_bar + eps * torch.eye(num_experts, device=device, dtype=dtype)
            C_bar_reg = C_bar + eps * torch.eye(hidden_size, device=device, dtype=dtype)

            logger.info(f"H_bar stats - min_eig: {torch.linalg.eigvalsh(H_bar).min():.6e}, max_eig: {torch.linalg.eigvalsh(H_bar).max():.6e}")
            logger.info(f"C_bar stats - min_eig: {torch.linalg.eigvalsh(C_bar).min():.6e}, max_eig: {torch.linalg.eigvalsh(C_bar).max():.6e}")

            L_H = torch.linalg.cholesky(H_bar_reg)
            L_C = torch.linalg.cholesky(C_bar_reg)

            def apply(v: Tensor) -> Tensor:
                """应用 Kronecker 积预条件器: M^{-1} v = vec(C^{-1} V H^{-1})"""
                V = v.view(num_experts, hidden_size)  # (num_experts, hidden_size)

                Y = torch.linalg.solve_triangular(L_C, V.T, upper=False, left=True)
                Y = torch.linalg.solve_triangular(L_C.T, Y, upper=True, left=True)
                Y = Y.T  # shape=(num_experts, hidden_size)

                Z = torch.linalg.solve_triangular(L_H, Y.T, upper=False, left=True)
                Z = torch.linalg.solve_triangular(L_H.T, Z, upper=True, left=True)

                result = Z.T.reshape(-1)
                return result

            preconditioner = apply
            logger.info("Kronecker preconditioner computed successfully.")
        else:
            pass
        if w_init == "zero":
            x0 = torch.zeros_like(rhs, device=device)
        elif w_init == "origin":
            x0 = original_weight.view(-1)
        elif w_init == "average":
            x0 = (sum(source_weights) / num_domains).view(-1)
        else:
            raise ValueError(f"Unknown w_init method: {w_init}")
        x_sol = _conjugate_gradient(
            compute_sigma_w, rhs, x0, preconditioner, cg_max_iter, cg_tol, verbose=True
        )

        merged_router_weight = x_sol.view(num_experts, hidden_size)
        writer.save_tensor(tensor_name, merged_router_weight)
        return merged_router_weight

    # Merge embedding and the first attention layer
    logger.info("Merging embedding and the first attention layer...")
    state_dict = {"embed_tokens.weight": base_model_loader.get_tensor("model.embed_tokens.weight", "cpu", dtype)}
    writer.save_tensor("model.embed_tokens.weight", state_dict["embed_tokens.weight"])
    for key in tensor_keys:
        if "layers.0." in key and is_attn_layer(key):
            short_key = key.replace("model.layers.0.", "")
            state_dict[short_key] = base_model_loader.get_tensor(key, "cpu", dtype)
            writer.save_tensor(key, state_dict[short_key])

    # Forward embedding and the first attention layer
    logger.info("Forward embedding and the first attention layer...")
    for i, dataset in enumerate(domain_datasets.values()):
        chunk_size = (len(dataset) + n_devices - 1) // n_devices
        chunks = (
            (i, state_dict, dataset[j * chunk_size : (j + 1) * chunk_size]["input_ids"]) for j in range(n_devices)
        )
        pool.starmap(_forward_embedding_attn_worker, chunks)

    # Merge layers
    for layer_idx in range(num_hidden_layers - 1):
        state_dict = {}

        ## Merge MLP
        logger.info(f"Merging MLP for layer {layer_idx}...")
        state_dict["mlp.gate.weight"] = merge_router_weights_with_calibration(layer_idx)

        ## Merge Experts + Next Attention (pool-parallel)
        logger.info(f"Merging experts and attention for layer {layer_idx+1}...")
        for key in tensor_keys:
            if f"layers.{layer_idx+1}." in key and is_attn_layer(key):
                short_key = key.replace(f"model.layers.{layer_idx+1}.", "")
                state_dict[short_key] = base_model_loader.get_tensor(key, "cpu", dtype)
                writer.save_tensor(key, state_dict[short_key])
            elif f"layers.{layer_idx}." in key and is_mlp_layer(key):
                short_key = key.replace(f"model.layers.{layer_idx}.", "")
                state_dict[short_key] = base_model_loader.get_tensor(key, "cpu", dtype)
                writer.save_tensor(key, state_dict[short_key])

        ## Forward MLP and Attention
        logger.info(f"Forward MLP and Attention  for layer {layer_idx+1}...")
        for i in range(len(datasets)):
            pool.starmap(_forward_mlp_attn_worker, ((i, state_dict, layer_idx) for _ in range(n_devices)))

    # Merge the last MLP layer and LM head
    logger.info("Merge the last MLP layer and LM head...")
    merge_router_weights_with_calibration(num_hidden_layers - 1)
    for key in tensor_keys:
        if (
            f"layers.{num_hidden_layers-1}" in key
            and is_mlp_layer(key)
            or key in ("model.norm.weight", "lm_head.weight")
        ):
            writer.save_tensor(key, base_model_loader.get_tensor(key, "cpu", dtype))

    pool.close()
    pool.join()
    return model_config.to_dict()


class OlmoeEmbeddingAttention(nn.Module):
    def __init__(self, config: OlmoeConfig, device: str):
        super().__init__()
        self.config = config
        self.device = device
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id, device=device)
        self.position_embedding = OlmoeRotaryEmbedding(config, device)
        self.input_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = OlmoeSdpaAttention(config, 0)

    def forward(self, input_ids: Tensor):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            squeeze = True
        elif input_ids.dim() == 2:
            squeeze = False
        else:
            raise ValueError(f"input_ids must be 1D or 2D, got shape={tuple(input_ids.shape)}")

        bsz, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(bsz, -1)

        residual = self.embed_tokens(input_ids)
        position_embeddings = self.position_embedding(residual, position_ids)
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings)[0]
        residual = hidden_states + residual
        hidden_states = self.post_attention_layernorm(residual)

        if squeeze:
            return hidden_states.squeeze(0), residual.squeeze(0)
        return hidden_states, residual


class OlmoeMLPAttention(nn.Module):
    def __init__(self, config: OlmoeConfig, device: str, layer_idx: int):
        super().__init__()
        self.config = config
        self.device = device
        self.layer_idx = layer_idx
        self.position_embedding = OlmoeRotaryEmbedding(config, device)
        self.mlp = OlmoeSparseMoeBlock(config)
        self.input_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = OlmoeSdpaAttention(config, layer_idx + 1)
        self.post_attention_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: Tensor, residual: Tensor):
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            residual = residual.unsqueeze(0)
            squeeze = True
        elif hidden_states.dim() == 3:
            squeeze = False
        else:
            raise ValueError(f"hidden_states must be 2D or 3D, got shape={tuple(hidden_states.shape)}")

        hidden_states, _ = self.mlp(hidden_states)
        residual = hidden_states + residual
        hidden_states = self.input_layernorm(residual)
        bsz, seq_len, _ = hidden_states.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(bsz, -1)
        position_embeddings = self.position_embedding(hidden_states, position_ids)
        hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings)[0]
        residual = hidden_states + residual
        hidden_states = self.post_attention_layernorm(residual)

        if squeeze:
            return hidden_states.squeeze(0), residual.squeeze(0)
        return hidden_states, residual


@dataclass
class WorkerContext:
    config: OlmoeConfig
    dtype: torch.dtype
    batch_size: int = 32
    pad_token_id: int = 0
    device: str = "cpu"
    residual_tensors: list[list[Tensor]] = field(default_factory=list)
    hidden_state_tensors: list[list[Tensor]] = field(default_factory=list)


def _init_worker(worker_ctx: WorkerContext, gpu_queue: mp.Queue):
    global ctx
    ctx = worker_ctx
    gpu_id = gpu_queue.get()
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        ctx.device = f"cuda:{gpu_id}"
    else:
        ctx.device = "cpu"


def _forward_embedding_attn_worker(
    domain_idx: int,
    state_dict: dict[str, Tensor],
    chunk_input_ids: list[list[int]],
):
    embedding_attn = OlmoeEmbeddingAttention(ctx.config, ctx.device)
    embedding_attn.load_state_dict(state_dict)
    embedding_attn.to(device=ctx.device, dtype=ctx.dtype)
    embedding_attn.eval()
    ctx.hidden_state_tensors.append([])
    ctx.residual_tensors.append([])

    # Sort by length (desc) to minimize padding within each batch.
    chunk_input_ids.sort(key=len, reverse=True)

    bs = ctx.batch_size
    pad_id = ctx.pad_token_id

    for start in range(0, len(chunk_input_ids), bs):
        batch = chunk_input_ids[start : start + bs]
        lengths = [len(x) for x in batch]
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            continue

        input_batch = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=ctx.device)
        for i, ids in enumerate(batch):
            input_batch[i, : lengths[i]] = torch.tensor(
                ids, dtype=torch.long, device=ctx.device
            )  # shape=(bsz, seq_len)

        with torch.no_grad():
            # Forward.
            hidden_states_b, residual_b = embedding_attn.forward(input_batch)

            # Store per-sample (unpadded) tensors for the next stage.
            for i, l in enumerate(lengths):
                ctx.hidden_state_tensors[domain_idx].append(hidden_states_b[i, :l].detach())
                ctx.residual_tensors[domain_idx].append(residual_b[i, :l].detach())


def _forward_mlp_attn_worker(domain_idx: int, state_dict: dict, layer_idx: int):
    """Top-level worker for QKV forward sigmas computation."""
    mlp_attn = OlmoeMLPAttention(ctx.config, ctx.device, layer_idx)
    mlp_attn.load_state_dict(state_dict)
    mlp_attn.to(device=ctx.device, dtype=ctx.dtype)
    mlp_attn.eval()
    hidden_state_tensor = ctx.hidden_state_tensors[domain_idx]
    residual_tensor = ctx.residual_tensors[domain_idx]
    bs = ctx.batch_size
    n = len(residual_tensor)

    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch_hidden = hidden_state_tensor[start:end]
        batch_resid = residual_tensor[start:end]
        lengths = [t.shape[0] for t in batch_hidden]
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            continue

        dtype = batch_hidden[0].dtype
        hidden_b = torch.zeros((len(batch_hidden), max_len, ctx.config.hidden_size), device=ctx.device, dtype=dtype)
        resid_b = torch.zeros((len(batch_resid), max_len, ctx.config.hidden_size), device=ctx.device, dtype=dtype)
        for i, l in enumerate(lengths):
            hidden_b[i, :l] = batch_hidden[i]
            resid_b[i, :l] = batch_resid[i]

        with torch.no_grad():
            # Forward.
            hidden_states_b, residual_b = mlp_attn.forward(hidden_b, resid_b)

            # Store per-sample (unpadded) tensors for the next stage.
            for i, l in enumerate(lengths):
                hidden_state_tensor[start + i] = hidden_states_b[i, :l].detach()
                residual_tensor[start + i] = residual_b[i, :l].detach()
            del residual_b


def _compute_sigma_w_worker(domain_idx: int, Wi: Tensor, W: Tensor):
    num_experts = Wi.shape[0]
    hidden_size = Wi.shape[1]

    result = torch.zeros((hidden_size, num_experts), device=ctx.device, dtype=torch.float64)
    diagonal_accum: Tensor = torch.zeros((num_experts, hidden_size), device=ctx.device, dtype=torch.float64)

    Wi = Wi.to(ctx.device)
    W = W.to(ctx.device)

    for X in ctx.hidden_state_tensors[domain_idx]:
        Zi = X @ Wi.T  # shape=(num_tokens, num_experts)
        Ri = torch.softmax(Zi, dim=1)  # shape=(num_tokens, num_experts)

        # 计算 Sigma @ W
        if W is not None:
            Z = X @ W.T
        else:
            Z = Zi
        HiZ = Ri * Z - Ri * (Ri * Z).sum(dim=1, keepdim=True)  # batch of (Hi @ W @ x), shape=(num_tokens, num_experts)
        result += X.T @ HiZ  # shape=(hidden_size, num_experts)

        Ri_var = Ri * (1 - Ri)  # shape=(num_tokens, num_experts)
        X_sq = X * X  # shape=(num_tokens, hidden_size)
        diag_batch = Ri_var.T @ X_sq  # shape=(num_experts, hidden_size)
        diagonal_accum += diag_batch

    # 向量化
    sigma_w = result.T.contiguous().view(-1)  # shape=(num_experts * hidden_size,)
    diagonal = diagonal_accum.view(-1)  # shape=(num_experts * hidden_size,)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return sigma_w.to(device), diagonal.to(device)


def _compute_kronecker_worker_optimized(domain_idx: int, Wi: Tensor):
    num_experts = Wi.shape[0]
    hidden_size = Wi.shape[1]
    H_sum: Tensor = torch.zeros((num_experts, num_experts), device=ctx.device, dtype=ctx.dtype)
    C_sum: Tensor = torch.zeros((hidden_size, hidden_size), device=ctx.device, dtype=ctx.dtype)
    Wi = Wi.to(ctx.device)

    for X in ctx.hidden_state_tensors[domain_idx]:
        Zi = X @ Wi.T  # shape=(num_tokens, num_experts)
        Ri = torch.softmax(Zi, dim=1)  # shape=(num_tokens, num_experts)

        sum_R = Ri.sum(dim=0)  # shape=(num_experts,)
        sum_RRT = Ri.T @ Ri  # shape=(num_experts, num_experts)

        H_batch = sum_RRT - torch.diag(sum_R)  # shape=(num_experts, num_experts)

        H_sum += H_batch

        C_sum += X.T @ X  # shape=(hidden_size, hidden_size)

    return (
        H_sum.to("cuda:0" if torch.cuda.is_available() else "cpu"),
        C_sum.to("cuda:0" if torch.cuda.is_available() else "cpu"),
    )


def _tokenize(item, tokenizer, max_length):
    if "messages" in item:
        messages = item["messages"]
    else:
        messages = [
            {"role": "user" if turn["from"] == "human" else "assistant", "content": turn["value"]}
            for turn in item["conversations"]
        ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, truncation=True, max_length=max_length)
    return {"input_ids": input_ids, "length": len(input_ids)}


def _conjugate_gradient(
    A_matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> torch.Tensor:
    if x0 is None:
        x = torch.zeros_like(b, device=b.device)
    else:
        x = x0.clone()
    r = b - A_matvec(x)

    if preconditioner is not None:
        z = preconditioner(r)
    else:
        z = r.clone()

    p = z.clone()
    rz = torch.dot(r, z)

    b_norm = torch.norm(b)
    if b_norm == 0:
        logger.warning("Right-hand side b is zero vector; returning zero solution.")
        return x

    if verbose:
        logger.info(f"CG initial residual: {torch.norm(r):.6e}")

    for i in range(max_iter):
        Ap = A_matvec(p)
        pAp = torch.dot(p, Ap)

        if abs(pAp) < 1e-14:
            if verbose:
                logger.warning(f"Non-positive curvature detected at iteration {i}: pAp={pAp:.6e}")
            break

        alpha = rz / pAp
        x = x + alpha * p
        r_new = r - alpha * Ap

        residual_norm = torch.norm(r_new)
        relative_residual = residual_norm / b_norm
        if relative_residual < tol:
            break

        if preconditioner is not None:
            z_new = preconditioner(r_new)
        else:
            z_new = r_new
        rz_new = torch.dot(r_new, z_new)

        beta = rz_new / rz

        p = z_new + beta * p

        r = r_new
        z = z_new
        rz = rz_new

    else:
        if verbose:
            final_residual = torch.norm(b - A_matvec(x)) / b_norm
            logger.info(f"CG reached max iterations ({max_iter}), final relative residual: {final_residual:.6e}")

    return x