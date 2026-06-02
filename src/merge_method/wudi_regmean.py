import os
import torch
import multiprocessing as mp
from multiprocessing.pool import ApplyResult
from torch import Tensor, nn
from typing import Any, Literal
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from transformers.models.olmoe.modeling_olmoe import OlmoeSdpaAttention, OlmoeRMSNorm, OlmoeRotaryEmbedding, OlmoeSparseMoeBlock
from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
from tqdm import tqdm, trange

from file import LazyTensorLoader, TensorWriter


@dataclass
class WorkerContext:
    config: OlmoeConfig
    dtype: torch.dtype
    batch_size: int = 32
    pad_token_id: int = 0
    source_model_paths: list[str] = field(default_factory=list)
    base_model_path: str = ""
    lazy_unpickle: bool = True
    device: str = "cpu"
    residual_tensors: list[list[Tensor]] = field(default_factory=list)
    hidden_state_tensors: list[list[Tensor]] = field(default_factory=list)
    source_model_loaders: list[LazyTensorLoader] = field(default_factory=list)
    base_model_loader: LazyTensorLoader | None = None


def init_worker(worker_ctx: WorkerContext, gpu_queue: mp.Queue):
    global ctx
    ctx = worker_ctx
    gpu_id = gpu_queue.get()
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        ctx.device = f"cuda:{gpu_id}"
    else:
        ctx.device = "cpu"

    # Re-create loaders inside each process to avoid pickling heavy objects.
    ctx.source_model_loaders = [
        LazyTensorLoader.from_disk(path, lazy_unpickle=ctx.lazy_unpickle) for path in ctx.source_model_paths
    ]
    ctx.base_model_loader = LazyTensorLoader.from_disk(ctx.base_model_path, lazy_unpickle=ctx.lazy_unpickle)


def nonlinear_merge_worker(tensor_name: str, nonlinear_merge_method: Literal["base", "average", "sum"] = "base") -> tuple[str, Tensor]:
    if nonlinear_merge_method == "base":
        assert ctx.base_model_loader is not None
        merged_tensor = ctx.base_model_loader.get_tensor(tensor_name, ctx.device, ctx.dtype).detach().cpu()
    elif nonlinear_merge_method == "sum":
        source_tensors = [loader.get_tensor(tensor_name, ctx.device, ctx.dtype) for loader in ctx.source_model_loaders]
        merged_tensor = torch.stack(source_tensors, dim=0).sum(dim=0).detach().cpu()
    elif nonlinear_merge_method == "average":
        source_tensors = [loader.get_tensor(tensor_name, ctx.device, ctx.dtype) for loader in ctx.source_model_loaders]
        merged_tensor = torch.stack(source_tensors, dim=0).mean(dim=0).detach().cpu()
    return tensor_name, merged_tensor


def wudi_merge_worker(tensor_name: str, iter_num: int = 300, lr: float = 2e-5) -> tuple[str, Tensor]:
    assert ctx.base_model_loader is not None
    source_tensors = [loader.get_tensor(tensor_name, ctx.device, ctx.dtype) for loader in ctx.source_model_loaders]
    base_tensor = ctx.base_model_loader.get_tensor(tensor_name, ctx.device, ctx.dtype)
    task_vectors = torch.stack(source_tensors, dim=0) - base_tensor.unsqueeze(0)
    merged_vector = torch.nn.Parameter(task_vectors.sum(dim=0))
    optimizer = torch.optim.Adam([merged_vector], lr=lr)
    l2_norms = torch.square(torch.norm(task_vectors.reshape(task_vectors.shape[0], -1), p=2, dim=1))
    for _ in range(iter_num):
        inner_product = torch.matmul(merged_vector.unsqueeze(0) - task_vectors, task_vectors.transpose(1, 2))
        loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    merged_tensor = (base_tensor + merged_vector.detach()).detach().cpu()
    return tensor_name, merged_tensor


def forward_embedding_attn_worker(
    domain_idx: int,
    state_dict: dict[str, Tensor],
    chunk_input_ids: list[list[int]],
) -> Tensor:
    embedding_attn = OlmoeEmbeddingAttention(ctx.config, state_dict, ctx.device)
    ctx.hidden_state_tensors.append([])
    ctx.residual_tensors.append([])
    grams = torch.zeros((ctx.config.hidden_size, ctx.config.hidden_size), device=ctx.device, dtype=torch.float32)

    # Sort by length (desc) to minimize padding within each batch.
    chunk_input_ids.sort(key=len, reverse=True)

    bs = max(int(getattr(ctx, "batch_size", 32)), 1)
    pad_id = int(getattr(ctx, "pad_token_id", 0))

    for start in range(0, len(chunk_input_ids), bs):
        batch = chunk_input_ids[start : start + bs]
        lengths = [len(x) for x in batch]
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            continue

        input_batch = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=ctx.device)
        for i, ids in enumerate(batch):
            input_batch[i, : lengths[i]] = torch.tensor(ids, dtype=torch.long, device=ctx.device)

        with torch.no_grad():
            hidden_states_b, residual_b = embedding_attn.forward(input_batch)

        # Accumulate grams over non-pad tokens only.
        mask = (
            torch.arange(max_len, device=ctx.device).unsqueeze(0) < torch.tensor(lengths, device=ctx.device).unsqueeze(1)
        )
        tokens = hidden_states_b[mask].to(torch.float32)
        if tokens.numel() > 0:
            grams += tokens.T @ tokens

        # Store per-sample (unpadded) tensors for the next stage.
        for i, l in enumerate(lengths):
            ctx.hidden_state_tensors[domain_idx].append(hidden_states_b[i, :l].detach())
            ctx.residual_tensors[domain_idx].append(residual_b[i, :l].detach())
    return grams.to("cuda:0" if torch.cuda.is_available() else "cpu")


def forward_mlp_attn_worker(domain_idx: int, state_dict: dict, layer_idx: int) -> Tensor:
    """Top-level worker for QKV forward grams computation."""
    mlp_attn = OlmoeMLPAttention(ctx.config, state_dict, ctx.device, layer_idx)
    hidden_state_tensor = ctx.hidden_state_tensors[domain_idx]
    residual_tensor = ctx.residual_tensors[domain_idx]
    grams = torch.zeros((ctx.config.hidden_size, ctx.config.hidden_size), device=ctx.device, dtype=torch.float32)

    bs = max(int(getattr(ctx, "batch_size", 32)), 1)

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
            hidden_b[i, :l] = batch_hidden[i].to(ctx.device)
            resid_b[i, :l] = batch_resid[i].to(ctx.device)

        with torch.no_grad():
            hidden_out_b, resid_out_b = mlp_attn.forward(hidden_b, resid_b)

        mask = (
            torch.arange(max_len, device=ctx.device).unsqueeze(0) < torch.tensor(lengths, device=ctx.device).unsqueeze(1)
        )
        tokens = hidden_out_b[mask].to(torch.float32)
        if tokens.numel() > 0:
            grams += tokens.T @ tokens

        for i, l in enumerate(lengths):
            hidden_state_tensor[start + i] = hidden_out_b[i, :l].detach()
            residual_tensor[start + i] = resid_out_b[i, :l].detach()
    return grams.to("cuda:0" if torch.cuda.is_available() else "cpu")


def wudi_regmean(
    source_model_loaders: list[LazyTensorLoader],
    base_model_loader: LazyTensorLoader,
    writer: TensorWriter,
    output_path: str,
    device: str,
    dtype: torch.dtype,
    datasets: dict[str, list[str]],
    max_samples_per_domain: int,
    batch_size: int = 32,
    reduce_non_diag_a: float = 1.0,
    nonlinear_merge_method: Literal["base", "average", "sum"] = "base",
) -> dict[str, Any]:
    def regmean_merge(tensor_name: str, grams: list[Tensor]):
        gram_sum_inv = torch.stack(grams, dim=0).sum(dim=0).pinverse()
        source_tensors = [loader.get_tensor(tensor_name, device, dtype) for loader in source_model_loaders]
        merged_tensor = (
            gram_sum_inv.to(dtype)
            @ torch.stack([gram.to(dtype) @ weight.T for gram, weight in zip(grams, source_tensors)], dim=0).sum(dim=0)
        ).T.cpu()
        writer.save_tensor(tensor_name, merged_tensor)
        return merged_tensor

    def run_pool_merges(
        *,
        state_dict: dict[str, Tensor] | None,
        nonlinear_map: dict[str, str],
        wudi_map: dict[str, str],
        wudi_iter_num: int = 300,
    ) -> None:
        pending: list[tuple[ApplyResult, str, str]] = []
        for tensor_name, state_key in nonlinear_map.items():
            pending.append((pool.apply_async(nonlinear_merge_worker, (tensor_name, nonlinear_merge_method)), tensor_name, state_key))
        for tensor_name, state_key in wudi_map.items():
            pending.append((pool.apply_async(wudi_merge_worker, (tensor_name, wudi_iter_num)), tensor_name, state_key))

        for ar, tensor_name, state_key in tqdm(pending, desc="Merging tensors", leave=False):
            out_name, merged_tensor = ar.get()
            assert out_name == tensor_name
            writer.save_tensor(out_name, merged_tensor)
            if state_dict is not None:
                state_dict[state_key] = merged_tensor

    assert len(source_model_loaders) == len(datasets), "Number of source models must be the same of dataset domains."

    # Get configs
    model_path = source_model_loaders[0].index.base_path
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    max_length = model_config.max_position_embeddings
    num_hidden_layers = model_config.num_hidden_layers
    num_experts = model_config.num_experts
    n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Preprocess data
    print("Preprocessing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    domain_datasets = {}
    total_tokens = []
    for domain, dataset_paths in tqdm(datasets.items()):
        dataset = load_dataset("json", data_files=dataset_paths, split="train")
        dataset = dataset.shuffle(seed=42).select(range(min(max_samples_per_domain, len(dataset))))
        dataset = dataset.map(
            _tokenize,
            batched=False,
            remove_columns=dataset.column_names,
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        )
        domain_datasets[domain] = dataset
        total_tokens.append(sum(dataset["length"]))
    del tokenizer

    # Setup worker context and pool
    pad_token_id = getattr(model_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model_config, "eos_token_id", 0) or 0
    source_model_paths = [loader.index.base_path for loader in source_model_loaders]
    base_model_path = base_model_loader.index.base_path
    worker_ctx = WorkerContext(
        model_config,
        dtype,
        batch_size=batch_size,
        pad_token_id=int(pad_token_id),
        source_model_paths=source_model_paths,
        base_model_path=base_model_path,
    )
    gpu_queue = mp.Queue()
    for i in range(n_devices):
        gpu_queue.put(i)
    pool = mp.Pool(processes=n_devices, initializer=init_worker, initargs=(worker_ctx, gpu_queue))

    # Merge embedding and the first attention layer
    print("Merging embedding and the first attention layer...")
    state_dict = {}
    nonlinear_map = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.layers.0.input_layernorm.weight": "input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "model.layers.0.self_attn.k_norm.weight": "k_norm.weight",
        "model.layers.0.self_attn.q_norm.weight": "q_norm.weight",
    }
    wudi_map = {
        "model.layers.0.self_attn.q_proj.weight": "q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight": "k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight": "v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight": "o_proj.weight",
    }
    run_pool_merges(state_dict=state_dict, nonlinear_map=nonlinear_map, wudi_map=wudi_map)

    # Forward embedding and the first attention layer to compute grams
    print("Computing embedding and first attention layer grams...")
    grams = []
    for i, dataset in enumerate(tqdm(domain_datasets.values())):
        chunk_size = (len(dataset) + n_devices - 1) // n_devices
        chunks = (
            (i, state_dict, dataset[j * chunk_size : (j + 1) * chunk_size]["input_ids"]) for j in range(n_devices)
        )
        results = pool.starmap(forward_embedding_attn_worker, chunks)
        gram = torch.stack(results, dim=0).to(device).sum(dim=0) / total_tokens[i]
        gram = _reduce_non_diag(gram, reduce_non_diag_a)
        grams.append(gram)

    # Merge layers
    for layer_idx in range(num_hidden_layers - 1):
        state_dict = {}

        ## Merge MLP
        print(f"Merging MLP for layer {layer_idx}...")
        state_dict["gate.weight"] = regmean_merge(f"model.layers.{layer_idx}.mlp.gate.weight", grams)

        ## Merge Experts + Next Attention (pool-parallel)
        print(f"Merging experts and attention for layer {layer_idx+1}...")
        nonlinear_map = {
            f"model.layers.{layer_idx+1}.input_layernorm.weight": "input_layernorm.weight",
            f"model.layers.{layer_idx+1}.post_attention_layernorm.weight": "post_attention_layernorm.weight",
            f"model.layers.{layer_idx+1}.self_attn.k_norm.weight": "k_norm.weight",
            f"model.layers.{layer_idx+1}.self_attn.q_norm.weight": "q_norm.weight",
        }
        wudi_map = {
            f"model.layers.{layer_idx+1}.self_attn.q_proj.weight": "q_proj.weight",
            f"model.layers.{layer_idx+1}.self_attn.k_proj.weight": "k_proj.weight",
            f"model.layers.{layer_idx+1}.self_attn.v_proj.weight": "v_proj.weight",
            f"model.layers.{layer_idx+1}.self_attn.o_proj.weight": "o_proj.weight",
        }
        for expert_idx in range(num_experts):
            for module in ("gate_proj", "up_proj", "down_proj"):
                tensor_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{module}.weight"
                state_key = f"experts.{expert_idx}.{module}.weight"
                wudi_map[tensor_name] = state_key

        run_pool_merges(state_dict=state_dict, nonlinear_map=nonlinear_map, wudi_map=wudi_map)

        ## Forward MLP and Attention to compute grams
        print(f"Computing grams for layer {layer_idx+1}...")
        grams.clear()
        for i in trange(len(datasets)):
            results = pool.starmap(forward_mlp_attn_worker, ((i, state_dict, layer_idx) for j in range(n_devices)))
            gram = torch.stack(results, dim=0).to(device).sum(dim=0) / total_tokens[i]
            gram = _reduce_non_diag(gram, reduce_non_diag_a)
            grams.append(gram)

    # Merge the last MLP layer and LM head
    print("Merge the last MLP layer and LM head...")
    regmean_merge(f"model.layers.{num_hidden_layers-1}.mlp.gate.weight", grams)
    nonlinear_map = {
        "model.norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    wudi_map: dict[str, str] = {}
    for expert_idx in range(num_experts):
        for module in ("gate_proj", "up_proj", "down_proj"):
            tensor_name = f"model.layers.{num_hidden_layers-1}.mlp.experts.{expert_idx}.{module}.weight"
            wudi_map[tensor_name] = tensor_name
    run_pool_merges(state_dict=None, nonlinear_map=nonlinear_map, wudi_map=wudi_map)

    pool.close()
    pool.join()
    return {}


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


def _reduce_non_diag(cov_mat: Tensor, a: float):
    diag_weight = torch.diag(torch.ones(cov_mat.size(0)) - a).to(cov_mat.device)
    non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
    weight = diag_weight + non_diag_weight
    ret = cov_mat * weight
    return ret


class OlmoeEmbeddingAttention(nn.Module):
    def __init__(self, config: OlmoeConfig, state_dict: dict[str, Tensor], device: str):
        super().__init__()
        self.config = config
        self.device = device
        self.embedding = nn.Embedding.from_pretrained(state_dict.pop("embed_tokens.weight"), freeze=True).to(device)
        self.position_embedding = OlmoeRotaryEmbedding(config, device)
        self.input_ln = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_ln.load_state_dict({"weight": state_dict.pop("input_layernorm.weight")})
        self.input_ln.to(device)
        self.post_attention_ln = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_ln.load_state_dict({"weight": state_dict.pop("post_attention_layernorm.weight")})
        self.post_attention_ln.to(device)
        self.attention = OlmoeSdpaAttention(config, 0)
        self.attention.load_state_dict(state_dict, strict=False)
        self.attention.to(device)

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

        residual = self.embedding(input_ids)
        position_embeddings = self.position_embedding(residual, position_ids)
        hidden_states = self.input_ln(residual)
        hidden_states = self.attention(hidden_states, position_embeddings=position_embeddings)[0]
        residual = hidden_states + residual
        hidden_states = self.post_attention_ln(residual)

        if squeeze:
            return hidden_states.squeeze(0), residual.squeeze(0)
        return hidden_states, residual


class OlmoeMLPAttention(nn.Module):
    def __init__(self, config: OlmoeConfig, state_dict: dict[str, Tensor], device: str, layer_idx: int):
        super().__init__()
        self.config = config
        self.device = device
        self.layer_idx = layer_idx
        self.position_embedding = OlmoeRotaryEmbedding(config, device)
        self.mlp = OlmoeSparseMoeBlock(config)
        self.mlp.load_state_dict(state_dict, strict=False)
        self.mlp.to(device)
        self.input_ln = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_ln.load_state_dict({"weight": state_dict.pop("input_layernorm.weight")})
        self.input_ln.to(device)
        self.attention = OlmoeSdpaAttention(config, layer_idx + 1)
        self.attention.load_state_dict(state_dict, strict=False)
        self.attention.to(device)
        self.post_attention_ln = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_ln.load_state_dict({"weight": state_dict.pop("post_attention_layernorm.weight")})
        self.post_attention_ln.to(device)

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
        hidden_states = self.input_ln(residual)
        bsz, seq_len, _ = hidden_states.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(bsz, -1)
        position_embeddings = self.position_embedding(hidden_states, position_ids)
        hidden_states = self.attention(hidden_states, position_embeddings=position_embeddings)[0]
        residual = hidden_states + residual
        hidden_states = self.post_attention_ln(residual)

        if squeeze:
            return hidden_states.squeeze(0), residual.squeeze(0)
        return hidden_states, residual
