import os
import sys
import torch
import numpy as np
import multiprocessing as mp
from torch import nn, Tensor
from typing import Any, cast
from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from transformers.models.olmoe.modeling_olmoe import OlmoeRotaryEmbedding, OlmoeSparseMoeBlock, OlmoeRMSNorm
from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
from itertools import batched
from tqdm import tqdm, trange

from file import LazyTensorLoader, TensorWriter
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class WorkerContext:
    residual_tensors: list[Tensor]
    hidden_state_tensors: list[Tensor]
    config: OlmoeConfig
    token_indices: list[Tensor]
    seq_lengths: list[Tensor]
    max_length: int
    batch_size: int
    dtype: torch.dtype
    gpu_id: int = -1


def init_worker(worker_ctx: WorkerContext, gpu_queue: mp.Queue):
    global ctx
    ctx = worker_ctx
    gpu_id = gpu_queue.get()
    ctx.gpu_id = gpu_id


def forward_embedding_worker(
    domain_idx: int,
    rank: int,
    chunk_size: int,
    embedding_tensor: torch.Tensor,
    input_ln_tensor: torch.Tensor,
    chunk_input_ids: list[list[int]],
) -> Tensor:
    device = f"cuda:{ctx.gpu_id}" if torch.cuda.is_available() else "cpu"
    embedding_layer = nn.Embedding.from_pretrained(embedding_tensor, freeze=True).to(device=device)
    input_ln = OlmoeRMSNorm(ctx.config.hidden_size, eps=ctx.config.rms_norm_eps)
    input_ln.load_state_dict({"weight": input_ln_tensor})
    input_ln.to(device)
    grams = torch.zeros((ctx.config.hidden_size, ctx.config.hidden_size), device=device, dtype=torch.float32)
    sample_idx = rank * chunk_size
    for batch in batched(chunk_input_ids, ctx.batch_size * 128):
        input_ids = torch.tensor([ids for ids_list in batch for ids in ids_list], device=device)
        batch_tokens = len(input_ids)
        start_index = ctx.token_indices[domain_idx][sample_idx]
        with torch.no_grad():
            hidden_states = embedding_layer(input_ids)
            ctx.residual_tensors[domain_idx][start_index : start_index + batch_tokens, :] = hidden_states
            hidden_states = input_ln(hidden_states)
            ctx.hidden_state_tensors[domain_idx][start_index : start_index + batch_tokens, :] = hidden_states
            grams += hidden_states.T @ hidden_states
        sample_idx += len(batch)
    return grams.to("cuda:0" if torch.cuda.is_available() else "cpu")


def forward_qkv_worker(domain_idx: int, rank: int, chunk_size: int, state_dict: dict) -> Tensor:
    """Top-level worker for QKV forward grams computation."""
    device = f"cuda:{ctx.gpu_id}" if torch.cuda.is_available() else "cpu"
    attention_layer = OlmoeAttentionQKV(ctx.config)
    attention_layer.load_state_dict(state_dict)
    attention_layer.to(device, ctx.dtype)
    position_embedding_layer = OlmoeRotaryEmbedding(ctx.config, device=device)
    grams = torch.zeros((ctx.config.hidden_size, ctx.config.hidden_size), device=device, dtype=torch.float32)
    start_sample_idx = rank * chunk_size
    end_sample_idx = min((rank + 1) * chunk_size, len(ctx.seq_lengths[domain_idx]))
    for idx in range(start_sample_idx, end_sample_idx, 1):
        attention_mask = [1] * ctx.seq_lengths[domain_idx][idx]
        position_ids = list(range(ctx.seq_lengths[domain_idx][idx]))
        start_token_idx = ctx.token_indices[domain_idx][idx]
        end_token_idx = start_token_idx + len(attention_mask)
        attention_mask_t = torch.tensor(attention_mask, device=device).unsqueeze(0)
        attention_mask_t = _prepare_4d_attention_mask(attention_mask_t, ctx.dtype).to(device)
        position_ids_t = torch.tensor(position_ids, device=device).unsqueeze(0)
        hidden_states = ctx.hidden_state_tensors[domain_idx][start_token_idx:end_token_idx, :].to(device).unsqueeze(0)
        with torch.no_grad():
            position_embeddings = position_embedding_layer(hidden_states, position_ids_t)
            hidden_states = attention_layer.forward(hidden_states, attention_mask_t, position_embeddings).squeeze(0)
            grams += hidden_states.T @ hidden_states
            ctx.hidden_state_tensors[domain_idx][start_token_idx:end_token_idx, :] = hidden_states
    return grams.to("cuda:0" if torch.cuda.is_available() else "cpu")


def forward_o_worker(domain_idx: int, rank: int, chunk_size: int, state_dict: dict) -> Tensor:
    device = f"cuda:{ctx.gpu_id}" if torch.cuda.is_available() else "cpu"
    attention_layer = OlmoeAttentionO(ctx.config)
    attention_layer.load_state_dict(state_dict)
    attention_layer.to(device, ctx.dtype)
    grams = torch.zeros((ctx.config.hidden_size, ctx.config.hidden_size), device=device, dtype=torch.float32)
    token_indices = ctx.token_indices[domain_idx]
    hidden_state_tensor = ctx.hidden_state_tensors[domain_idx]
    residual_tensor = ctx.residual_tensors[domain_idx]
    start_sample_idx = rank * chunk_size
    end_sample_idx = min((rank + 1) * chunk_size, len(ctx.seq_lengths[domain_idx]))
    start_token_idx = token_indices[start_sample_idx]
    end_token_idx = token_indices[end_sample_idx]
    batch_length = ctx.max_length * ctx.batch_size
    for idx in range(start_token_idx, end_token_idx, batch_length):
        hidden_states = hidden_state_tensor[idx : idx + batch_length, :].to(device)
        residual = residual_tensor[idx : idx + batch_length, :].to(device)
        with torch.no_grad():
            hidden_states, residual = attention_layer.forward(hidden_states, residual)
            grams += hidden_states.T @ hidden_states
            hidden_state_tensor[idx : idx + batch_length, :] = hidden_states
            residual_tensor[idx : idx + batch_length, :] = residual
    return grams.to("cuda:0" if torch.cuda.is_available() else "cpu")


def merge_expert_down(state_dict: dict[str, Tensor], source_tensors: list[Tensor]):
    device = f"cuda:{ctx.gpu_id}" if torch.cuda.is_available() else "cpu"
    mlp_layer = OlmoeMLPUpAndGate(ctx.config)
    mlp_layer.load_state_dict(state_dict)
    mlp_layer.to(device, ctx.dtype)
    grams = []
    for domain_idx in range(len(ctx.residual_tensors)):
        gram = torch.zeros(
            (ctx.config.intermediate_size, ctx.config.intermediate_size), device=device, dtype=torch.float32
        )
        total_tokens = ctx.token_indices[domain_idx][-1]
        hidden_state_tensor = ctx.hidden_state_tensors[domain_idx]
        batch_length = ctx.max_length * ctx.batch_size
        for idx in range(0, total_tokens, batch_length):
            hidden_states = hidden_state_tensor[idx : idx + batch_length, :].to(device)
            with torch.no_grad():
                hidden_states = mlp_layer.forward(hidden_states)
                gram += hidden_states.T @ hidden_states
        grams.append(gram / total_tokens)
    gram_sum_inv = torch.stack(grams, dim=0).sum(dim=0).pinverse()
    merged_tensor = (
        gram_sum_inv.to(ctx.dtype)
        @ torch.stack([gram.to(ctx.dtype) @ weight.to(device).T for gram, weight in zip(grams, source_tensors)], dim=0).sum(dim=0)
    ).T.cpu()
    return merged_tensor


def forward_mlp_full_worker(
    domain_idx: int, rank: int, chunk_size: int, state_dict: dict, next_input_ln_weight: Tensor | None
) -> Tensor:
    device = f"cuda:{ctx.gpu_id}" if torch.cuda.is_available() else "cpu"
    mlp_layer = OlmoeSparseMoeBlock(ctx.config)
    mlp_layer.load_state_dict(state_dict)
    mlp_layer.to(device, ctx.dtype)
    input_ln = OlmoeRMSNorm(ctx.config.hidden_size, eps=ctx.config.rms_norm_eps)
    input_ln.load_state_dict({"weight": next_input_ln_weight})
    input_ln.to(device, ctx.dtype)
    grams = torch.zeros((ctx.config.hidden_size, ctx.config.hidden_size), device=device, dtype=torch.float32)
    token_indices = ctx.token_indices[domain_idx]
    hidden_state_tensor = ctx.hidden_state_tensors[domain_idx]
    residual_tensor = ctx.residual_tensors[domain_idx]
    start_sample_idx = rank * chunk_size
    end_sample_idx = min((rank + 1) * chunk_size, len(ctx.seq_lengths[domain_idx]))
    start_token_idx = token_indices[start_sample_idx]
    end_token_idx = token_indices[end_sample_idx]
    batch_length = ctx.max_length * ctx.batch_size
    for idx in range(start_token_idx, end_token_idx, batch_length):
        hidden_states = hidden_state_tensor[idx : idx + batch_length, :].to(device)
        residual = residual_tensor[idx : idx + batch_length, :].to(device)
        with torch.no_grad():
            hidden_states = mlp_layer.forward(hidden_states.unsqueeze(0))[0].squeeze(0)
            residual = hidden_states + residual
            hidden_states = input_ln(residual)
            grams += hidden_states.T @ hidden_states
            residual_tensor[idx : idx + batch_length, :] = residual
            hidden_state_tensor[idx : idx + batch_length, :] = hidden_states
    return grams.to("cuda:0" if torch.cuda.is_available() else "cpu")


def com(
    source_model_loaders: list[LazyTensorLoader],
    base_model_loader: LazyTensorLoader,
    writer: TensorWriter,
    output_path: str,
    device: str,
    dtype: torch.dtype,
    com: bool,
    datasets: dict[str, list[str]],
    max_samples_per_domain: int,
    cache_dir: str,
    batch_size: int = 32,
    reduce_non_diag_a: float = 1.0,
) -> dict[str, Any]:
    def average_merge(tensor_name: str):
        source_tensors = [loader.get_tensor(tensor_name, device, dtype) for loader in source_model_loaders]
        merged_tensor = torch.stack(source_tensors, dim=0).mean(dim=0).cpu()
        writer.save_tensor(tensor_name, merged_tensor)
        return source_tensors, merged_tensor

    def regmean_merge(tensor_name: str, grams: list[Tensor]):
        gram_sum_inv = torch.stack(grams, dim=0).sum(dim=0).pinverse()
        source_tensors = [loader.get_tensor(tensor_name, device, dtype) for loader in source_model_loaders]
        merged_tensor = (
            gram_sum_inv.to(dtype)
            @ torch.stack([gram.to(dtype) @ weight.T for gram, weight in zip(grams, source_tensors)], dim=0).sum(dim=0)
        ).T.cpu()
        writer.save_tensor(tensor_name, merged_tensor)
        return source_tensors, merged_tensor

    assert len(source_model_loaders) == len(datasets), "Number of source models must be the same of dataset domains."
    os.makedirs(cache_dir, exist_ok=True)

    # Get configs
    model_path = source_model_loaders[0].index.base_path
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    max_length = model_config.max_position_embeddings
    hidden_size = model_config.hidden_size
    num_hidden_layers = model_config.num_hidden_layers
    num_experts = model_config.num_experts
    n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Preprocess data
    logger.info("Preprocessing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    domain_datasets = {}
    seq_lengths, token_indices = [], []
    residual_tensors, hidden_state_tensors = [], []
    for domain, dataset_paths in tqdm(datasets.items()):
        dataset = load_dataset("json", data_files=dataset_paths, split="train")
        dataset = dataset.shuffle(seed=42).select(range(min(max_samples_per_domain, len(dataset))))
        dataset = dataset.map(
            _tokenize,
            batched=False,
            remove_columns=dataset.column_names,
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        )
        lengths = torch.tensor(dataset["length"]).share_memory_()
        seq_lengths.append(lengths)
        token_indices.append(torch.cumsum(torch.cat((torch.tensor([0]), lengths)), dim=0).share_memory_())
        domain_datasets[domain] = dataset
        total_tokens = token_indices[-1][-1].item()
        residual_tensors.append(torch.zeros((total_tokens, hidden_size), dtype=dtype).share_memory_())
        hidden_state_tensors.append(torch.zeros((total_tokens, hidden_size), dtype=dtype).share_memory_())
    del tokenizer

    # Setup worker context and pool
    worker_ctx = WorkerContext(
        residual_tensors, hidden_state_tensors, model_config, token_indices, seq_lengths, max_length, batch_size, dtype
    )
    gpu_queue = mp.Queue()
    for i in range(n_devices):
        gpu_queue.put(i)
    pool = mp.Pool(processes=n_devices, initializer=init_worker, initargs=(worker_ctx, gpu_queue))

    # Merge embedding layers and input LN of the first layer
    logger.info("Merging embedding layer...")
    source_embeddings, merged_embedding = average_merge("model.embed_tokens.weight")
    source_input_ln, merged_input_ln = average_merge("model.layers.0.input_layernorm.weight")

    # Forward embedding layer and compute grams
    print("Computing embedding grams...")
    grams = []
    for i, dataset in enumerate(tqdm(domain_datasets.values())):
        embedding_tensor = merged_embedding if com else source_embeddings[i].cpu()
        input_ln_tensor = merged_input_ln if com else source_input_ln[i].cpu()
        chunk_size = (len(dataset) + n_devices - 1) // n_devices
        chunks = [
            (
                i,
                j,
                chunk_size,
                embedding_tensor,
                input_ln_tensor,
                dataset[j * chunk_size : (j + 1) * chunk_size]["input_ids"],
            )
            for j in range(n_devices)
        ]
        results = pool.starmap(forward_embedding_worker, chunks)
        gram = torch.stack(results, dim=0).to(device).sum(dim=0) / token_indices[i][-1]
        gram = _reduce_non_diag(gram, reduce_non_diag_a)
        grams.append(gram)
    del embedding_tensor, chunks, results

    # Merge layers
    for layer_idx in range(num_hidden_layers):
        ## Merge Q/K/V layers
        print(f"Merging Q/K/V for layer {layer_idx}...")
        state_dict = {}
        ### Merge linear layers using regmean
        for module in ("q_proj", "k_proj", "v_proj"):
            source_weights, merged_weight = regmean_merge(f"model.layers.{layer_idx}.self_attn.{module}.weight", grams)
            state_dict[f"{module}.weight"] = merged_weight
        ### Merge layer norms using mean
        for module in ("q_norm", "k_norm"):
            source_weights, merged_weight = average_merge(f"model.layers.{layer_idx}.self_attn.{module}.weight")
            state_dict[f"{module}.weight"] = merged_weight

        ## Forward QKV layers and compute grams
        print(f"Computing QKV grams for layer {layer_idx}...")
        grams.clear()
        for i in trange(len(datasets)):
            chunk_size = (len(token_indices[i]) + n_devices - 1) // n_devices
            if not com:
                state_dict = {
                    f"{module}.weight": source_model_loaders[i].get_tensor(
                        f"model.layers.{layer_idx}.self_attn.{module}.weight", "cpu", dtype
                    )
                    for module in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm")
                }
                state_dict["input_layer_norm.weight"] = source_model_loaders[i].get_tensor(
                    f"model.layers.{layer_idx}.input_layer_norm.weight", "cpu", dtype
                )

            results = pool.starmap(forward_qkv_worker, [(i, j, chunk_size, state_dict) for j in range(n_devices)])
            gram = torch.stack(results, dim=0).to(device).sum(dim=0) / token_indices[i][-1]
            gram = _reduce_non_diag(gram, reduce_non_diag_a)
            grams.append(gram)

        ## Merge O layer
        print(f"Merging O for layer {layer_idx}...")
        source_weights, merged_weight = regmean_merge(f"model.layers.{layer_idx}.self_attn.o_proj.weight", grams)
        state_dict = {"o_proj.weight": merged_weight}
        source_weights, merged_weight = average_merge(f"model.layers.{layer_idx}.post_attention_layernorm.weight")
        state_dict["post_attention_layernorm.weight"] = merged_weight

        ## Forward O layer to update hidden states
        print(f"Computing O grams for layer {layer_idx}...")
        grams.clear()
        for i in trange(len(datasets)):
            chunk_size = (len(token_indices[i]) + n_devices - 1) // n_devices
            if not com:
                state_dict = {
                    "o_proj.weight": source_model_loaders[i].get_tensor(
                        f"model.layers.{layer_idx}.self_attn.o_proj.weight", "cpu", dtype
                    ),
                    "post_attention_layernorm.weight": source_model_loaders[i].get_tensor(
                        f"model.layers.{layer_idx}.post_attention_layernorm.weight", "cpu", dtype
                    ),
                }

            results = pool.starmap(forward_o_worker, [(i, j, chunk_size, state_dict) for j in range(n_devices)])
            gram = torch.stack(results, dim=0).to(device).sum(dim=0) / token_indices[i][-1]
            gram = _reduce_non_diag(gram, reduce_non_diag_a)
            grams.append(gram)

        ## Merge MLP
        print(f"Merging MLP for layer {layer_idx}...")
        mlp_state_dict = {}
        for module in ("gate",) + tuple(
            f"experts.{i}.{module}" for i in range(num_experts) for module in ("gate_proj", "up_proj")
        ):
            source_weights, merged_weight = regmean_merge(f"model.layers.{layer_idx}.mlp.{module}.weight", grams)
            mlp_state_dict[f"{module}.weight"] = merged_weight
        results = []
        pbar = tqdm(total=num_experts)
        for expert_idx in range(num_experts):
            if com:
                state_dict = {
                    f"{module}.weight": mlp_state_dict[f"experts.{expert_idx}.{module}.weight"]
                    for module in ("gate_proj", "up_proj")
                }
            else:
                state_dict = {
                    f"{module}.weight": source_model_loaders[i].get_tensor(
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{module}.weight", "cpu", dtype
                    )
                    for module in ("gate_proj", "up_proj")
                }
            source_tensors = [source_model_loaders[j].get_tensor(
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", "cpu", dtype
            ) for j in range(len(datasets))]
            results.append(pool.apply_async(merge_expert_down, args=(state_dict, source_tensors), callback=lambda _: pbar.update()))
        for expert_idx, result in enumerate(results):
            merged_down = result.get()
            mlp_state_dict[f"experts.{expert_idx}.down_proj.weight"] = merged_down
            writer.save_tensor(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", merged_down)
        pbar.close()

        ## Merge input LN of the next layer
        if layer_idx + 1 < num_hidden_layers:
            print(f"Merging input LN for layer {layer_idx + 1}...")
            source_weights, merged_weight = average_merge(f"model.layers.{layer_idx + 1}.input_layernorm.weight")
        else:
            print("Merging output LN...")
            source_weights, merged_weight = average_merge(f"model.norm.weight")

        ## Forward MLP and input LN of the next layer
        print(f"Computing MLP grams for layer {layer_idx}...")
        grams.clear()
        for i in trange(len(datasets)):
            chunk_size = (len(token_indices[i]) + n_devices - 1) // n_devices
            if not com:
                mlp_state_dict = {
                    f"experts.{expert_idx}.{module}.weight": source_model_loaders[i].get_tensor(
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{module}.weight", "cpu", dtype
                    )
                    for expert_idx in range(num_experts)
                    for module in ("gate", "up_proj", "down_proj")
                }
                mlp_state_dict["gate.weight"] = source_model_loaders[i].get_tensor(
                    f"model.layers.{layer_idx}.mlp.gate.weight", "cpu", dtype
                )

            next_input_ln_weight = merged_weight if com else source_weights[i].cpu()
            results = pool.starmap(
                forward_mlp_full_worker,
                [(i, j, chunk_size, mlp_state_dict, next_input_ln_weight) for j in range(n_devices)],
            )
            gram = torch.stack(results, dim=0).to(device).sum(dim=0) / token_indices[i][-1]
            gram = _reduce_non_diag(gram, reduce_non_diag_a)
            grams.append(gram)

    # Merge LM head
    print("Merging LM head...")
    source_weights, merged_weight = regmean_merge("lm_head.weight", grams)
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


def _prepare_4d_attention_mask(attention_mask_with_indices: Tensor, dtype: torch.dtype) -> Tensor:
    r"""Expand 2d attention mask to 4d attention mask.

    Expand the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    handle packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    e.g.
    ```python
    # input
    [[1, 1, 2, 2, 2, 0]]
    # output
    [
        [
            [
                [o, x, x, x, x, x],
                [o, o, x, x, x, x],
                [x, x, o, x, x, x],
                [x, x, o, o, x, x],
                [x, x, o, o, o, x],
                [x, x, x, x, x, x],
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    _, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    zero_tensor = torch.tensor(0, dtype=dtype, device=attention_mask_with_indices.device)

    # Create a non-padding mask.
    non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    # Create indices for comparison.
    indices = attention_mask_with_indices.unsqueeze(1).unsqueeze(2)  # [bsz, 1, 1, seq_len]
    indices_t = attention_mask_with_indices.unsqueeze(1).unsqueeze(3)  # [bsz, 1, seq_len, 1]
    # Create a lower triangular mask.
    tril_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=attention_mask_with_indices.device))
    attention_mask_4d = (indices == indices_t) & non_padding_mask & tril_mask
    # Invert the attention mask.
    attention_mask_4d = torch.where(attention_mask_4d, zero_tensor, min_dtype)
    return attention_mask_4d


def _reduce_non_diag(cov_mat: Tensor, a: float):
    diag_weight = torch.diag(torch.ones(cov_mat.size(0)) - a).to(cov_mat.device)
    non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
    weight = diag_weight + non_diag_weight
    ret = cov_mat * weight
    return ret


class OlmoeAttentionQKV(nn.Module):
    def __init__(self, config: OlmoeConfig):
        super().__init__()
        self.config = config

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.q_norm = OlmoeRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.k_norm = OlmoeRMSNorm(
            (self.hidden_size // self.num_heads) * self.num_key_value_heads, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        # if attention_mask is not None and cache_position is not None:
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        return attn_output


class OlmoeAttentionO(nn.Module):
    def __init__(self, config: OlmoeConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.post_attention_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: Tensor, residual: Tensor):
        attn_output = self.o_proj(hidden_states)
        residual = attn_output + residual
        hidden_states = self.post_attention_layernorm(residual)
        return hidden_states, residual


class OlmoeMLPUpAndGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.act_fn(self.gate_proj(x)) * self.up_proj(x)
