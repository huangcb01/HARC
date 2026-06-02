import os
import sys
import shutil
import fire
import torch
import multiprocessing as mp
from typing import Literal, cast
from functools import partial

import merge_method
from file import LazyTensorLoader, ShardedTensorIndex, TensorWriter
from utils import load_json, save_json, get_logger

logger = get_logger()


FILES_TO_COPY = [
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
]


def merge(
    method: Literal["average", "task_arithmetic", "ties", "dare_ties", "sce", "wudi", "expert", "more_experts"],
    output_path: str,
    source_models: list[str],
    base_model: str | None = None,
    device: str = "cpu",
    work_dtype: str | None = None,
    target_dtype: str = "float32",
    **kwargs,
):
    if work_dtype is None:
        work_dtype = target_dtype

    # Get model loader/writer
    source_model_loaders = [LazyTensorLoader(ShardedTensorIndex.from_disk(path)) for path in source_models]
    base_model_loader = LazyTensorLoader(ShardedTensorIndex.from_disk(base_model)) if base_model else None
    writer = TensorWriter(output_path, target_dtype)

    # Merge
    if method in {"task_arithmetic", "ties", "dare_ties", "sce"}:
        assert base_model_loader is not None, "A base model is needed for task arithmetic merging methods."
        merge_fn = partial(merge_method.gta, method=method)
    else:
        merge_fn = getattr(merge_method, method)
    new_config = merge_fn(
        source_model_loaders, base_model_loader, writer, output_path, device, getattr(torch, work_dtype), **kwargs
    )
    writer.finalize()

    # Copy files
    for file in FILES_TO_COPY:
        source_path = os.path.join(source_models[0], file)
        if os.path.exists(source_path):
            target_path = os.path.join(output_path, file)
            shutil.copy(source_path, target_path)
        else:
            logger.warning(f"File {file} does not exist!")

    # Update config.json
    if new_config:
        config = cast(dict, load_json(os.path.join(output_path, "config.json")))
        config.update(new_config)
        save_json(config, os.path.join(output_path, "config.json"))

if __name__ == "__main__":
    mp.set_start_method("spawn")
    fire.Fire(merge)
