# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import json
import logging
import os
import threading
import time
from typing import Dict, Optional

import safetensors
import torch

from utils import get_logger

LOG = get_logger(__name__)


class TensorWriter:
    out_path: str
    override_basename: Optional[str]
    max_shard_size: int
    shards_written: int
    weight_map = Dict[str, str]
    current_shard: Dict[str, torch.Tensor]
    current_shard_size: int
    safe_serialization: bool
    lock: threading.Lock
    max_threads: int
    semaphore: threading.Semaphore
    threads: list

    def __init__(
        self,
        out_path: str,
        dtype: str = "float32",
        max_shard_size: int = 1000 * 1000 * 1000 * 5,
        safe_serialization: bool = True,
        override_basename: Optional[str] = None,
        max_threads: int = 16,
    ) -> None:
        os.makedirs(out_path, exist_ok=True)
        self.out_path = out_path
        self.dtype = getattr(torch, dtype)
        self.override_basename = override_basename
        self.max_shard_size = max_shard_size
        self.safe_serialization = safe_serialization
        self.shards_written = 0
        self.weight_map = {}
        self.current_shard = {}
        self.current_shard_size = 0
        self.lock = threading.Lock()
        self.max_threads = max_threads
        self.semaphore = threading.Semaphore(max_threads)
        self.threads = []

    def save_tensor(self, name: str, tensor: torch.Tensor, clone: bool = False):
        tensor = tensor.to(self.dtype)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if clone:
            tensor = tensor.clone()

        tensor_size = tensor.numel() * tensor.element_size()
        with self.lock:
            if (
                self.current_shard
                and self.max_shard_size >= 0
                and self.current_shard_size + tensor_size > self.max_shard_size
            ):
                self._flush_current_shard()

            self.current_shard[name] = tensor
            self.current_shard_size += tensor_size

    def save_tensors(self, tensors: list[tuple[str, torch.Tensor]]):
        for name, tensor in tensors:
            self.save_tensor(name, tensor)

    def _flush_current_shard(self):
        if not self.current_shard:
            return

        # Copy the current shard data to avoid sharing with thread
        shard_data = self.current_shard.copy()
        shard_size = self.current_shard_size
        shard_index = self.shards_written

        # Reset current shard
        self.current_shard = {}
        self.current_shard_size = 0
        self.shards_written += 1

        # Start thread to write shard
        self.semaphore.acquire()
        thread = threading.Thread(target=self._flush_current_shard_async, args=(shard_data, shard_index))
        thread.start()
        self.threads.append(thread)

    def finalize(self):
        with self.lock:
            self._flush_current_shard()

        # Wait for all threads to complete
        while self.threads:
            remaining = len(self.threads)
            LOG.info(f"Waiting for {remaining} threads to complete...")
            time.sleep(2)
            # Remove completed threads
            self.threads = [t for t in self.threads if t.is_alive()]

        LOG.info("Finalizing shard names")

        prefix, extension = self._get_name_components()

        # standardize shard names to hf format
        total_shards = self.shards_written
        name_remap = {}
        for idx in range(total_shards):
            name_remap[f"{prefix}-{idx + 1}.{extension}"] = f"{prefix}-{idx + 1:05d}-of-{total_shards:05d}.{extension}"

        if total_shards < 2:
            name_remap[f"{prefix}-1.{extension}"] = f"{prefix}.{extension}"

        for old_name, new_name in name_remap.items():
            os.rename(
                os.path.join(self.out_path, old_name),
                os.path.join(self.out_path, new_name),
            )

        if total_shards < 2:
            return

        for key in self.weight_map:
            self.weight_map[key] = name_remap[self.weight_map[key]]

        with open(
            os.path.join(self.out_path, f"{prefix}.{extension}.index.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump({"weight_map": self.weight_map}, file, indent=4)

    def _get_name_components(self):
        if self.override_basename:
            return self.override_basename, ("safetensors" if self.safe_serialization else "bin")
        if self.safe_serialization:
            return "model", "safetensors"
        return "pytorch_model", "bin"

    def _flush_current_shard_async(self, shard_data: Dict[str, torch.Tensor], shard_index: int):
        try:
            LOG.info(f"Writing shard #{shard_index + 1} to disk")

            prefix, extension = self._get_name_components()
            shard_name = f"{prefix}-{shard_index + 1}.{extension}"

            with self.lock:
                for key in shard_data:
                    self.weight_map[key] = shard_name

            shard_path = os.path.join(self.out_path, shard_name)
            if self.safe_serialization:
                self._save_st_async(shard_path, shard_data)
            else:
                torch.save(shard_data, shard_path)
        finally:
            self.semaphore.release()

    def _save_st_async(self, shard_path: str, shard_data: Dict[str, torch.Tensor]):
        def _do_save():
            safetensors.torch.save_file(
                shard_data,
                shard_path,
                metadata={"format": "pt"},
            )

        try:
            _do_save()
        except RuntimeError as e:
            if len(e.args) > 0 and isinstance(e.args[0], str) and "share memory" in e.args[0]:
                LOG.warning("Your model has duplicated tensors but the --clone-tensors " "flag is not set.")
                shard_data = {key: shard_data[key].clone() for key in shard_data}
                _do_save()
            else:
                raise
