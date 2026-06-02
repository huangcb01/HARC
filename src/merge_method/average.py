import torch
from tqdm import tqdm

from file import LazyTensorLoader, TensorWriter


def average(
    source_model_loaders: list[LazyTensorLoader],
    base_model_loader: LazyTensorLoader,
    writer: TensorWriter,
    output_path: str,
    device: str,
    dtype: torch.dtype,
):
    tensor_names = list(source_model_loaders[0].index.tensor_paths.keys())
    for tensor_name in tqdm(tensor_names):
        source_tensors = [loader.get_tensor(tensor_name, device, dtype) for loader in source_model_loaders]
        stacked = torch.stack(source_tensors)
        new_tensor = stacked.mean(dim=0)
        writer.save_tensor(tensor_name, new_tensor)
    return {}