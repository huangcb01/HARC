import re
import torch
from typing import Literal
from torch import Tensor


def get_sign_consensus_mask(
    delta: Tensor,
    method: Literal["sum", "count"] = "sum",
    int8_mask: bool = False,
):
    """Returns a mask determining which delta vectors should be merged into the final model.

    Parameters:
        method: For the methodology described in the TIES paper use 'sum'. For a simpler naive count of signs, use 'count'.
        int8_mask: Use int8 for mask type or not.
    """
    mask_dtype = torch.int8 if int8_mask else delta.dtype


    sign = delta.sign().to(mask_dtype)

    if method == "sum":
        sign_weight = delta.sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return sign == majority_sign

def group_by_layer(names: list[str]) -> list[list[str]]:
    layer_dict = {}
    other_names = []

    for name in names:
        match = re.search(r"model\.layers\.(\d+)\.", name)
        if match:
            layer_num = match.group(1)
            if layer_num not in layer_dict:
                layer_dict[layer_num] = []
            layer_dict[layer_num].append(name)
        else:
            other_names.append(name)

    sorted_layers = []
    for layer_num in sorted(layer_dict.keys(), key=int):
        sorted_layers.append(layer_dict[layer_num])

    for name in other_names:
        sorted_layers.append([name])

    return sorted_layers


def group_by_expert(names: list[str]) -> list[list[str]]:
    expert_dict = {}
    other_names = []

    for name in names:
        match = re.search(r"experts\.(\d+)\.", name)
        if match:
            expert_num = match.group(1)
            if expert_num not in expert_dict:
                expert_dict[expert_num] = []
            expert_dict[expert_num].append(name)
        else:
            other_names.append(name)

    sorted_experts = []
    for expert_num in sorted(expert_dict.keys(), key=int):
        sorted_experts.append(expert_dict[expert_num])

    for name in other_names:
        sorted_experts.append([name])

    return sorted_experts


def group_by_expert_sublayer(names: list[str]) -> list[list[str]]:
    sublayer_dict = {}

    for name in names:
        sublayer = ".".join(name.split(".")[2:4]) if "mlp" in name else name
        if sublayer not in sublayer_dict:
            sublayer_dict[sublayer] = []
        sublayer_dict[sublayer].append(name)

    sorted_sublayers = []
    for sublayer in sorted(sublayer_dict.keys()):
        sorted_sublayers.append(sublayer_dict[sublayer])

    return sorted_sublayers


def group_by_sublayer(names: list[str]) -> list[list[str]]:
    sublayer_dict = {}
    other_names = []

    for name in names:
        if name.startswith("model.layers."):
            sublayer = ".".join(name.split(".")[2:4])
            if sublayer not in sublayer_dict:
                sublayer_dict[sublayer] = []
            sublayer_dict[sublayer].append(name)
        else:
            other_names.append(name)

    sorted_sublayers = []
    for sublayer in sorted(sublayer_dict.keys()):
        sorted_sublayers.append(sublayer_dict[sublayer])
    for name in other_names:
        sorted_sublayers.append([name])

    return sorted_sublayers


def get_layer_num(names: list[str]) -> int:
    layer_num = 0
    for name in names:
        match = re.search(r"model\.layers\.(\d+)\.", name)
        if match:
            layer_num = max(layer_num, int(match.group(1)))
    return layer_num + 1


def get_expert_num(names: list[str]) -> int:
    expert_num = 0
    for name in names:
        match = re.search(r"experts\.(\d+)\.", name)
        if match:
            expert_num = max(expert_num, int(match.group(1)))
    return expert_num + 1
