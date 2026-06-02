import os
import sys
import json
import logging
from typing import Generator


# Configure logging globally
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s|%(asctime)s] %(filename)s:%(lineno)s >> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance with the global configuration.

    Args:
        name: The name of the logger. If None, returns the root logger.

    Returns:
        A logger instance with the global configuration.
    """
    return logging.getLogger(name)


def load_json(path: str, default: list | dict | None = None) -> list | dict:
    if os.path.exists(path) or default is None:
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return default


def save_json(data:  list | dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_jsonl(path: str, default: list[dict] | None = None) -> list[dict]:
    if os.path.exists(path) or default is None:
        with open(path, 'r') as f:
            return [json.loads(line) for line in f]
    else:
        return default


def read_jsonl(file_path: str, default: list[dict] | None = None) -> Generator[dict, None, None]:
    """Read a JSONL file line by line.

    Args:
        file_path: The file path.
        default: The default value if the file does not exist.

    Returns:
        The generator of loaded JSONL data.
    """
    if default is not None and not os.path.exists(file_path):
        yield from default
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)  # type: ignore


def save_jsonl(data: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def copy_file(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, 'rb') as fsrc:
        with open(dst, 'wb') as fdst:
            fdst.write(fsrc.read())


def copy_files(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        if os.path.isfile(src_path):
            copy_file(src_path, dst_path)


def mean(numbers: list[float | int]) -> float:
    return sum(numbers) / len(numbers)
