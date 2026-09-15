import os
import sys
import shutil
import re
import time
import multiprocessing
from typing import Optional, cast, Union
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import load_jsonl, save_jsonl, load_json, save_json, get_logger

logger = get_logger(__name__)


def infer(
    model_name_or_path: str,
    dataset_dir: str,
    datasets: Union[list[str], str],
    output_dir: str,
    vllm_config: dict = {},
    temperature: float = 0,
    top_p: float = 1,
    top_k: int = -1,
    max_new_tokens: int = 4096,
    seed: Optional[int] = None,
):
    r"""Perform batch generation using vLLM engine and save results to disk."""
    import torch
    from vllm import LLM, SamplingParams

    os.makedirs(output_dir, exist_ok=True)
    if isinstance(datasets, str):
        datasets = [datasets]
    engine_args = {
        "model": model_name_or_path,
        "trust_remote_code": True,
        "tensor_parallel_size": torch.cuda.device_count(),
        "gpu_memory_utilization": 0.96,
        "dtype": "bfloat16",
    }
    if isinstance(vllm_config, dict):
        engine_args.update(vllm_config)
    else:
        raise ValueError("vllm_config should be a dict.")
    llm = LLM(**engine_args)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        seed=seed,
    )

    for dataset in datasets:
        logger.info(f"Generating responses for dataset: {dataset}")
        dataset_path = os.path.join(dataset_dir, f"{dataset}.jsonl")
        output_path = os.path.join(output_dir, f"{dataset}.jsonl")
        logger.info(f"Loading dataset from {dataset_path}...")
        data = load_jsonl(dataset_path)
        logger.info(f"Loaded {len(data)} samples.")
        all_messages = [[item["messages"][0]] for item in data]
        logger.info(f"Total {len(all_messages)} samples to generate.")
        responses = llm.chat(all_messages, sampling_params)

        for item, response in zip(data, responses):
            item["messages"] = [
                item["messages"][0],
                {"role": "assistant", "content": response.outputs[0].text},
            ]

        save_jsonl(data, output_path)
        logger.info(f"{len(data)} generated results have been saved at {output_path}.")


def validate(
    model_name_or_path: str,
    datasets: dict,  # dataset_name -> domain
    output_dir: str,
    reward_model: Optional[str] = None,
):
    VALIDATE_DOMAINS = {
        "math": _validate_math,
        "code": _validate_code,
        "if": _validate_if,
    }
    os.makedirs(output_dir, exist_ok=True)
    metrics = defaultdict(dict)
    for dataset_name, domain in datasets.items():
        logger.info(f"Validating dataset: {dataset_name}, Domain: {domain}")
        result_path = os.path.join(output_dir, f"{dataset_name}.jsonl")
        result = load_jsonl(result_path)
        result_metrics = VALIDATE_DOMAINS[domain](result, reward_model, dataset_name)

        # Separate correct and incorrect samples
        data = result_metrics.pop("data")
        correct = [item for item in data if item["correct"]]
        incorrect = [item for item in data if not item["correct"]]

        # Save results
        save_jsonl(data, os.path.join(output_dir, f"{dataset_name}.jsonl"))
        save_jsonl(correct, os.path.join(output_dir, f"{dataset_name}_correct.jsonl"))
        save_jsonl(incorrect, os.path.join(output_dir, f"{dataset_name}_incorrect.jsonl"))

        metrics[dataset_name] = result_metrics
    metrics = dict(metrics)
    metric_path = os.path.join(output_dir, "metrics.json")
    existing_metrics = cast(dict, load_json(metric_path, {}))
    if model_name_or_path not in existing_metrics:
        existing_metrics[model_name_or_path] = {}
    existing_metrics[model_name_or_path].update(metrics)
    average_score = sum(v["score"] for v in metrics.values()) / len(metrics) if metrics else 0
    existing_metrics[model_name_or_path]["average_score"] = average_score
    save_json(existing_metrics, metric_path)


def full(
    model_name_or_path: str,
    dataset_dir: str,
    datasets: dict,  # dataset_name -> domain
    output_dir: str,
    vllm_config: dict = {},
    temperature: float = 0,
    top_p: float = 1,
    top_k: int = -1,
    max_new_tokens: int = 4096,
    seed: Optional[int] = None,
    reward_model: Optional[str] = None,
):
    """Execute infer and validate in separate subprocesses for isolation."""
    import multiprocessing

    logger.info("Starting inference subprocess...")
    infer_process = multiprocessing.Process(
        target=infer,
        args=(
            model_name_or_path,
            dataset_dir,
            list(datasets.keys()),
            output_dir,
            vllm_config,
            temperature,
            top_p,
            top_k,
            max_new_tokens,
            seed,
        ),
    )
    infer_process.start()
    infer_process.join()
    logger.info("Inference completed successfully")

    logger.info("Starting validation subprocess...")
    time.sleep(5)  # Avoid reading incomplete files
    validate_process = multiprocessing.Process(
        target=validate,
        args=(model_name_or_path, datasets, output_dir, reward_model),
    )
    validate_process.start()
    validate_process.join()
    logger.info("Validation completed successfully")


def _validate_math_worker(idx: int, content: str, answer: str) -> tuple[int, dict]:
    """Worker function to validate a single math problem."""
    from math_verify import parse, verify

    prediction = parse(content) or ["", ""]
    answer_parsed = parse(f"${answer}$")
    correct = verify(answer_parsed, prediction)
    return idx, {"prediction": str(prediction[-1]), "correct": correct}


def _validate_math(
    data: list[dict], reward_model: Optional[str] = None, dataset_name: str = None
):
    executor = ProcessPoolExecutor(max_workers=min(os.cpu_count() or 2, 64) // 2)
    futures = []
    for idx, item in enumerate(data):
        future = executor.submit(_validate_math_worker, idx, item["messages"][1]["content"], item["answer"])
        futures.append(future)

    for future in tqdm(as_completed(futures), total=len(futures), desc="Validating math"):
        idx, result = future.result()
        data[idx].update(result)
    executor.shutdown()

    # Calculate metrics
    correct_count = sum(item["correct"] for item in data)
    accuracy = correct_count / len(data)
    logger.info(f"Dataset: {dataset_name}, Domain: math, Accuracy: {accuracy:.2%} ({correct_count}/{len(data)})")

    return {
        "score": accuracy,
        "total": len(data),
        "correct": correct_count,
        "data": data,
    }


def _unsafe_execute(prediction: str, test_case: str, timeout: float, result):
    from human_eval.execution import create_tempdir, reliability_guard, swallow_io, time_limit, TimeoutException

    # 在进程开始时重定向 stdout 和 stderr，防止输出到终端
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    with create_tempdir():
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        unlink = os.unlink

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()

        # Construct the check program and run it.
        check_program = f"{prediction}\n{test_case}"
        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(check_program, exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
        os.unlink = unlink


def _validate_code_worker(idx: int, response: str, test: str, timeout: float) -> tuple[int, dict]:
    """Copied from human_eval.evaluation, modified to run test cases separately."""
    # Extract program
    blocks = re.findall(r"```\w*\n(.*?)```", response, re.DOTALL)
    if len(blocks) >= 1:
        prediction = blocks[0]
    else:
        prediction = response

    # Split test cases
    context = ""
    test_cases = []
    for line in test.split("\n"):
        if "assert" in line:
            test_cases.append(f"{context}\n{line}")
        else:
            context += f"\n{line}"
    if not test_cases:
        print(f"No test cases found for prompt {idx}")
        test_cases = [context]

    # Test
    results = []
    for test_case in test_cases:
        manager = multiprocessing.Manager()
        result = manager.list()
        p = multiprocessing.Process(
            target=_unsafe_execute,
            args=(prediction, test_case, timeout, result),
        )
        p.start()
        p.join(timeout=timeout + 1)
        if p.is_alive():
            p.kill()
        results.append(result[0] if result else "timed out")
    passed = sum(r == "passed" for r in results)
    return idx, {"prediction": prediction, "results": results, "passed": passed, "score": passed / len(results)}


def _validate_code(
    data: list[dict], reward_model: Optional[str] = None, dataset_name: str = None
):
    executor = ProcessPoolExecutor(max_workers=min(os.cpu_count() or 2, 64) // 2)
    futures = []
    for idx, item in enumerate(data):
        future = executor.submit(_validate_code_worker, idx, item["messages"][1]["content"], item["test"], 3)
        futures.append(future)

    for future in tqdm(as_completed(futures), total=len(futures), desc="Validating code"):
        prompt_id, result = future.result()
        data[prompt_id].update(result)
        data[prompt_id]["correct"] = result["passed"] == len(result["results"])
    executor.shutdown()

    # Calculate metrics
    correct_count = sum(item["correct"] for item in data)
    accuracy = correct_count / len(data)

    logger.info(f"Dataset: {dataset_name}, Domain: code, Accuracy: {accuracy:.2%} ({correct_count}/{len(data)})")

    return {"score": accuracy, "total": len(data), "correct": correct_count, "data": data}


def _validate_if_worker(
    gpu_data: list[tuple[int, dict]],
    device: int,
    reward_model: str,
) -> list[tuple[int, dict]]:
    """Worker function to validate using reward model on a specific GPU."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Load model and tokenizer once on specified device
    device_str = f"cuda:{device}"
    rm = AutoModelForSequenceClassification.from_pretrained(
        reward_model,
        torch_dtype=torch.bfloat16,
        device_map=device_str,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(reward_model)

    results = []
    for idx, item in gpu_data:
        # Format conversation
        conv_formatted = tokenizer.apply_chat_template(item["messages"], tokenize=False)
        if tokenizer.bos_token is not None and conv_formatted.startswith(tokenizer.bos_token):
            conv_formatted = conv_formatted[len(tokenizer.bos_token) :]

        # Tokenize
        conv_tokenized = tokenizer(conv_formatted, return_tensors="pt").to(device_str)

        # Get reward score
        with torch.no_grad():
            score = rm(**conv_tokenized).logits[0][0].item()

        results.append((idx, {"score": score}))

        # Clean up individual tensors
        del conv_tokenized

    # Clean up GPU memory after processing all items
    del rm
    del tokenizer
    torch.cuda.empty_cache()

    return results


def _validate_if(
    data: list[dict], reward_model: Optional[str] = None, dataset_name: str = None
):
    """Validate using reward model, loading model once per GPU."""
    if reward_model is None:
        raise ValueError("reward_model must be specified for 'if' domain validation")

    import torch

    num_gpus = torch.cuda.device_count()
    logger.info(f"Validating with reward model on {num_gpus} GPUs")

    # Group data by GPU
    data_by_gpu = [[] for _ in range(num_gpus)]
    for idx, item in enumerate(data):
        data_by_gpu[idx % num_gpus].append((idx, item))

    # Submit one task per GPU
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for gpu_idx, gpu_data in enumerate(data_by_gpu):
            future = executor.submit(
                _validate_if_worker,
                gpu_data,
                gpu_idx,
                reward_model,
            )
            futures.append(future)

        # Collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating with reward model"):
            results = future.result()
            for idx, result in results:
                data[idx].update(result)

    # Calculate average reward score and classify samples
    reward_scores = [item["score"] for item in data]
    avg_score = sum(reward_scores) / len(reward_scores) if reward_scores else 0

    # Classify based on average score
    for item in data:
        item["correct"] = item["score"] >= avg_score

    logger.info(f"Dataset: {dataset_name}, Domain: if, Avg Reward Score: {avg_score:.4f}")

    return {"score": avg_score, "total": len(data), "data": data}


if __name__ == "__main__":
    import fire

    fire.Fire({"infer": infer, "validate": validate, "full": full})
