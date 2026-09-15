import json
import os
import pandas as pd
import fire
import subprocess
import time
import numpy as np
from glob import glob
from tqdm import trange


LM_EVAL_TASKS = {
    # "math": ["gsm8k", "gsm1k", "math500"],
    "if": ["ifeval", "commonsense_qa"],
    "math": ["gsm8k", "math500"],
    "code": ["humaneval_plus", "mbpp_plus"],
}


def run_eval(
    model_path: str,
    domains: list[str] | str,
    output_path: str | None = None,
    tensor_parallel_size: int = 1,
    repeats: int = 1,
    bashrc_path: str = "~/.bashrc",
    gen_env_name: str = "base",
    eval_env_name: str = "base",
):
    num_gpus = _get_gpu_count()
    assert num_gpus >= tensor_parallel_size, f"Not enough GPUs. Required: {tensor_parallel_size}, Available: {num_gpus}"

    if domains == "all":
        domains = list(LM_EVAL_TASKS.keys())
    elif isinstance(domains, str):
        domains = [domains]
    if output_path is None:
        output_path = os.path.join(model_path, f"eval_{'_'.join(domains)}_{repeats}")
    os.makedirs(output_path, exist_ok=True)

    # 按照 tensor_parallel_size 将 GPU 分组
    gpu_groups = _partition_gpus_by_tp(num_gpus, tensor_parallel_size)
    print(f"Available GPUs: {num_gpus}, Tensor Parallel Size: {tensor_parallel_size}")
    print(f"GPU Groups: {gpu_groups}")

    # 判断是否需要运行 bigcodebench
    # need_bigcodebench = "code" in domains or "all" in domains
    need_bigcodebench = False

    # 构建任务列表，优先执行 bigcodebench 生成任务
    task_queue = []
    if need_bigcodebench:
        for i in range(repeats):
            task_queue.append(("bigcodebench_gen", i + 1))
    for i in range(repeats):
        task_queue.append(("lm_eval", i + 1))
    print(f"Total tasks to run: {len(task_queue)}")

    # 将任务分配到 GPU 组上并行执行
    running_tasks = {}  # gpu_group_idx -> (process, task_info)
    count = 0
    bigcodebench_gen_total = repeats if need_bigcodebench else 0
    bigcodebench_gen_completed = 0
    bigcodebench_eval_proc = None  # bigcodebench 评测进程

    while task_queue or running_tasks or bigcodebench_eval_proc:
        # 检查已完成的任务，释放 GPU 组
        completed_groups = []
        for group_idx, (proc, task_info) in running_tasks.items():
            if proc.poll() is not None:
                completed_groups.append(group_idx)
                task_type, task_idx = task_info
                if proc.returncode != 0:
                    print(f"Warning: Task {task_info} on GPU group {group_idx} exited with code {proc.returncode}")
                else:
                    print(f"Task {task_info} on GPU group {group_idx} completed successfully")
                # 跟踪 bigcodebench 生成任务完成数
                if task_type == "bigcodebench_gen":
                    bigcodebench_gen_completed += 1
        for group_idx in completed_groups:
            del running_tasks[group_idx]

        # 当所有 bigcodebench 生成任务完成时，立即启动评测任务
        if need_bigcodebench and bigcodebench_gen_completed == bigcodebench_gen_total and bigcodebench_eval_proc is None:
            print("All BigCodeBench generation tasks completed. Starting evaluation...")
            time.sleep(2)  # 确保文件写入完成
            bigcodebench_eval_proc = _start_bigcodebench_evaluation(output_path, bashrc_path, eval_env_name, repeats)

        # 检查 bigcodebench 评测进程是否完成
        if bigcodebench_eval_proc is not None and bigcodebench_eval_proc.poll() is not None:
            if bigcodebench_eval_proc.returncode != 0:
                print(f"Warning: BigCodeBench evaluation exited with code {bigcodebench_eval_proc.returncode}")
            else:
                print("BigCodeBench evaluation completed successfully.")
            bigcodebench_eval_proc = None  # 标记为已完成

        # 分配新任务到空闲的 GPU 组
        available_groups = [i for i in range(len(gpu_groups)) if i not in running_tasks]
        while task_queue and available_groups:
            group_idx = available_groups.pop(0)
            gpu_ids = gpu_groups[group_idx]
            task_type, task_idx = task_queue.pop(0)
            task_output_path = os.path.join(output_path, str(task_idx))
            if task_type == "lm_eval":
                seed = 42 + task_idx - 1
                proc = _start_lm_eval(model_path, task_output_path, domains, gpu_ids, seed, bashrc_path, gen_env_name)
            elif task_type == "bigcodebench_gen":
                proc = _start_bigcodebench_generation(model_path, task_output_path, gpu_ids, bashrc_path, gen_env_name)
            running_tasks[group_idx] = (proc, (task_type, task_idx))
            print(f"Started task {(task_type, task_idx)} on GPU group {group_idx} (GPUs: {gpu_ids})")
            time.sleep(2)  # avoid port conflicting

        if running_tasks or bigcodebench_eval_proc:
            count += 1
            if count % 30 == 0:
                eval_status = ", bigcodebench eval running" if bigcodebench_eval_proc else ""
                print(
                    f"[{time.strftime('%H:%M:%S')}] {len(task_queue)} waiting tasks, {len(running_tasks)} running tasks{eval_status}..."
                )
            time.sleep(1)

    print("All generation tasks completed.")
    time.sleep(2)

    # 汇总评测结果
    results = _collect_lm_eval_results(output_path, repeats, domains)
    if need_bigcodebench:
        bigcodebench_results = _collect_bigcodebench_results(output_path, repeats)
        results["bigcodebench"] = bigcodebench_results * 100
    row_means = results.mean(axis=1)
    results["Average"] = row_means
    col_means = results.mean(axis=0)
    results.loc["Overall"] = col_means

    print("Evaluation Results:")
    print(results)
    results.to_csv(os.path.join(output_path, "summary.csv"), float_format="%.4f")


def _get_gpu_count():
    # 优先读取 CUDA_VISIBLE_DEVICES 环境变量
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices and cuda_visible_devices.strip():
        # 解析 CUDA_VISIBLE_DEVICES，例如 "0,2,4-6" -> [0, 2, 4, 5, 6]
        visible_gpus = []
        for part in cuda_visible_devices.split(","):
            part = part.strip()
            if "-" in part:
                start, end = map(int, part.split("-"))
                visible_gpus.extend(range(start, end + 1))
            else:
                visible_gpus.append(int(part))
        return len(visible_gpus)

    # 如果未设置 CUDA_VISIBLE_DEVICES，使用 nvidia-smi 获取总数量
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        # 注意：--query-gpu=count 返回的是总数量（所有行相同），取第一行
        count = int(result.stdout.strip().split("\n")[0])
        return count
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return 0


def _partition_gpus_by_tp(num_gpus: int, tensor_parallel_size: int) -> list[list[int]]:
    # 获取 CUDA_VISIBLE_DEVICES 中的 GPU ID 列表
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices and cuda_visible_devices.strip():
        visible_gpus = []
        for part in cuda_visible_devices.split(","):
            part = part.strip()
            if "-" in part:
                start, end = map(int, part.split("-"))
                visible_gpus.extend(range(start, end + 1))
            else:
                visible_gpus.append(int(part))
    else:
        # 如果未设置 CUDA_VISIBLE_DEVICES，使用 0, 1, 2, ...
        visible_gpus = list(range(num_gpus))

    num_groups = num_gpus // tensor_parallel_size
    groups = []
    for i in range(num_groups):
        start = i * tensor_parallel_size
        end = start + tensor_parallel_size
        groups.append(visible_gpus[start:end])
    return groups


def _start_lm_eval(
    model_path: str,
    output_path: str,
    domains: list[str],
    gpu_ids: list[int],
    seed: int,
    bashrc_path: str,
    env_name: str,
):
    output_path = os.path.join(output_path, "lm_eval")
    os.makedirs(output_path, exist_ok=True)  # 确保输出目录存在
    tasks = set()
    for domain in domains:
        if domain == "all":
            for t in LM_EVAL_TASKS.values():
                tasks.update(t)
        else:
            tasks.update(LM_EVAL_TASKS[domain])
    tasks = list(tasks)
    command = f"""
source {bashrc_path} &&  \
conda activate {env_name} &&  \
CUDA_VISIBLE_DEVICES={",".join(map(str, gpu_ids))} lm_eval \
    --model vllm \
    --model_args pretrained={model_path},tensor_parallel_size={len(gpu_ids)},data_parallel_size=1,dtype=bfloat16,gpu_memory_utilization=0.96,seed={seed},trust_remote_code=True \
    --seed {seed} \
    --include_path ./config \
    --tasks {",".join(tasks)} \
    --apply_chat_template True \
    --fewshot_as_multiturn \
    --batch_size auto \
    --confirm_run_unsafe_code \
    --output_path {output_path} \
    --log_samples \
    > {output_path}/stdout.log 2> {output_path}/stderr.log
"""
    print("Running LM Evaluation with command:")
    print(command)
    return subprocess.Popen(command, shell=True, executable="/bin/bash")


def _start_bigcodebench_generation(
    model_path: str, output_path: str, gpu_ids: list[int], bashrc_path: str, env_name: str
):
    output_path = os.path.join(output_path, "bigcodebench")
    os.makedirs(output_path, exist_ok=True)  # 确保输出目录存在
    command = f"""
source {bashrc_path} &&  \
conda activate {env_name} &&  \
CUDA_VISIBLE_DEVICES={",".join(map(str, gpu_ids))} bigcodebench.evaluate \
    --model {model_path} \
    --max_model_len 4096 \
    --max_new_tokens 4096 \
    --tp {len(gpu_ids)} \
    --n_samples 2 \
    --temperature 0.1 \
    --split complete \
    --subset full \
    --no_execute \
    --output_path {output_path} \
    > {output_path}/generation_stdout.log 2> {output_path}/generation_stderr.log
"""
    print("Running BigCodeBench Generation with command:")
    print(command)
    return subprocess.Popen(command, shell=True, executable="/bin/bash")


def _start_bigcodebench_evaluation(output_path: str, bashrc_path: str, env_name: str, repeats: int):
    command = f"""
source {bashrc_path} &&  \
conda activate {env_name}"""
    for i in range(repeats):
        task_output_path = os.path.join(output_path, str(i + 1), "bigcodebench")
        command += f""" &&  \
echo Evaluating repeat {i + 1} &&  \
bigcodebench.evaluate \
    --samples {task_output_path}/predictions.jsonl \
    --split complete \
    --subset full \
    --execution local \
    --min-time-limit 2 \
    --parallel {(os.cpu_count() or 4)} \
    > {task_output_path}/evaluation_stdout.log 2> {task_output_path}/evaluation_stderr.log"""

    print("Running BigCodeBench Evaluation with command:")
    print(command)
    return subprocess.Popen(command, shell=True, executable="/bin/bash")


def _collect_lm_eval_results(output_path: str, repeats: int, domains: list[str]):
    benchmarks = {
        "ifeval": "inst_level_strict_acc,none",
        "commonsense_qa": "exact_match,flexible-extract",
        "gsm8k": "exact_match,flexible-extract",
        "gsm1k": "exact_match,flexible-extract",
        "math500": "math_verify,extract_answers",
        "humaneval_plus": "pass@1,extract_code",
        "mbpp_plus": "pass@1,extract_code",
    }
    eval_benchmarks = [b for domain in domains for b in LM_EVAL_TASKS[domain]]
    results = np.zeros((repeats, len(eval_benchmarks)))
    for i in range(repeats):
        result_path = glob(os.path.join(output_path, str(i + 1), "lm_eval", "results*"))[-1]
        with open(result_path, "r") as f:
            result = json.load(f)["results"]
        for j, name in enumerate(eval_benchmarks):
            key = benchmarks[name]
            if name in result:
                num = result[name][key] * 100
                results[i, j] = num
    df = pd.DataFrame(data=results, index=[str(i) for i in range(repeats)], columns=eval_benchmarks)
    return df


def _collect_bigcodebench_results(output_path: str, repeats: int):
    results = np.zeros((repeats, 1))
    for i in range(repeats):
        result_path = os.path.join(output_path, str(i + 1), "bigcodebench", "predictions_pass_at_k.json")
        with open(result_path, "r") as f:
            results[i, 0] = json.load(f)["pass@1"]
    return results


if __name__ == "__main__":
    fire.Fire(run_eval)
