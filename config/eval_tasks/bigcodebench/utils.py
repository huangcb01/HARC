import warnings
import json
import subprocess
import tempfile
import os
from typing import Any
from joblib import Parallel, delayed, cpu_count
from bigcodebench.sanitize import sanitize

warnings.filterwarnings('ignore', category=SyntaxWarning)
BCB_PYTHON_PATH = os.environ.get("BCB_PYTHON_PATH", "python")


def _process_single(resp, doc) -> list[str]:
    return [doc["code_prompt"] + "\n    pass\n" + sanitize(r, doc["entry_point"]) for r in resp]


def extract_code(resps: list[list[str]], docs: list[dict]):
    return Parallel(n_jobs=cpu_count(), verbose=5)(
        delayed(_process_single)(resp, doc) for resp, doc in zip(resps, docs)
    )

def process_results(doc: dict[str, Any], results: list[str]):
    TIMEOUT = 10
    SPACE_LIMIT = 30 * 1024
    STACK_LIMIT = 10

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(
            {
                "code": results[0],
                "test_code": doc["test"],
                "timeout": TIMEOUT,
                "max_as_limit": SPACE_LIMIT,
                "max_data_limit": SPACE_LIMIT,
                "max_stack_limit": STACK_LIMIT,
            },
            f,
        )
        input_file = f.name

    output_file = input_file.replace(".json", "_result.json")

    try:
        executor_script = os.path.join(os.path.dirname(__file__), "executor.py")
        proc = subprocess.run(
            [BCB_PYTHON_PATH, executor_script, input_file, output_file],
            timeout=TIMEOUT + 5,
            capture_output=True,
        )
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                details = json.load(f)
        else:
            details = {"status": "fail", "failure": {"ALL": f"No output file. stderr: {proc.stderr.decode()}"}}
    except subprocess.TimeoutExpired:
        details = {"status": "fail", "failure": {"ALL": "Subprocess timeout"}}
    except Exception as e:
        details = {"status": "fail", "failure": {"ALL": str(e)}}
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)

    details["pass"] = 1 if details.get("status") == "pass" else 0
    return details
