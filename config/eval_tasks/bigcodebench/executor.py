#!/usr/bin/env python

import sys
import json
import types
import unittest
import os
import shutil
import builtins

from bigcodebench.eval.utils import safe_environment, create_tempdir, reliability_guard, swallow_io, time_limit


def unsafe_execute(
    code: str,
    test_code: str,
    timeout: float,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
) -> dict:
    details = {"status": "fail", "failure": {}}

    with safe_environment(), create_tempdir():
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        reliability_guard(max_as_limit, max_data_limit, max_stack_limit)

        module_name = "__test__"
        new_module = types.ModuleType(module_name)
        new_module.__dict__.update({
            '__builtins__': builtins,
            '__file__': f"{module_name}.py",
            '__package__': None,
            '__doc__': None,
            'sys': sys,
            'os': os,
            'environ': os.environ,
        })

        try:
            full_code = code + "\n" + test_code

            with swallow_io():
                exec(compile(full_code, f"{module_name}.py", 'exec'), new_module.__dict__)
                sys.modules[module_name] = new_module
                TestCases = getattr(new_module, 'TestCases')
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromTestCase(TestCases)
                test_result = unittest.TestResult()
                with time_limit(timeout):
                    suite.run(test_result)

            issues = test_result.failures + test_result.errors
            for test, trace in issues:
                details["failure"][test.id().split(".")[-1]] = trace
            if not details["failure"]:
                details["status"] = "pass"
        except BaseException as e:
            details["failure"]["ALL"] = str(e)
            details["status"] = "fail"

        # Restore for cleanup
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir

    return details


def main():
    if len(sys.argv) != 3:
        print("Usage: executor.py <input_json> <output_json>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, 'r') as f:
        params = json.load(f)

    result = unsafe_execute(
        code=params["code"],
        test_code=params["test_code"],
        timeout=params["timeout"],
        max_as_limit=params["max_as_limit"],
        max_data_limit=params["max_data_limit"],
        max_stack_limit=params["max_stack_limit"],
    )

    with open(output_file, 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
