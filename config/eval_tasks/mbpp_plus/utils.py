import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval", keep_in_memory=True)
    # test_cases = ["assert add(2, 3)==5"]
    # candidates = [["def add(a,b): return a*b"]]
    # results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def mbpp_build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[(r if r.find("```") == -1 else r[: r.find("```")]) for r in resp] for resp, doc in zip(resps, docs)]


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = [1]):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]
