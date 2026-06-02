from importlib.metadata import version


try:
    import antlr4
    import sympy
    from math_verify import parse, verify
    from sympy.parsing.latex import parse_latex

    assert version("antlr4-python3-runtime").startswith("4.11")
except (ModuleNotFoundError, AssertionError) as e:
    raise type(e)(
        "`sympy`, `math_verify` and `antlr4-python3-runtime==4.11` are required for generating translation task prompt templates. "
        "Please install the required packages via pip install lm-eval[math] or pip install -e .[math]"
    ) from e

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def math_parse(responses: list[list[str]], docs: list[dict]):
    return [[parse(y) for y in x] for x in responses]


def exact_match(references: list[str], predictions: list[list[str]]):
    scores = []
    for reference, prediction in zip(references, predictions):
        answer = parse(reference)
        results = [verify(answer, x) for x in prediction]
        scores.append(sum(results) / len(results))
    return sum(scores) / len(scores)


def format_cot_example(example):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"

    for i, opt in enumerate(options):
        if i >= len(choices):
            break
        prompt += "{}. {}\n".format(choices[i], opt)

    prompt += "Answer: Let's think step by step."

    return prompt
