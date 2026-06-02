from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    LCBCodeGenerationDataset,
    LCBCodeExecutionDataset,
    LCBTestOutputPredictionDataset,
    LCBCodeGenerationEvaluator,
    LCBCodeExecutionEvaluator,
    LCBTestOutputEvaluator,
)
from opencompass.datasets.livecodebench import TestOutputPromptConstants


lcb_code_generation_reader_cfg = dict(
    input_columns=[
        "question_content",
        "format_prompt",
    ],
    # output_column='evaluation_sample',
    output_column="question_id",
)

SYSTEM_MESSAGE_GENERIC = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

prompt_template = (
    "{question_content}\n\n{format_prompt}"
)

# Code Generation Tasks
lcb_code_generation_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template=dict(round=[dict(role="HUMAN", prompt=prompt_template)])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

lcb_code_generation_eval_cfg = dict(
    evaluator=dict(
        type=LCBCodeGenerationEvaluator,
        num_process_evaluate=64,
        timeout=6,
    ),
    pred_role="BOT",
)

LCBCodeGeneration_dataset = dict(
    type=LCBCodeGenerationDataset,
    abbr="lcb_code_generation",
    path="opencompass/code_generation_lite",
    reader_cfg=lcb_code_generation_reader_cfg,
    infer_cfg=lcb_code_generation_infer_cfg,
    eval_cfg=lcb_code_generation_eval_cfg,
)

LCB_datasets = [
    LCBCodeGeneration_dataset,
]
