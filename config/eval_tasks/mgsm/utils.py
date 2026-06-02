import re
import string
import numpy as np
from typing import List, Tuple, Any, Dict
from lingua import Language, LanguageDetectorBuilder
from datasets import Dataset


# 初始化语言检测器（只在需要时创建）
_detector = None


def get_language_detector():
    """获取语言检测器实例，延迟初始化"""
    global _detector
    if _detector is None:
        _detector = LanguageDetectorBuilder.from_all_languages().build()
    return _detector


def detect_language_fallback(text: str) -> str:
    """基于关键词的简单语言检测fallback方法

    Args:
        text: 要检测的文本

    Returns:
        语言代码
    """
    if not text:
        return 'unknown'

    text = text.lower()

    # 基于特征词汇和字符的简单检测
    if any(char in text for char in '答案是答えは'):
        if '答案是' in text:
            return 'zh'
        elif '答えは' in text:
            return 'ja'
    elif 'the answer is' in text:
        return 'en'
    elif 'la réponse est' in text or 'la reponse est' in text:
        return 'fr'
    elif 'die antwort lautet' in text:
        return 'de'
    elif 'la respuesta es' in text:
        return 'es'
    elif 'ответ —' in text or 'ответ' in text:
        return 'ru'
    elif 'jibu ni' in text:
        return 'sw'
    elif 'সমাধান' in text or any(char in text for char in 'বাংলা'):
        return 'bn'
    elif 'సమాధానం' in text or any(char in text for char in 'తెలుగు'):
        return 'te'
    elif 'คำตอบคือ' in text or any(char in text for char in 'ไทย'):
        return 'th'

    # 基于字符集的检测
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        return 'zh'
    elif any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
        return 'ja'
    elif any('\u0e00' <= char <= '\u0e7f' for char in text):
        return 'th'
    elif any('\u0980' <= char <= '\u09ff' for char in text):
        return 'bn'
    elif any('\u0c00' <= char <= '\u0c7f' for char in text):
        return 'te'
    elif any('\u0400' <= char <= '\u04ff' for char in text):
        return 'ru'

    return 'en'  # 默认为英语


def detect_language(texts: list[str]) -> list[str]:
    """
    检测文本的语言

    Args:
        text: 要检测的文本

    Returns:
        语言代码（如 'en', 'zh' 等），如果检测失败返回 'unknown'
    """
    detector = get_language_detector()
    detected_language = detector.detect_languages_in_parallel_of(texts)
    results = []
    for lang, text in zip(detected_language, texts):
        if lang is None:
            results.append(detect_language_fallback(text))
        else:
            results.append(lang.iso_code_639_1.name.lower())
    return results


def extract_answer_number(text: str) -> str:
    """从文本中提取数字答案

    Args:
        text: 包含答案的文本

    Returns:
        提取的数字字符串，如果没找到返回空字符串
    """
    if not text:
        return ""
    pattern = "(-?[0-9.,]{2,})|(-?[0-9]+)"
    match = re.findall(pattern, text)
    if match:
        match = match[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m]
            if match:
                match = match[0]
            else:
                match = ""
        match = match.strip()
        return match
    return ""


def accuracy(
    predictions: list[str],
    references: list[str],
    language: str="en",
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
) -> Dict[str, float]:
    """
    综合评测函数：同时计算语种正确性和语种+答案正确性

    Args:
        predictions: 模型生成的答案列表，每个元素是一个包含多个候选答案的列表
        references: 正确答案列表，每个元素是一个字符串或包含问题、答案、语言的元组/字典
        language: 语言代码（如 'en', 'zh' 等）
        regexes_to_ignore: 用于忽略的正则表达式列表
        ignore_case: 是否忽略大小写
        ignore_punctuation: 是否忽略标点符号
        ignore_numbers: 是否忽略数字

    Returns:
        language_correctness: 语种正确性 (0.0-1.0)
        language_and_answer_correctness: 语种和答案都正确的比例 (0.0-1.0)
    """
    pred_langs = np.array(detect_language(predictions))
    lang_match = pred_langs == language

    answers = [extract_answer_number(p) for p in predictions]
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            answers = [re.sub(s, "", x) for x in answers]
            references = [re.sub(s, "", x) for x in references]
    answer_array = np.asarray(answers)
    reference_array = np.asarray(references)

    if ignore_case:
        answer_array = np.char.lower(answer_array)
        reference_array = np.char.lower(reference_array)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        answer_array = np.char.translate(answer_array, table=repl_table)
        reference_array = np.char.translate(reference_array, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        answer_array = np.char.translate(answer_array, table=repl_table)
        reference_array = np.char.translate(reference_array, table=repl_table)

    answer_match = answer_array == reference_array
    both_match = lang_match & answer_match

    return {
        "language_accuracy": np.mean(lang_match),
        "answer_accuracy": np.mean(answer_match),
        "accuracy": np.mean(both_match),
    }
