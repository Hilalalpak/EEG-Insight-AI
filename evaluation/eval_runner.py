import time
import logging

from infrastructure.conf.pydantic_models import BenchmarkConfigModel

from evaluation.judges.base import BaseJudge
from evaluation.api_client import query_target


def run_evaluation(
        test_cases: list,
        judge: BaseJudge,
        api_config: BenchmarkConfigModel,
        delay_sec: int,
        logger: logging.Logger) -> list:

    all_results = []
    total_cases = len(test_cases)

    api_url = api_config.api_url
    payload = api_config.api_payload
    timeout = api_config.api_timeout
    input_key = api_config.api_input_key
    output_key = api_config.api_output_key

    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        ideal_answer = case["ideal_answer"]
        category = case.get("category", "Unknown")

        logger.info(f"\nTest {i}/{total_cases} - Category: {category}")
        logger.info(f"Question: {question}")

        model_response, status, duration = query_target(question, api_url, payload, timeout, logger, input_key, output_key)
        logger.info(f"Target responded in {duration:.2f}s: {model_response[:100]}...")

        if status == "error":
            score, reasoning = 0, "Target API failed to respond"
        else:
            logger.info(f"Judging with {judge.name}...")
            try:
                score, reasoning = judge.evaluate(
                    question, ideal_answer, model_response)
            except Exception as e:
                logger.error(f"Judge failed: {e}")
                score, reasoning = 0, f"Judge Error: {str(e)}"

        logger.info(f"Score: {score}/5 - {reasoning}")

        result_data = {
            "category": category,
            "question": question,
            "ideal_answer": ideal_answer,
            "model_answer": model_response,
            "model_status": status,
            "model_duration_sec": duration,
            "judge_score": score,
            "judge_reasoning": reasoning}

        all_results.append(result_data)

        if i < total_cases:
            logger.info(f"Waiting {delay_sec}s")
            time.sleep(delay_sec)

    return all_results