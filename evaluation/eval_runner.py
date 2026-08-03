import time
import logging
import concurrent.futures

from infrastructure.conf.pydantic_models import BenchmarkConfigModel

from evaluation.judges.base import BaseJudge
from evaluation.api_client import query_target


def run_evaluation(
        test_cases: list,
        judges: list[BaseJudge],
        api_config: BenchmarkConfigModel,
        logger: logging.Logger) -> list:

    all_results = []
    total_cases = len(test_cases)

    provider_configs = api_config.providers
    delays = [provider_configs.get(j.name.lower(), {}).get('delay_sec', 0) for j in judges]
    max_delay = max(delays) if delays else 0
    logger.info(f"Using a max delay between API limits: {max_delay}s")

    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        ideal_answer = case["ideal_answer"]
        category = case.get("category", "Unknown")

        logger.info(f"\nTest {i}/{total_cases} - Category: {category}")
        logger.info(f"Question: {question}")

        model_response, status, duration = query_target(
            question,
            api_config.api_url,
            api_config.api_payload,
            api_config.api_timeout,
            logger,
            api_config.api_input_key,
            api_config.api_output_key)

        logger.info(f"Target responded in {duration:.2f}s: {model_response[:100]}...")

        evaluation_results = []
        if status == "error":
            score, reasoning = 0, "Target API failed to respond"
            for judge in judges:
                evaluation_results.append({
                    "judge_name": judge.name,
                    "judge_model": judge.model_name,
                    "score": 0,
                    "reasoning": "Target API failed to respond"})
        else:
            logger.info(f"All judges starting as parallel")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(judges)) as executor:
                futures = {executor.submit(judge.evaluate, question, ideal_answer, model_response): judge for judge in judges}

                for future in concurrent.futures.as_completed(futures):
                    judge = futures[future]
                    try:
                        score, reasoning = future.result()
                        logger.info(f"  -> Judge {judge.name} responsed. Score: {score}/5 - {reasoning[:50]}...")
                        evaluation_results.append({
                            "judge_name": judge.name,
                            "judge_model": judge.model_name,
                            "score": score,
                            "reasoning": reasoning})
                    except Exception as e:
                        logger.error(f"{judge.name} failed: {e}")
                        evaluation_results.append({
                            "judge_name": judge.name,
                            "judge_model": judge.model_name,
                            "score": 0,
                            "reasoning": f"Judge Error: {str(e)}"})

        result_data = {
            "category": category,
            "question": question,
            "ideal_answer": ideal_answer,
            "model_answer": model_response,
            "model_status": status,
            "model_duration_sec": duration,
            "evaluation_results": evaluation_results}

        all_results.append(result_data)

        if i < total_cases and max_delay > 0:
            logger.info(f"Waiting {max_delay}s")
            time.sleep(max_delay)

    return all_results