import os
import logging
from evaluation.judges.base import BaseJudge
from evaluation.providers.gemini import GeminiJudge
from evaluation.providers.groq import GroqJudge
from evaluation.providers.cohere import CohereJudge


def create_judge(provider_name: str, judge_config: dict, logger: logging.Logger) -> BaseJudge:
    if provider_name == "GROQ":
        api_key = os.getenv("GROQ_API_KEY")
        return GroqJudge(
            api_key=api_key,
            model=judge_config['model'],
            logger=logger,
            retry_config=judge_config.get('evaluate_retry', {}),
            client_config=judge_config.get('client_config'))

    elif provider_name == "GEMINI":
        api_key = os.getenv("GEMINI_API_KEY")
        return GeminiJudge(
            api_key=api_key,
            model=judge_config['model'],
            logger=logger,
            retry_config=judge_config.get('evaluate_retry', {}))

    elif provider_name == "COHERE":
        api_key = os.getenv("COHERE_API_KEY")
        return CohereJudge(
            api_key=api_key,
            model=judge_config['model'],
            logger=logger)
    else:
        raise ValueError(f"Unknown JUDGE_PROVIDER: '{provider_name}'")