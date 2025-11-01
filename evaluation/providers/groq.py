import json
import time
import openai
import logging
from evaluation.judges.base import BaseJudge
from evaluation.judges.prompts import JUDGE_PROMPT


class GroqJudge(BaseJudge):
    def __init__(self, api_key: str, model: str, logger: logging.Logger,
                 retry_config: dict = None, client_config: dict = None):
        super().__init__("Groq", model, logger)
        if not api_key:
            raise ValueError("Need a Groq API key.")

        self.retry_config = retry_config
        self.client_config = client_config

        self.client = openai.OpenAI(api_key=api_key, base_url=self.client_config.get('base_url'))
        self.model = model

    def evaluate(self, question: str, ideal: str, answer: str) -> tuple[int, str]:
        prompt = JUDGE_PROMPT.format(question=question, ideal_answer=ideal, model_answer=answer)

        total_attempts = self.retry_config.get('max_retries', 2) + 1
        retry_sec = self.retry_config.get('retry_sec', 30)

        raw_text = ""
        for attempt in range(1, total_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"})

                raw_text = response.choices[0].message.content
                result = json.loads(raw_text)

                score = int(result.get("score", 0))
                reasoning = result.get("reasoning", "No reasoning provided.")

                return score, reasoning

            except (openai.RateLimitError, openai.APIStatusError) as e:
                self.logger.warning(f"Groq API issue (attempt {attempt + 1}/{total_attempts}): {e}")
                if attempt < total_attempts:
                    self.logger.warning(f"Retrying in {retry_sec} seconds")
                    time.sleep(retry_sec)

            except json.JSONDecodeError as e:
                self.logger.error("Groq judge failed to parse JSON: %s", raw_text[:100])
                break

            except Exception as e:
                self.logger.error(f"Groq judge error: {e}")
                break

        raise Exception("Groq evaluation failed after all retries")