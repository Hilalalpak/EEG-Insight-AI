import json
import time
import logging
from google import genai
from google.genai import types as genai_types

from evaluation.judges.base import BaseJudge
from evaluation.judges.prompts import JUDGE_PROMPT

class GeminiJudge(BaseJudge):
    def __init__(self, api_key: str, model: str, logger: logging.Logger, retry_config: dict = None):
        super().__init__(name="Gemini", model_name=model, logger=logger)
        if not api_key:
            raise ValueError("Gemini requires an API key.")

        self.retry_config = retry_config
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def evaluate(self, question: str, ideal: str, answer: str) -> tuple[int, str]:
        prompt = JUDGE_PROMPT.format(question=question, ideal_answer=ideal, model_answer=answer)

        total_attempts = self.retry_config.get('max_retries') + 1
        retry_sec = self.retry_config.get('retry_sec')

        raw_text = ""
        for attempt in range(1, total_attempts + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"response_mime_type": "application/json"})

                raw_text = getattr(response, "text", None)
                if not raw_text:
                    raw_text = response.candidates[0].content.parts[0].text

                result = json.loads(raw_text)

                score = int(result.get("score", 0))
                reasoning = result.get("reasoning", "No reasoning provided.")

                return score, reasoning

            except (genai_types.generation_types.StopCandidateException,
                    genai_types.generation_types.BlockedPromptException) as e:
                self.logger.error(f"Gemini blocked the content: {e}")
                raise Exception(f"Content filtering blocked evaluation: {e}")

            except (genai.errors.ClientError, genai.errors.ServerError) as e:
                self.logger.warning(f"Gemini API error (attempt {attempt}/{total_attempts}): {e}")

                if attempt < total_attempts:
                    self.logger.warning(f"Pausing for {retry_sec}s before retry...")
                    time.sleep(retry_sec)

            except json.JSONDecodeError as e:
                self.logger.error(f"Gemini returned invalid JSON: {e}")
                break

            except Exception as e:
                self.logger.error(f"Unexpected Gemini error: {e}")
                break

        raise Exception("Gemini evaluation failed after all retries")