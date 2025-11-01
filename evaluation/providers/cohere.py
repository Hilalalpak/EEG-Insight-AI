import json
import cohere
import logging

from evaluation.judges.base import BaseJudge
from evaluation.judges.prompts import JUDGE_PROMPT

class CohereJudge(BaseJudge):
    def __init__(self, api_key: str, model: str, logger: logging.Logger):
        super().__init__(name="Cohere", model_name=model, logger=logger)
        if not api_key:
            raise ValueError("CohereJudge requires an COHERE_API_KEY")

        self.client = cohere.Client(api_key=api_key)
        self.model = model

    def evaluate(self, question: str, ideal: str, answer: str) -> tuple[int, str]:
        prompt = JUDGE_PROMPT.format(question=question, ideal_answer=ideal, model_answer=answer)

        raw_text = ""
        try:
            response = self.client.chat(
                model=self.model,
                message=prompt,
                temperature=0.0)

            raw_text = response.text
            try:
                json_start = raw_text.index('{')
                json_end = raw_text.rindex('}') + 1
                clean_json_text = raw_text[json_start:json_end]
            except ValueError:
                raise json.JSONDecodeError("Couldn't find any JSON in Cohere's response",raw_text,0)
            result = json.loads(clean_json_text)

            score = int(result.get("score", 0))
            reasoning = result.get("reasoning", "No reasoning provided.")

            if not (1 <= score <= 5):
                self.logger.warning(f"Got unusual score {score} - expected 1-5")
            return score, reasoning

        except Exception as e:
            self.logger.error(f"Unexpected Cohere error: {e}")
            raise