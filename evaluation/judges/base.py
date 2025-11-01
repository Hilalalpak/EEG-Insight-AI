from abc import ABC, abstractmethod
import logging

class BaseJudge(ABC):

    def __init__(self, name: str, model_name: str, logger: logging.Logger):
        self.name = name
        self.model_name = model_name
        self.logger = logger

    @abstractmethod
    def evaluate(self, question: str, ideal: str, answer: str) -> tuple[int, str]:
        raise NotImplementedError