"""
A wrapper class for the SentenceTransformer (SapBERT) model to create text embeddings
"""
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Union

class TextEmbedder:

    def __init__(self, logger: logging.Logger, hf_model_path: str):

        self.logger = logger
        self.model_name = hf_model_path
        try:
            self.model = SentenceTransformer(hf_model_path)
            self.logger.info(f"Embedding model loaded: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Model load failed for '{self.model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Could not load model: {self.model_name}") from e

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Converts a text or text list to an embedding vector.
        """
        if not text:
            self.logger.warning("Empty input to embed()")
            return []

        try:
            embeddings = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            input_type_repr = repr(type(text))
            self.logger.error(f"Encoding failed for input type {input_type_repr}: {e}", exc_info=True)
            return []