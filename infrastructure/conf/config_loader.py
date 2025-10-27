"""
Loads, merges, and validates all .yml config files.
Overrides with ENV VARS if exist.
"""
import yaml
import os
import sys
from typing import Dict, Any, Type
from pydantic import ValidationError

from infrastructure.conf.interfaces import (DBConfigInterface, LLMConfigInterface, PipelineConfigInterface, RAGCoreConfigInterface,LoggingConfigInterface, MiscConfigInterface)
from infrastructure.conf.pydantic_models import (DBConfigModel, LLMConfigModel, PipelineConfigModel, RAGCoreConfigModel,LoggingConfigModel, MiscConfigModel)

def _deep_merge(dict1: dict, dict2: dict) -> dict:
    """Recursively merges dict2 into dict1."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

class ConfigLoader:
    def __init__(self, config_paths: dict[str, str]):
        raw_config = {}

        for name, path in config_paths.items():
            try:
                file_path = os.environ.get(f"CONFIG_PATH_{name.upper()}", path)

                with open(file_path, 'r') as f:
                    new_config = yaml.safe_load(f)
                    if new_config:
                        raw_config = _deep_merge(raw_config, new_config)
                print(f"ConfigLoader: Loaded '{name}': {file_path}")
            except FileNotFoundError:
                if 'llm_env' in name:
                    print(f"ConfigLoader: Optional config '{name}' not found at {path}. Using defaults.")
                    continue
                raise FileNotFoundError(f"Config file not found: {path}. Check path.")
            except yaml.YAMLError as e:
                raise RuntimeError(f"YAML Read Error ({path}): {e}")

        self.raw_config = raw_config
        self.flat_config = self._flatten_config(self.raw_config)

    @staticmethod
    def _flatten_config(d: dict, parent_key: str = '', sep: str = '__') -> dict:
        """Flattens dict and overrides with env vars."""
        items = {}
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items[new_key] = v
                items.update(ConfigLoader._flatten_config(v, new_key, sep=sep))
            else:
                env_value = os.environ.get(new_key.upper())
                items[new_key] = env_value if env_value is not None else v
        return items

    def _load_model(self, model_class: Type):
        """Validates the flat config against a pydantic model."""
        try:
            return model_class.model_validate(self.flat_config)
        except ValidationError as e:
            print(f"Config validation failed for {model_class.__name__}.", file=sys.stderr)
            print(e, file=sys.stderr)
            raise RuntimeError(f"{model_class.__name__} validation error: {e}")

    def get_db_config(self) -> DBConfigInterface:
        return self._load_model(DBConfigModel)
    def get_llm_config(self) -> LLMConfigInterface:
        return self._load_model(LLMConfigModel)
    def get_pipeline_config(self) -> PipelineConfigInterface:
        return self._load_model(PipelineConfigModel)
    def get_rag_core_config(self) -> RAGCoreConfigInterface:
        return self._load_model(RAGCoreConfigModel)
    def get_logging_config(self) -> LoggingConfigInterface:
        return self._load_model(LoggingConfigModel)
    def get_misc_config(self) -> MiscConfigInterface:
        return self._load_model(MiscConfigModel)