import os
import json
import logging
import sys


def load_dataset(file_path: str, logger: logging.Logger) -> list:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        logger.info(f"Loaded {len(test_cases)} test cases from {file_path}")
        return test_cases
    except FileNotFoundError:
        logger.error(f"Test file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in test file: {e}")
        sys.exit(1)


def save_results(results: list, file_path: str, logger: logging.Logger):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Couldn't save results: {e}")