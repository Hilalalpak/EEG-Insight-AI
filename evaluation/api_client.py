import time
import requests
import logging


def query_target(
        question: str,
        api_url: str,
        payload_template: dict,
        timeout: int,
        logger: logging.Logger,
        input_key: str,
        output_key: str) -> tuple[str, str, float]:

    start_time = time.time()
    try:
        payload = payload_template.copy()
        payload[input_key] = question

        response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()

        data = response.json()
        duration = time.time() - start_time

        result = data.get(output_key)
        if result is None:
            logger.warning(f"Missing '{output_key}' key in API response.")
            return f"API Error: '{output_key}' key missing.", "error", duration

        logger.info(f"Successful response in {duration:.2f}s")
        return result, "success", duration

    except requests.exceptions.Timeout:
        duration = time.time() - start_time
        logger.error(f"Request timed out after {timeout}s")
        return f"API Error: Timed out after {timeout}s", "error", duration

    except requests.exceptions.RequestException as e:
        duration = time.time() - start_time
        logger.error(f"Request failed: {type(e).__name__} - {e}")
        return f"API Error: {type(e).__name__} - {e}", "error", duration