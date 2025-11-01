import os
from dotenv import load_dotenv
from datetime import datetime

from infrastructure.conf.config_loader import ConfigLoader

from evaluation.eval_runner import run_evaluation
from evaluation.utils.logger_config import setup_logging
from evaluation.utils.data_io import load_dataset, save_results
from evaluation.report import print_summary
from evaluation.judges.judge_factory import create_judge

load_dotenv()

def main():
    logger = setup_logging()

    config_paths = {"benchmark": "infrastructure/conf/benchmark.yml"}
    loader = ConfigLoader(config_paths=config_paths)
    config = loader.get_benchmark_config()

    provider = os.getenv("JUDGE_PROVIDER", "GEMINI").upper()
    provider_config = config.get_provider_config(provider.lower())
    judge = create_judge(provider, provider_config, logger)
    delay = provider_config.get('delay_sec', 5)

    output_file = os.path.join(config.results_dir, f"{config.api_name}_{provider.lower()}_{datetime.now():%Y%Y%m%d_%H%M}.json")

    cases = load_dataset(config.test_file, logger=logger)
    results = run_evaluation(
        test_cases=cases,
        judge=judge,
        api_config=config,
        delay_sec=delay,
        logger=logger)

    if results:
        save_results(results, output_file, logger)
        print_summary(results, config.api_name, judge.model_name, output_file, logger)
    else:
        logger.warning("No results")

    logger.info("Done.\n")


if __name__ == "__main__":
    main()