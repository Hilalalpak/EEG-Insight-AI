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

    provider_names = config.providers.keys()
    judges = []

    logger.info(f"Providers: {list(provider_names)}")

    for provider_name in provider_names:
        provider_config = config.get_provider_config(provider_name)
        judges.append(create_judge(provider_name.upper(), provider_config, logger))

    output_file = os.path.join(config.results_dir, f"{config.api_name}_multi-judge_{datetime.now():%Y%m%d_%H%M}.json")

    cases = load_dataset(config.test_file, logger=logger)

    results = run_evaluation(
        test_cases=cases,
        judges=judges,
        api_config=config,
        logger=logger)

    if results:
        save_results(results, output_file, logger)
        print_summary(results, config.api_name, "Multi-Judge (Parallel)", output_file, logger)
    else:
        logger.warning("No results")

    logger.info("Done.\n")


if __name__ == "__main__":
    main()