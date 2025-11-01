import logging

def print_summary(
        results: list,
        api_name: str,
        judge_name: str,
        results_path: str,
        logger: logging.Logger):

    if not results:
        logger.warning("No results to summarize")
        return

    logger.info("Benchmark Summary")
    logger.info(f"Target: {api_name}")
    logger.info(f"Judge:  {judge_name}")

    total_tests = len(results)
    scores = [r["judge_score"] for r in results]
    times = [r["model_duration_sec"] for r in results]

    avg_score = sum(scores) / total_tests
    avg_time = sum(times) / total_tests

    logger.info(f"\nOverall Average: {avg_score:.2f} / 5.0")
    logger.info(f"Average Response Time: {avg_time:.2f}s")

    logger.info("\nScore Distribution:")
    for i in range(6):
        count = scores.count(i)
        logger.info(f"  {i}-Star: {count} tests ({count / total_tests * 100:.1f}%)")

    categories = sorted(list(set(r["category"] for r in results)))
    logger.info("\nAverage Scores by category:")
    for cat in categories:
        cat_scores = [r["judge_score"] for r in results if r["category"] == cat]
        if cat_scores:
            cat_avg = sum(cat_scores) / len(cat_scores)
            logger.info(f"  - {cat:<20}: {cat_avg:.2f} / 5.0 ({len(cat_scores)} tests)")

    logger.info(f"All results saved to {results_path}")