import logging
from collections import defaultdict

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
    times = [r["model_duration_sec"] for r in results]
    avg_time = sum(times) / total_tests
    logger.info(f"Average Response Time: {avg_time:.2f}s")

    judge_stats = defaultdict(lambda: {"scores": [], "categories": {}})
    judge_names = set()

    for r in results:
        category = r["category"]
        for eval_res in r.get("evaluation_results", []):
            j_name = eval_res["judge_name"]
            judge_names.add(j_name)

            score = eval_res.get("score", 0)
            judge_stats[j_name]["scores"].append(score)

            if category not in judge_stats[j_name]["categories"]:
                judge_stats[j_name]["categories"][category] = []
            judge_stats[j_name]["categories"][category].append(score)

    for j_name in sorted(list(judge_names)):
        stats = judge_stats[j_name]
        scores = stats["scores"]

        if not scores:
            continue

        avg_score = sum(scores) / len(scores)

        logger.info(f"\nStats for Judge: {j_name}")
        logger.info(f"Overall Average: {avg_score:.2f} / 5.0")

        logger.info("Score Distribution:")
        for i in range(6):
            count = scores.count(i)
            logger.info(f"  {i}-Star: {count} tests ({count / len(scores) * 100:.1f}%)")

        logger.info("Average Scores by category:")
        categories = sorted(stats["categories"].keys())
        for cat in categories:
            cat_scores = stats["categories"][cat]
            if cat_scores:
                cat_avg = sum(cat_scores) / len(cat_scores)
                logger.info(f"  - {cat:<20}: {cat_avg:.2f} / 5.0 ({len(cat_scores)} tests)")

    logger.info(f"All results saved to {results_path}")