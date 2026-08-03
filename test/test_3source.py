# advanced_test.py - Added Video RAG Test
import requests
import json
import time
from datetime import datetime

API_URL = "http://localhost:8001/query"

# Test Categories
TESTS = {
    "Medical Terms": [
        "What is LPD?",
        "Explain GPD and its clinical significance",
        "What are BIRDs in EEG terminology?",
        "Differentiate between LRDA and GRDA",
        "What is the Ictal-Interictal Continuum?",
    ],

    # 🆕 REASONING QUERIES (Video retrieval test)
    "Reasoning & How-To": [
        "How to detect LPD?",  # Video: detection method
        "How to count frequency in EEG?",  # Video: frequency calculation
        "Why are breaks important in periodic discharges?",  # Video: reasoning
        "Explain the difference between periodic and rhythmic",  # Video: comparison
        "How to identify artifact in EEG?",  # Video: warning/tips
    ],

    "Patient Queries": [
        "Tell me about patient 1002379034",
        "What patterns are in patient 1001717358?",
        "Show seizure events for patient 1001717358",
        "Find patient 42165 data",
        "What is the confidence for patient 1002379034's LPD?",
        "Tell me about patient 999999999",
    ],

    "Pattern Analysis": [
        "Show high-confidence seizure segments",
        "What are typical seizure signal characteristics?",
        "Find segments with mixed expert opinions",
        "What patterns have high amplitude and fast activity?",
        "Show segments with clean signal quality",
    ],

    "Comparisons": [
        "Compare LPD vs GPD patterns",
        "Compare seizure in EEG 1001717358 vs 1002197945",
        "Difference between high and low confidence classifications?",
    ],

    "Edge Cases": [
        "Tell me about patient 999999999",
        "What is XYZ pattern?",
        "Is LPD the same as low-power discharge?",
    ]
}


def test_question(question, category):
    """Test single question with metrics"""
    start = time.time()

    try:
        response = requests.post(
            API_URL,
            json={
                "query": question,
                "n_results": 5,
                "n_videos": 2  # 🆕 Video retrieval
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        elapsed = time.time() - start
        answer = data.get("llm_response", "")

        return {
            "success": True,
            "time": elapsed,
            "answer": answer,
            "length": len(answer),
            "eeg_count": len(data.get("retrieved_signal_segments", {}).get("documents", [[]])[0]),
            "medical_count": len(data.get("retrieved_document_chunks", {}).get("documents", [[]])[0]),
            "video_count": len(data.get("retrieved_transcript_chunks", {}).get("documents", [[]])[0]),  # 🆕
            "confidence": data.get("validation", {}).get("confidence", "N/A"),
            "query_type": data.get("query_type", "N/A"),  # 🆕
        }
    except Exception as e:
        return {
            "success": False,
            "time": time.time() - start,
            "error": str(e),
        }


def run_tests():
    """Run all tests and collect metrics"""
    results = []
    total = sum(len(questions) for questions in TESTS.values())
    current = 0

    print("=" * 80)
    print("EEG RAG SYSTEM - 3-SOURCE TEST (EEG + Medical + Video)")  # 🆕
    print("=" * 80)
    print(f"Total Tests: {total}\n")

    for category, questions in TESTS.items():
        print(f"\n{'=' * 80}")
        print(f"CATEGORY: {category}")
        print(f"{'=' * 80}\n")

        for i, question in enumerate(questions, 1):
            current += 1
            print(f"[{current}/{total}] {question}")
            print("-" * 80)

            result = test_question(question, category)
            result["category"] = category
            result["question"] = question
            results.append(result)

            if result["success"]:
                print(f"✓ Time: {result['time']:.2f}s")
                print(f"✓ Length: {result['length']} chars")
                print(
                    f"✓ EEG: {result['eeg_count']} | Medical: {result['medical_count']} | Video: {result.get('video_count', 0)}")  # 🆕
                print(f"✓ Confidence: {result['confidence']} | Type: {result.get('query_type', 'N/A')}")  # 🆕
                print(f"✓ Answer: {result['answer'][:150]}...")
            else:
                print(f"✗ FAILED: {result.get('error', 'Unknown')}")
            print()

    return results


def print_summary(results):
    """Print performance summary"""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    times = [r["time"] for r in successful]
    lengths = [r["length"] for r in successful]

    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    print(f"\n📊 Overall:")
    print(f"  Total: {len(results)}")
    print(f"  Success: {len(successful)} ({len(successful) / len(results) * 100:.1f}%)")
    print(f"  Failed: {len(failed)}")

    if times:
        print(f"\n⏱️  Response Times:")
        print(f"  Average: {sum(times) / len(times):.2f}s")
        print(f"  Median: {sorted(times)[len(times) // 2]:.2f}s")
        print(f"  Min: {min(times):.2f}s")
        print(f"  Max: {max(times):.2f}s")

    if lengths:
        print(f"\n📝 Answer Lengths:")
        print(f"  Average: {sum(lengths) / len(lengths):.0f} chars")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")

    # 🆕 VIDEO USAGE STATS
    video_used = [r for r in successful if r.get('video_count', 0) > 0]
    if video_used:
        print(f"\n🎥 Video Reasoning Usage:")
        print(
            f"  Used in: {len(video_used)}/{len(successful)} queries ({len(video_used) / len(successful) * 100:.1f}%)")
        avg_video = sum(r.get('video_count', 0) for r in video_used) / len(video_used)
        print(f"  Avg chunks per query: {avg_video:.1f}")

    # 🆕 QUERY TYPE DISTRIBUTION
    query_types = {}
    for r in successful:
        qt = r.get('query_type', 'N/A')
        query_types[qt] = query_types.get(qt, 0) + 1

    if query_types:
        print(f"\n🔍 Query Type Distribution:")
        for qt, count in sorted(query_types.items(), key=lambda x: -x[1]):
            print(f"  {qt}: {count} queries ({count / len(successful) * 100:.1f}%)")

    # By category
    print(f"\n📁 By Category:")
    for category in TESTS.keys():
        cat_results = [r for r in results if r.get("category") == category]
        cat_success = [r for r in cat_results if r["success"]]
        if cat_results:
            avg_time = sum(r["time"] for r in cat_success) / len(cat_success) if cat_success else 0
            print(f"  {category}: {len(cat_success)}/{len(cat_results)} success | Avg: {avg_time:.2f}s")

    # Slow tests
    slow = [r for r in successful if r["time"] > 10]
    if slow:
        print(f"\n🐌 Slow Tests (>10s):")
        for r in slow:
            print(f"  - {r['time']:.2f}s: {r['question'][:60]}...")

    # Failed tests
    if failed:
        print(f"\n❌ Failed Tests:")
        for r in failed:
            print(f"  - {r['question'][:60]}...")
            print(f"    Error: {r.get('error', 'Unknown')}")


def save_results(results):
    """Save results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved: {filename}")


if __name__ == "__main__":
    results = run_tests()
    print_summary(results)
    save_results(results)

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)

