# advanced_test.py - Comprehensive Performance Test
import requests
import json
import time
from datetime import datetime

API_URL = "http://localhost:8001/query"

# Test Categories
TESTS = {
    "Medical Terms": [
        "What is LPD?",  # Basic medical definition - tests PDF retrieval
        "Explain GPD and its clinical significance",  # Clinical context understanding
        "What are BIRDs in EEG terminology?",  # Brief Potentially Ictal Rhythmic Discharges
        "Differentiate between LRDA and GRDA",  # Lateralized vs Generalized rhythmic delta
        "What is the Ictal-Interictal Continuum?",  # Complex medical concept
    ],

    "Patient Queries": [
        "Tell me about patient 1002379034",  # Valid EEG ID from your list
        "What patterns are in patient 1001717358?",  # Valid EEG ID - pattern detection
        "Show seizure events for patient 1001717358",  # Pattern + EEG ID combination
        "Find patient 42165 data",  # Valid patient ID from your list
        "What is the confidence for patient 1002379034's LPD?",  # Confidence + pattern query
        "Tell me about patient 999999999",  # Non-existent ID - error handling
    ],

    "Pattern Analysis": [
        "Show high-confidence seizure segments",  # Confidence filtering (>0.8) + seizure pattern
        "What are typical seizure signal characteristics?",  # Pattern description from data
        "Find segments with mixed expert opinions",  # Low consensus (confidence <0.6)
        "What patterns have high amplitude and fast activity?",  # Signal features: power>20, SEF>20
        "Show segments with clean signal quality",  # SNR filtering (>5)
    ],

    "Comparisons": [
        "Compare LPD vs GPD patterns",  # Medical definition comparison
        "Compare seizure in EEG 1001717358 vs 1002197945",  # Two valid EEGs comparison
        "Difference between high and low confidence classifications?",  # Meta-analysis
    ],

    "Edge Cases": [
        "Tell me about patient 999999999",  # Non-existent patient ID
        "What is XYZ pattern?",  # Unknown medical term
        "Is LPD the same as low-power discharge?",  # Terminology confusion - LPD = Lateralized Periodic Discharges
    ]
}


def test_question(question, category):
    """Test single question with metrics"""
    start = time.time()

    try:
        response = requests.post(API_URL, json={"query": question, "n_results": 5}, timeout=120)
        response.raise_for_status()
        data = response.json()

        elapsed = time.time() - start
        answer = data.get("llm_response", "")

        return {
            "success": True,
            "time": elapsed,
            "answer": answer,
            "length": len(answer),
            "eeg_count": len(data.get("retrieved_eeg_segments", {}).get("documents", [[]])[0]),
            "medical_count": len(data.get("retrieved_medical_definitions", {}).get("documents", [[]])[0]),
            "confidence": data.get("validation", {}).get("confidence", "N/A"),
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
    print("EEG RAG SYSTEM - COMPREHENSIVE TEST")
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
                print(f"“ Time: {result['time']:.2f}s")
                print(f"“ Length: {result['length']} chars")
                print(f"“ EEG: {result['eeg_count']} | Medical: {result['medical_count']}")
                print(f"“ Confidence: {result['confidence']}")
                print(f"“ Answer: {result['answer'][:150]}...")
            else:
                print(f"— FAILED: {result.get('error', 'Unknown')}")
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

    print(f"\n Overall:")
    print(f"  Total: {len(results)}")
    print(f"  Success: {len(successful)} ({len(successful) / len(results) * 100:.1f}%)")
    print(f"  Failed: {len(failed)}")

    if times:
        print(f"\n Response Times:")
        print(f"  Average: {sum(times) / len(times):.2f}s")
        print(f"  Median: {sorted(times)[len(times) // 2]:.2f}s")
        print(f"  Min: {min(times):.2f}s")
        print(f"  Max: {max(times):.2f}s")

    if lengths:
        print(f"\nAnswer Lengths:")
        print(f"  Average: {sum(lengths) / len(lengths):.0f} chars")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")

    # By category
    print(f"\n By Category:")
    for category in TESTS.keys():
        cat_results = [r for r in results if r.get("category") == category]
        cat_success = [r for r in cat_results if r["success"]]
        if cat_results:
            avg_time = sum(r["time"] for r in cat_success) / len(cat_success) if cat_success else 0
            print(f"  {category}: {len(cat_success)}/{len(cat_results)} success | Avg: {avg_time:.2f}s")

    # Slow tests
    slow = [r for r in successful if r["time"] > 10]
    if slow:
        print(f"\nSlow Tests (>10s):")
        for r in slow:
            print(f"  - {r['time']:.2f}s: {r['question'][:60]}...")

    # Failed tests
    if failed:
        print(f"\n Failed Tests:")
        for r in failed:
            print(f"  - {r['question'][:60]}...")
            print(f"    Error: {r.get('error', 'Unknown')}")


def save_results(results):
    """Save results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nðŸ’¾ Results saved: {filename}")


if __name__ == "__main__":
    results = run_tests()
    print_summary(results)
    save_results(results)

    print("\n" + "=" * 80)
    print("âœ… TEST COMPLETE")
    print("=" * 80)