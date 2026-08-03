# quick_test.py - 5-Question Sanity Check
import requests
import json
import time
from datetime import datetime

API_URL = "http://localhost:8001/query"

# A 5-question sanity check covering all main categories:
# 1. Medical Term
# 2. Reasoning (Video)
# 3. Patient Query
# 4. Comparison
# 5. Edge Case
TESTS = {
    "Quick Sanity Check": [
        "What is LPD?",
        "How to detect LPD?",
        "What patterns are in patient 1001717358?",
        "Compare LPD vs GPD patterns",
        "What is XYZ pattern?",
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
                "n_videos": 2  # Test video retrieval
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
            "video_count": len(data.get("retrieved_transcript_chunks", {}).get("documents", [[]])[0]),
            "confidence": data.get("validation", {}).get("confidence", "N/A"),
            "query_type": data.get("query_type", "N/A"),
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
    print("EEG RAG SYSTEM - QUICK SANITY CHECK (5 Questions)")
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
                    f"✓ EEG: {result['eeg_count']} | Medical: {result['medical_count']} | Video: {result.get('video_count', 0)}")
                print(f"✓ Confidence: {result['confidence']} | Type: {result.get('query_type', 'N/A')}")
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

    # ... (Other summary stats like Length, Video Usage, etc.) ...

    if failed:
        print(f"\n❌ Failed Tests:")
        for r in failed:
            print(f"  - {r['question'][:60]}...")
            print(f"    Error: {r.get('error', 'Unknown')}")


def save_results(results):
    """Save results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quick_test_results_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved: {filename}")


if __name__ == "__main__":
    results = run_tests()
    print_summary(results)
    save_results(results)

    print("\n" + "=" * 80)
    print("✅ QUICK TEST COMPLETE")
    print("=" * 80)