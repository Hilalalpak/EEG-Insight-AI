# test_metadata_aware_rag.py - Complete Metadata & Query Type Testing
import requests
import json
import time
from datetime import datetime

API_URL = "http://localhost:8001/query"

# 🆕 COMPREHENSIVE TEST CATEGORIES (10 Query Types)
TESTS = {
    # 1. DEFINITION QUERIES (metadata: is_complete_section, has_definition)
    "Medical Definitions": [
        "What is LPD?",
        "Define GPD and its clinical significance",
        "What are BIRDs in EEG terminology?",
        "Explain the difference between LRDA and GRDA",
        "What is the Ictal-Interictal Continuum?",
        "Define lateralized periodic discharges",
    ],

    # 2. REASONING QUERIES (metadata: has_step_by_step, has_example)
    "Reasoning & How-To": [
        "How to detect LPD step by step?",
        "How do you count frequency in EEG?",
        "Why are breaks important in periodic discharges?",
        "Explain the process of identifying seizure onset",
        "How to differentiate periodic from rhythmic patterns?",
        "What steps are involved in artifact detection?",
    ],

    # 3. PATIENT DATA QUERIES (metadata: eeg_id, expert_consensus, has_clean_signal)
    "Patient Queries": [
        "Tell me about patient 1002379034",
        "What patterns are in patient 1001717358?",
        "Show seizure events for patient 1001717358",
        "Find clean signal data for patient 1002197945",
        "What is the confidence level for patient 1002379034's LPD classification?",
    ],

    # 🆕 4. RISK ASSESSMENT QUERIES (metadata: prob_seizure, is_mixed_pattern, is_edge_case)
    "Risk Assessment": [
        "Show me high-risk seizure patterns",
        "Find segments with concerning mixed patterns",
        "What are dangerous EEG patterns requiring immediate attention?",
        "Show cases with high seizure probability",
        "Find edge cases that might indicate seizure risk",
        "What patterns show transition to ictal activity?",
    ],

    # 🆕 5. COMPARISON QUERIES (metadata: is_high_confidence, mean_snr > 3.0)
    "Pattern Comparisons": [
        "Compare LPD vs GPD patterns",
        "What's the difference between seizure and GRDA?",
        "Compare high-confidence vs low-confidence classifications",
        "Contrast lateralized and generalized periodic discharges",
        "Compare seizure patterns in patient 1001717358 vs 1002197945",
    ],

    # 🆕 6. QUALITY-FOCUSED QUERIES (metadata: has_clean_signal, mean_snr > 5.0)
    "Signal Quality": [
        "Show me clean EEG signals without artifacts",
        "Find high-quality recordings with minimal noise",
        "What segments have the best signal quality?",
        "Show artifact-free seizure data",
        "Find segments with SNR above 5",
    ],

    # 🆕 7. EDGE CASE QUERIES (metadata: is_edge_case, is_mixed_pattern)
    "Edge Cases & Uncertainty": [
        "Show segments with mixed expert opinions",
        "Find uncertain or ambiguous classifications",
        "What are borderline cases between LPD and seizure?",
        "Show edge cases with low confidence",
        "Find patterns where experts disagreed",
    ],

    # 🆕 8. FREQUENCY ANALYSIS (metadata: mean_sef, has_fast_activity, has_slow_activity)
    "Frequency Analysis": [
        "Show me fast frequency activity patterns",
        "Find slow wave activity segments",
        "What patterns have dominant delta frequencies?",
        "Show beta-range activity in EEG",
        "Find segments with spectral edge frequency above 20 Hz",
    ],

    # 🆕 9. AMPLITUDE ANALYSIS (metadata: mean_power, has_high_amplitude)
    "Amplitude Analysis": [
        "Show high amplitude EEG patterns",
        "Find low voltage activity segments",
        "What patterns have elevated power?",
        "Show segments with amplitude above 50 microvolts",
        "Find low amplitude background activity",
    ],

    # 10. GENERAL & COMPLEX QUERIES
    "Complex Multi-Criteria": [
        "Find high-confidence seizure with clean signal and high amplitude",
        "Show me LPD patterns with fast frequencies and mixed expert opinions",
        "What are clean, high-quality GPD patterns from patient 1001717358?",
        "Find uncertain edge cases with slow frequency activity",
    ],

    # 11. EDGE CASES & ERROR HANDLING
    "Error Handling": [
        "Tell me about patient 999999999",  # Non-existent
        "What is XYZ pattern?",  # Unknown term
        "Show me data from 2030",  # Invalid query
        "Is LPD the same as low-power discharge?",  # Misconception
    ]
}


def test_question(question, category):
    """Test single question with comprehensive metrics"""
    start = time.time()

    try:
        response = requests.post(
            API_URL,
            json={
                "query": question,
                "n_results": 5,
                "n_definitions": 3,
                "n_videos": 2
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        elapsed = time.time() - start
        answer = data.get("llm_response", "")
        validation = data.get("validation", {})

        # 🆕 Extract metadata quality metrics
        data_quality = validation.get("data_quality", {})

        return {
            "success": True,
            "time": elapsed,
            "answer": answer,
            "length": len(answer),

            # Source counts
            "eeg_count": len(data.get("retrieved_eeg_segments", {}).get("documents", [[]])[0]),
            "medical_count": len(data.get("retrieved_medical_definitions", {}).get("documents", [[]])[0]),
            "video_count": len(data.get("retrieved_video_reasoning", {}).get("documents", [[]])[0]),

            # Query analysis
            "query_type": data.get("query_type", "N/A"),
            "confidence": validation.get("confidence", "N/A"),

            # 🆕 Metadata quality metrics
            "avg_expert_confidence": data_quality.get("avg_expert_confidence", "N/A"),
            "avg_signal_quality": data_quality.get("avg_signal_quality", "N/A"),
            "high_confidence_sources": data_quality.get("high_confidence_sources", 0),
            "medical_completeness": validation.get("medical_completeness", "N/A"),
            "reasoning_sources": validation.get("reasoning_sources", "N/A"),

            # 🆕 Applied filters
            "filters_applied": data.get("filters_applied", {}),

            # 🆕 Metadata from first EEG result (if available)
            "eeg_metadata_sample": data.get("retrieved_eeg_segments", {}).get("metadatas", [[]])[0][0] if
            data.get("retrieved_eeg_segments", {}).get("metadatas", [[]])[0] else None,
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

    print("=" * 100)
    print("EEG RAG SYSTEM - METADATA-AWARE 3-SOURCE TEST")
    print("Testing: EEG Signals + Medical PDF + Video Transcripts")
    print("Features: 10 Query Types + Intelligent Metadata Filtering")
    print("=" * 100)
    print(f"Total Tests: {total}\n")

    for category, questions in TESTS.items():
        print(f"\n{'=' * 100}")
        print(f"CATEGORY: {category}")
        print(f"{'=' * 100}\n")

        for i, question in enumerate(questions, 1):
            current += 1
            print(f"[{current}/{total}] {question}")
            print("-" * 100)

            result = test_question(question, category)
            result["category"] = category
            result["question"] = question
            results.append(result)

            if result["success"]:
                print(f"✓ Time: {result['time']:.2f}s | Type: {result['query_type']}")
                print(
                    f"✓ Sources → EEG: {result['eeg_count']} | Medical: {result['medical_count']} | Video: {result['video_count']}")
                print(f"✓ Confidence: {result['confidence']}")

                # 🆕 Show metadata quality
                if result.get('avg_expert_confidence') != "N/A":
                    print(
                        f"✓ Data Quality → Expert Consensus: {result['avg_expert_confidence']} | Signal: {result['avg_signal_quality']}")

                # 🆕 Show applied filters
                filters = result.get('filters_applied', {})
                if any(filters.values()):
                    print(f"✓ Filters Applied:")
                    for source, filter_str in filters.items():
                        if filter_str and filter_str != 'None':
                            print(f"    • {source}: {filter_str}")

                # 🆕 Show sample metadata
                if result.get('eeg_metadata_sample'):
                    meta = result['eeg_metadata_sample']
                    print(f"✓ Sample EEG Metadata:")
                    print(f"    • Consensus: {meta.get('expert_consensus', 'N/A')} ({meta.get('confidence', 0):.0%})")
                    print(f"    • Signal: Power={meta.get('mean_power', 0):.1f}, SNR={meta.get('mean_snr', 0):.1f}")

                print(f"✓ Answer Preview: {result['answer'][:120]}...")
            else:
                print(f"✗ FAILED: {result.get('error', 'Unknown')}")
            print()

    return results


def print_summary(results):
    """Print comprehensive performance summary"""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    times = [r["time"] for r in successful]
    lengths = [r["length"] for r in successful]

    print("\n" + "=" * 100)
    print("PERFORMANCE SUMMARY")
    print("=" * 100)

    # Overall stats
    print(f"\n📊 Overall:")
    print(f"  Total Tests: {len(results)}")
    print(f"  Success: {len(successful)} ({len(successful) / len(results) * 100:.1f}%)")
    print(f"  Failed: {len(failed)}")

    # Response times
    if times:
        print(f"\n⏱️  Response Times:")
        print(f"  Average: {sum(times) / len(times):.2f}s")
        print(f"  Median: {sorted(times)[len(times) // 2]:.2f}s")
        print(f"  Min: {min(times):.2f}s")
        print(f"  Max: {max(times):.2f}s")

    # Answer lengths
    if lengths:
        print(f"\n📝 Answer Lengths:")
        print(f"  Average: {sum(lengths) / len(lengths):.0f} chars")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")

    # 🆕 SOURCE USAGE STATS
    print(f"\n📚 Source Usage:")
    eeg_used = [r for r in successful if r.get('eeg_count', 0) > 0]
    medical_used = [r for r in successful if r.get('medical_count', 0) > 0]
    video_used = [r for r in successful if r.get('video_count', 0) > 0]

    print(f"  EEG: {len(eeg_used)}/{len(successful)} queries ({len(eeg_used) / len(successful) * 100:.1f}%)")
    print(
        f"  Medical: {len(medical_used)}/{len(successful)} queries ({len(medical_used) / len(successful) * 100:.1f}%)")
    print(f"  Video: {len(video_used)}/{len(successful)} queries ({len(video_used) / len(successful) * 100:.1f}%)")

    multi_source = [r for r in successful if
                    sum([r.get('eeg_count', 0) > 0, r.get('medical_count', 0) > 0, r.get('video_count', 0) > 0]) >= 2]
    print(
        f"  Multi-source (2+): {len(multi_source)}/{len(successful)} queries ({len(multi_source) / len(successful) * 100:.1f}%)")

    # 🆕 QUERY TYPE DISTRIBUTION
    query_types = {}
    for r in successful:
        qt = r.get('query_type', 'N/A')
        query_types[qt] = query_types.get(qt, 0) + 1

    if query_types:
        print(f"\n🔍 Query Type Distribution:")
        for qt, count in sorted(query_types.items(), key=lambda x: -x[1]):
            percentage = count / len(successful) * 100
            print(f"  {qt:.<25} {count:>3} queries ({percentage:>5.1f}%)")

    # 🆕 METADATA QUALITY METRICS
    quality_available = [r for r in successful if r.get('avg_expert_confidence') != "N/A"]
    if quality_available:
        print(f"\n⭐ Data Quality Metrics (available in {len(quality_available)} queries):")
        high_conf = [r for r in quality_available if r.get('high_confidence_sources', 0) > 0]
        print(f"  High-confidence sources used: {len(high_conf)}/{len(quality_available)} queries")

    # 🆕 FILTER USAGE STATS
    filter_used = [r for r in successful if any(r.get('filters_applied', {}).values())]
    if filter_used:
        print(f"\n🎯 Intelligent Filter Usage:")
        print(
            f"  Queries with filters: {len(filter_used)}/{len(successful)} ({len(filter_used) / len(successful) * 100:.1f}%)")

        # Count filter types
        eeg_filters = [r for r in filter_used if r.get('filters_applied', {}).get('eeg', 'None') != 'None']
        medical_filters = [r for r in filter_used if r.get('filters_applied', {}).get('medical', 'None') != 'None']
        video_filters = [r for r in filter_used if r.get('filters_applied', {}).get('video', 'None') != 'None']

        print(f"    • EEG filters: {len(eeg_filters)} queries")
        print(f"    • Medical filters: {len(medical_filters)} queries")
        print(f"    • Video filters: {len(video_filters)} queries")

    # By category
    print(f"\n📁 By Category:")
    for category in TESTS.keys():
        cat_results = [r for r in results if r.get("category") == category]
        cat_success = [r for r in cat_results if r["success"]]
        if cat_results:
            avg_time = sum(r["time"] for r in cat_success) / len(cat_success) if cat_success else 0
            success_rate = len(cat_success) / len(cat_results) * 100
            print(
                f"  {category:.<40} {len(cat_success):>2}/{len(cat_results):>2} ({success_rate:>5.1f}%) | Avg: {avg_time:>5.2f}s")

    # Slow tests
    slow = [r for r in successful if r["time"] > 10]
    if slow:
        print(f"\n🐌 Slow Tests (>10s): {len(slow)} queries")
        for r in sorted(slow, key=lambda x: -x['time'])[:5]:
            print(f"  • {r['time']:.2f}s: {r['question'][:70]}...")

    # Fast tests
    fast = [r for r in successful if r["time"] < 2]
    if fast:
        print(f"\n⚡ Fast Tests (<2s): {len(fast)} queries")

    # Failed tests
    if failed:
        print(f"\n❌ Failed Tests: {len(failed)}")
        for r in failed:
            print(f"  • {r['question'][:70]}...")
            print(f"    Error: {r.get('error', 'Unknown')[:80]}")


def save_results(results):
    """Save results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_metadata_aware_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved: {filename}")


def analyze_query_type_accuracy(results):
    """🆕 Analyze if query types are correctly detected"""
    print("\n" + "=" * 100)
    print("QUERY TYPE ACCURACY ANALYSIS")
    print("=" * 100)

    # Expected mappings (category -> expected query_type)
    expected_types = {
        "Medical Definitions": "definition",
        "Reasoning & How-To": "reasoning",
        "Patient Queries": "patient_data",
        "Risk Assessment": "risk_assessment",
        "Pattern Comparisons": "comparison",
        "Signal Quality": "quality_focused",
        "Edge Cases & Uncertainty": "edge_case",
        "Frequency Analysis": "frequency_analysis",
        "Amplitude Analysis": "amplitude_analysis",
    }

    correct = 0
    total = 0

    for category, expected_type in expected_types.items():
        cat_results = [r for r in results if r.get("category") == category and r["success"]]
        if cat_results:
            correct_detections = [r for r in cat_results if r.get("query_type") == expected_type]
            accuracy = len(correct_detections) / len(cat_results) * 100
            print(
                f"  {category:.<40} {len(correct_detections):>2}/{len(cat_results):>2} ({accuracy:>5.1f}%) → {expected_type}")
            correct += len(correct_detections)
            total += len(cat_results)

    if total > 0:
        overall_accuracy = correct / total * 100
        print(f"\n  Overall Query Type Detection Accuracy: {overall_accuracy:.1f}% ({correct}/{total})")


if __name__ == "__main__":
    print("\n🚀 Starting Metadata-Aware RAG Test Suite...\n")

    results = run_tests()
    print_summary(results)
    analyze_query_type_accuracy(results)  # 🆕
    save_results(results)

    print("\n" + "=" * 100)
    print("✅ TEST COMPLETE")
    print("=" * 100)
    print("\n💡 Tip: Check the JSON file for detailed results and metadata inspection.")