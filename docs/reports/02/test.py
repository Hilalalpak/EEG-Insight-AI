# simple_test.py
import requests
import json
from datetime import datetime

API_URL = "http://localhost:8001/query"

# Test soruları
QUESTIONS = [
    "What is LPD?",
    "Tell me about patient 9999999999",
    "I heard LPD means low-power discharge. Is that correct?",
    "Compare seizure patterns in EEG 1001717358 vs 1002197945",
    "Show me seizure segments from patient 1001717358"
]


def ask_question(question):
    """Tek bir soru sor ve cevabı al"""
    try:
        payload = {
            "query": question,
            "n_results": 5
        }

        response = requests.post(API_URL, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()

        return {
            "question": question,
            "answer": data.get("llm_response", "No response"),
            "validation": data.get("validation", {}),
            "status": "success"
        }
    except Exception as e:
        return {
            "question": question,
            "answer": f"ERROR: {str(e)}",
            "validation": {},
            "status": "error"
        }


def run_simple_test():
    """Tüm soruları sor ve kaydet"""

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("EEG RAG MODEL TEST")
    print("=" * 80)
    print()

    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] Asking: {question}")
        print("-" * 80)

        result = ask_question(question)
        results.append(result)

        # Ekrana yazdır
        print(f"Answer:\n{result['answer']}\n")

        if result['validation']:
            print(f"Validation: {result['validation']}")

        print()

    # Dosyaya kaydet
    filename = f"test_results_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"✅ Results saved to: {filename}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_simple_test()