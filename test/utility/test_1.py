# debug_test.py - Detailed debugging for RAG system
import requests
import json
from datetime import datetime

API_URL = "http://localhost:8001/query"


def debug_question(question):
    """Test with detailed output"""
    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    try:
        payload = {
            "query": question,
            "n_results": 5,
            "n_definitions": 3
        }

        print("📤 Sending request...")
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        # Check retrieved data
        eeg_segments = data.get("retrieved_eeg_segments", {})
        medical_defs = data.get("retrieved_medical_definitions", {})

        print("\n📊 RETRIEVED DATA:")
        print(f"  - EEG segments: {len(eeg_segments.get('documents', [[]])[0])}")
        print(f"  - Medical definitions: {len(medical_defs.get('documents', [[]])[0])}")

        # Show first EEG segment
        if eeg_segments.get('documents') and eeg_segments['documents'][0]:
            print("\n📄 FIRST EEG SEGMENT (preview):")
            first_doc = eeg_segments['documents'][0][0]
            print(f"  {first_doc[:200]}...")

            if eeg_segments.get('metadatas') and eeg_segments['metadatas'][0]:
                print("\n🏷️  FIRST METADATA:")
                first_meta = eeg_segments['metadatas'][0][0]
                print(f"  {json.dumps(first_meta, indent=2)}")
        else:
            print("\n⚠️  NO EEG SEGMENTS RETRIEVED!")

        # Show first medical definition
        if medical_defs.get('documents') and medical_defs['documents'][0]:
            print("\n📚 FIRST MEDICAL DEFINITION (preview):")
            first_med = medical_defs['documents'][0][0]
            print(f"  {first_med[:200]}...")
        else:
            print("\n⚠️  NO MEDICAL DEFINITIONS RETRIEVED!")

        # Show LLM response
        llm_response = data.get("llm_response", "No response")
        print("\n🤖 LLM RESPONSE:")
        print(f"  {llm_response}")

        # Show validation
        validation = data.get("validation", {})
        print("\n✅ VALIDATION:")
        print(f"  {json.dumps(validation, indent=2)}")

        return data

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return None


if __name__ == "__main__":
    # Test questions
    questions = [
        "What is LPD?",
        "Tell me about patient 42516",
        "Show me seizure segments"
    ]

    for q in questions:
        result = debug_question(q)
        input("\nPress Enter to continue...")