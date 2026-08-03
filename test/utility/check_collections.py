# test/utility/check_collections.py
import chromadb

client = chromadb.HttpClient(host="localhost", port=8000)
collections = ["signal_segments", "document_chunks", "transcript_chunks", "eeg_insights", "medical_definitions", "video_reasoning"]

for name in collections:
    try:
        col = client.get_collection(name)
        count = col.count()
        print(f"✓ {name:25s}: {count:5d} docs")
    except Exception as e:
        print(f"✗ {name:25s}: Not found - {e}")