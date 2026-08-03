import chromadb
import json
import sys

# --- SETTINGS ---
# Connection settings from docker-compose.yml
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

# Which collection do you want to dump?
# Options: "signal", "document", "transcript"
COLLECTION_NAME = "signal"

# Output file name
OUTPUT_FILE = f"dump_{COLLECTION_NAME}.jsonl"


# -------------

def dump_collection():
    """Fetches all contents of a collection from ChromaDB."""

    print(f"Connecting to ChromaDB at http://{CHROMA_HOST}:{CHROMA_PORT}...")
    try:
        # Same method used in app.py
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()  # Test the connection
        print("Connection successful.")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Error: Make sure your Docker stack (docker-compose up) is running.")
        sys.exit(1)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        if count == 0:
            print(f"Collection '{COLLECTION_NAME}' is empty.")
            return

        print(f"Found collection '{COLLECTION_NAME}' with {count} items.")
    except Exception as e:
        print(f"Failed to get collection '{COLLECTION_NAME}': {e}")
        sys.exit(1)

    print(f"Fetching all {count} items (this may take a moment)...")
    try:
        # Similar to what is done in search_indices.py,
        # we request everything including 'documents' and 'metadatas'.
        # Use 'limit=count' to fetch all data.
        data = collection.get(
            limit=count,
            include=['documents', 'metadatas']
        )
        print(f"Successfully fetched {len(data['ids'])} items.")

    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)

    print(f"Saving data to {OUTPUT_FILE}...")
    saved_count = 0
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # Write each record (id, document, metadata) as a JSON line
            for i in range(len(data['ids'])):
                item = {
                    "id": data['ids'][i],
                    # 'documents' is a list, we get each element
                    "document_text": data['documents'][i],
                    # 'metadatas' is also a list
                    "metadata": data['metadatas'][i] if data['metadatas'] else {}
                }
                # JSONL format (each line is a JSON object)
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                saved_count += 1

    except Exception as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)

    print(f"\n✅ Done. Saved {saved_count} items to {OUTPUT_FILE}.")
    print(f"You can open this file and use the ('document_text') field to build ideal answers.")


if __name__ == "__main__":
    # The 'chromadb' library must be installed before running:
    # pip install chromadb

    # Check if a collection name argument was provided
    if len(sys.argv) > 1:
        # If an argument is provided (e.g., "document_chunks"),
        # assign it DIRECTLY as COLLECTION_NAME.
        COLLECTION_NAME = sys.argv[1]
        print(f"Target collection set to: {COLLECTION_NAME}")
    else:
        # If no argument is provided, show usage instructions
        print("Error: Please specify a collection name.")
        print("Example: python test/ext.py document_chunks")
        sys.exit(1)  # Exit with error

    # Set the output file name dynamically
    OUTPUT_FILE = f"dump_{COLLECTION_NAME}.jsonl"

    dump_collection()  # Run the main function