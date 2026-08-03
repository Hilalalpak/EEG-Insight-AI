import chromadb
import sys

# Connection settings from docker-compose.yml
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

print(f"Connecting to ChromaDB at http://{CHROMA_HOST}:{CHROMA_PORT}...")
try:
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    client.heartbeat()
    print("Connection successful.")
except Exception as e:
    print(f"Connection failed: {e}")
    print("Error: Make sure your Docker stack (docker-compose up) is running.")
    sys.exit(1)

try:
    collections = client.list_collections()

    if not collections:
        print("\n❌ Error: No collections found in the database.")
        print("   If you believe the database should have data, this is unexpected.")
        print("   Please try running 'docker-compose run --rm ingest-document' again.")
    else:
        print("\n✅ Collections found in the database:")

        # Print collection names and their item counts
        for c in collections:
            print(f"  - Name: '{c.name}' | Items: {c.count()}")

        print("\nPlease run the 'test/ext.py' (dump) script with these names.")
        print("Example: if you see 'eegi_documents' instead of 'document',")
        print(f"Command: python test/ext.py eegi_documents")

except Exception as e:
    print(f"\n❌ Error while listing collections: {e}")