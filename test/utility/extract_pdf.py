import chromadb

def export_pdf_chunks(
    collection_name="medical_definitions",
    host="localhost",
    port=8000,
    output_file="data/chunks/pdf_chunks_definitions1.txt"):

    client = chromadb.HttpClient(host=host, port=port)
    collection = client.get_collection(collection_name)

    print(f"Fetching documents from collection '{collection_name}'...")
    data = collection.get(include=["documents", "metadatas"])

    chunks = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        chunks.append({
            "chunk_index": meta.get("chunk_index", -1),
            "page": meta.get("page", -1),
            "text": doc.strip()})

    chunks.sort(key=lambda x: x["chunk_index"])

    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"Chunk {i} (Page {chunk['page']}):\n")
            f.write(chunk["text"])
            f.write("\n\n")

    print(f"✅ Export complete. Saved to '{output_file}'")

if __name__ == "__main__":
    export_pdf_chunks()
