#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
List and delete ChromaDB collections.
Usage:
  python manage_chroma.py list
  python manage_chroma.py delete <collection_name>
"""

import sys
import chromadb

HOST = "localhost"   # use "eegi-chroma" if running inside Docker
PORT = 8000


def list_collections(client):
    """Print available vector collections."""
    cols = client.list_collections()
    if not cols:
        print("⚠️  No collections found.")
        return
    print("📦 Collections:")
    for c in cols:
        print(f" - {c.name}")


def delete_collection(client, name):
    """Delete a specific collection."""
    try:
        client.delete_collection(name=name)
        print(f"🗑️  Deleted collection: {name}")
    except Exception as e:
        print(f"❌ Error deleting {name}: {e}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    client = chromadb.HttpClient(host=HOST, port=PORT)
    command = sys.argv[1].lower()

    if command == "list":
        list_collections(client)
    elif command == "delete":
        if len(sys.argv) < 3:
            print("Please provide a collection name.")
            sys.exit(1)
        delete_collection(client, sys.argv[2])
    else:
        print("Unknown command. Use: list | delete <name>")


if __name__ == "__main__":
    main()
