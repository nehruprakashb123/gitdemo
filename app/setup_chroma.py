from sentence_transformers import SentenceTransformer
import chromadb
import os

print("🚀 Setting up ChromaDB locally (Embedded Mode)...\n")

# Initialize
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="devops_knowledge")

print(f"✅ ChromaDB ready!")
print(f"   Storage path : {os.path.abspath('./chroma_db')}")
print(f"   Collection   : {collection.name}")
print(f"   Current count: {collection.count()} documents")