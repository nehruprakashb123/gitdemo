from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading embedding model...\n")

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Test text
text = "what is kubernetes?"

# Generate embedding
embedding = model.encode([text])[0]

print("✅ Embedding generated successfully!")
print("=" * 60)
print("Embedding shape     :", embedding.shape)
print("Vector length       :", len(embedding))
print("Data type           :", type(embedding))
print("=" * 60)

print("\nFirst 20 values:")
print(embedding[:20].tolist())

print("\nLast 20 values:")
print(embedding[-20:].tolist())

print("\nFull vector (384 numbers):")
print(list(embedding))