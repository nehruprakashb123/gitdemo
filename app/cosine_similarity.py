from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Two sentences
sent1 = "what is kubernetes?"
sent2 = "explain kubernetes container orchestration"

# Generate embeddings
emb1 = model.encode([sent1])[0]
emb2 = model.encode([sent2])[0]

# Manual Cosine Similarity Formula
dot_product = np.dot(emb1, emb2)
norm1 = norm(emb1)
norm2 = norm(emb2)
cosine_similarity = dot_product / (norm1 * norm2)

print("Sentence 1 :", sent1)
print("Sentence 2 :", sent2)
print("="*60)
print("Cosine Similarity :", round(float(cosine_similarity), 4))
print("Similarity (%)    :", round(float(cosine_similarity) * 100, 2), "%")
print("\nVector 1 length :", len(emb1))
print("Vector 2 length :", len(emb2))