from sentence_transformers import SentenceTransformer
import chromadb
import uuid

print("🚀 Seeding DevOps Knowledge Base into ChromaDB...\n")

# Initialize ChromaDB (embedded + persistent)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="devops_knowledge")

# Embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 10 DevOps Documents with Metadata
documents = [
    "Kubernetes is an open-source container orchestration platform that automates deployment, scaling, and management of containerized applications.",
    "To debug a CrashLoopBackOff pod: kubectl describe pod <pod-name> and kubectl logs <pod-name> --previous",
    "A good Kubernetes Deployment YAML should use labels, selectors, and resource requests/limits for production stability.",
    "Helm is the package manager for Kubernetes. Use 'helm install', 'helm upgrade', and 'helm template' for managing charts.",
    "Terraform is an Infrastructure as Code tool. Always use terraform plan before apply and maintain state files securely.",
    "ArgoCD is a declarative GitOps continuous delivery tool for Kubernetes. It watches Git repositories and automatically syncs changes.",
    "Dockerfile best practices: Use multi-stage builds, run as non-root user, and minimize image layers for security and size.",
    "CI/CD Pipeline best practice: Build → Test → Scan → Push Image → Deploy to Kubernetes using GitOps.",
    "kubectl get pods -o wide shows which node a pod is running on. Useful for debugging node-specific issues.",
    "Always use ResourceQuota and LimitRange in namespaces to prevent noisy neighbor problems in shared Kubernetes clusters."
]

metadatas = [
    {"source": "kubernetes_basics", "category": "orchestration", "type": "definition"},
    {"source": "troubleshooting", "category": "debugging", "type": "command"},
    {"source": "deployment_best_practices", "category": "yaml", "type": "config"},
    {"source": "helm", "category": "packaging", "type": "tool"},
    {"source": "terraform", "category": "iac", "type": "tool"},
    {"source": "argocd", "category": "gitops", "type": "tool"},
    {"source": "docker", "category": "container", "type": "best_practice"},
    {"source": "cicd", "category": "pipeline", "type": "best_practice"},
    {"source": "kubectl", "category": "debugging", "type": "command"},
    {"source": "resource_management", "category": "cluster", "type": "best_practice"}
]

# Generate embeddings and add to collection
print("Generating embeddings and adding 10 documents...")
embeddings = embedder.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=[str(uuid.uuid4()) for _ in range(len(documents))]
)

print("✅ Successfully seeded 10 DevOps documents!")
print(f"   Collection name : {collection.name}")
print(f"   Total documents : {collection.count()}")
print("\nYou can now query them using the /chat endpoint!")