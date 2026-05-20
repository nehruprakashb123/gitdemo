# DevOps AI Copilot

A lightweight, fast **multi-turn AI assistant** specialized in DevOps and Infrastructure.

Powered by **Llama 3.1 8B** via Groq + FastAPI.

---

## Features

- **Full conversation history** — context is maintained across turns
- Real-time chat with streaming support
- Specialized in Kubernetes, Docker, Terraform, CI/CD, Helm, ArgoCD, and Cloud platforms
- Session-based conversations (multiple independent chats)
- View, clear, or delete conversation history
- Fully Dockerized and ready for GitHub Container Registry (GHCR)

---

## Tech Stack

- **FastAPI** – High-performance backend
- **Groq** – Ultra-fast inference (Llama 3.1 8B)
- **Python 3.11**
- **Docker** + **docker-compose**

---

## Quick Start

```bash
# 1. Clone & run locally
git clone <your-repo>
cd devops-ai-copilot

# 2. Add your API key
echo "GROQ_API_KEY=gsk_..." > .env

# 3. Run with Docker
docker compose up --build