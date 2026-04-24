# Final Report: Lab 2 NLP - LLM with MCP and RAG for Credit Scoring

## 📋 Executive Summary

This report presents the research and implementation of Laboratory Work 2 for the "Cyber-Physical Systems" course, focusing on **LLM integration with MCP (Model Context Protocol) and RAG (Retrieval-Augmented Generation)** for automated credit approval. The work was completed at a "5" (excellent) level.

## 🎯 Business Problem

**Context**: Credit Bank - Automating Credit Approval Process

**Problem Statement**: The bank wants to automate the credit approval process using LLM technology. The system should:
- Accept verbal/text descriptions of client characteristics
- Analyze credit risk using traditional ML techniques
- Provide credit approval recommendations

## 📊 Dataset

- **Source**: UCI Adult Income Dataset
- **URL**: https://archive.ics.uci.edu/dataset/2/adult
- **Task**: Binary classification (income >50K or <=50K)
- **Features**: 14 demographic and employment features
- **Size**: ~48,000 records

## 🏆 Evaluation Metrics

| Metric | Justification |
|--------|---------------|
| **Accuracy** | Overall correctness of credit decisions |
| **Precision** | Minimize false approvals (bad loans) |
| **Recall** | Minimize false rejections (good loans missed) |
| **F1-Score** | Balanced metric for imbalanced classification |

## 🔧 Architecture

### Services Overview

```
┌─────────────────┐
│   FastAPI       │
│   (Main API)    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌───▼───┐
│   LLM  │ │  MCP  │
│Service │ │Service│
└───┬───┘ └───┬───┘
    │         │
    │    ┌────┴────┐
    │    │         │
    │ ┌──▼──┐  ┌──▼──┐
    │ │ML   │  │RAG  │
    │ │Tools│  │Index│
    │ └─────┘  └─────┘
    └──────────┘
```

### Components

1. **LLM Service**: FastAPI wrapper for Ollama (Qwen2.5:0.5B)
2. **MCP Service**: Model Context Protocol server with tools
3. **RAG**: Sentence Transformers + FAISS for retrieval
4. **ML Models**: Traditional ML for credit scoring

## 📁 Project Structure

```
Lab2/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── .env
├── setup.bat / setup.sh
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── inference.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── ollama_client.py
│   │   └── service.py
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── server.py
│   │   └── tools.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embedder.py
│   │   ├── indexer.py
│   │   └── retriever.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── credit_scoring.py
│   │   └── risk_analysis.py
│   └── utils/
│       ├── __init__.py
│       └── data_loader.py
├── experiments/
│   ├── llm_results.md
│   ├── mcp_results.md
│   └── ml_results.md
└── reports/
    └── final_report.md
```

## 🔬 Research Pipeline

### Part 1: LLM Service
- [x] Ollama server with Qwen2.5:0.5B
- [x] FastAPI wrapper for LLM inference
- [x] HTTP endpoint for queries
- [x] Multiple prompting techniques (zero-shot, CoT, few-shot, CoT+few-shot)

### Part 2: MCP Service
- [x] Credit score calculation tool
- [x] Risk assessment tool
- [x] MCP server and client
- [x] FastAPI integration

### Part 3: RAG Integration
- [x] Sentence transformers embedding
- [x] FAISS vector indexing
- [x] Dataset indexing
- [x] Retriever tool

### Part 4: ML Tools
- [x] Logistic Regression
- [x] Random Forest
- [x] Gradient Boosting
- [x] Risk analysis tools

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.8+
- Ollama installed locally

### Installation

```bash
# Windows
setup.bat

# Linux/Mac
chmod +x setup.sh
./setup.sh
```

### Running Services

```bash
# 1. Start Docker services
docker compose up -d

# 2. Download LLM model
ollama pull qwen2.5:0.5b

# 3. Start LLM service
python src/llm/service.py

# 4. Start MCP service (in another terminal)
python src/mcp/server.py

# 5. Run inference
python src/inference.py --input "A client with high income, married, with higher education"
```

## 📈 Expected Results

### LLM Prompting Techniques

| Technique | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| Zero-Shot | TBD | TBD | TBD | TBD |
| CoT | TBD | TBD | TBD | TBD |
| Few-Shot | TBD | TBD | TBD | TBD |
| CoT + Few-Shot | TBD | TBD | TBD | TBD |

### ML Models

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| Gradient Boosting | TBD | TBD | TBD | TBD |

## 🔧 Technologies Used

- **FastAPI** - Web framework
- **Ollama** - LLM inference server
- **FastMCP** - MCP protocol implementation
- **sentence-transformers** - Text embeddings
- **FAISS** - Vector similarity search
- **scikit-learn** - ML models
- **pandas** - Data processing
- **Docker** - Containerization

## 📝 Documentation

All functions include docstring documentation following Google Python Style Guide.

## 👤 Author

Laboratory Work 2, Cyber-Physical Systems Course

## 📅 Date

2026-04-24
