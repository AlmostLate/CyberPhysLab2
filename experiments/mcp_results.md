# MCP Results - Lab 2 NLP

## Experiment Configuration

This document contains the results of MCP (Model Context Protocol) service experiments.

## MCP Tools Implemented

### 1. Credit Score Calculator
- **Purpose**: Calculate credit score based on client demographics
- **Inputs**: age, income, employment_years, education_level, has_credit_card, has_mortgage, has_loans
- **Outputs**: credit_score (300-850), grade, factor breakdown

### 2. Risk Assessment Tool
- **Purpose**: Assess credit risk based on metadata
- **Inputs**: age, marital_status, education, occupation, capital_gain, capital_loss, hours_per_week
- **Outputs**: risk_level (low/medium/high/very_high), risk_score, factors, recommendation

### 3. RAG Retriever Tool
- **Purpose**: Retrieve similar historical cases
- **Inputs**: query, top_k, similarity_threshold
- **Outputs**: List of similar cases with outcomes

## Service Architecture

```
LLM Service (FastAPI) -> MCP Client -> MCP Server (Tools)
```

## Results (TBD - Requires Running Services)

| Tool | Execution Time | Accuracy | Notes |
|------|---------------|----------|-------|
| Credit Score | TBD ms | N/A | Deterministic calculation |
| Risk Assessment | TBD ms | N/A | Heuristic-based |
| RAG Retrieval | TBD ms | TBD | Depends on index quality |

## Conclusions

MCP enables modular tool integration with LLM services, allowing for:
- Separation of concerns (ML vs reasoning)
- Reusable tools across different LLM applications
- Transparent decision-making process
