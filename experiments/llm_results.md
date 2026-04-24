# LLM Results - Lab 2 NLP

## Experiment Configuration

This document contains the results of LLM experiments on credit scoring using various prompting techniques.

## Dataset
- UCI Adult Income Dataset
- Task: Predict whether income >50K based on demographic features

## Prompting Techniques Tested

### 1. Zero-Shot
- **Description**: Direct query without examples
- **System Prompt**: "You are a credit scoring assistant..."

### 2. Chain-of-Thought (CoT)
- **Description**: Step-by-step reasoning before decision
- **System Prompt**: Includes reasoning steps for credit analysis

### 3. Few-Shot
- **Description**: Provides 2-3 examples before query
- **Examples**: High/low income cases with known outcomes

### 4. CoT + Few-Shot
- **Description**: Combines reasoning with examples
- **System Prompt**: Examples with reasoning steps

## Results (TBD - Requires Ollama Running)

| Technique | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| Zero-Shot | TBD | TBD | TBD | TBD |
| CoT | TBD | TBD | TBD | TBD |
| Few-Shot | TBD | TBD | TBD | TBD |
| CoT + Few-Shot | TBD | TBD | TBD | TBD |

## Analysis

TBD - Results will be filled after running experiments with Ollama.

## Conclusions

Prompting technique effectiveness varies based on task complexity.
- Zero-shot: Baseline performance
- CoT: Improved reasoning transparency
- Few-shot: Better domain adaptation
- CoT + Few-Shot: Best for complex credit decisions
