"""
Inference script for Lab 2 NLP.

This script provides examples for running inference with:
- LLM (Ollama)
- MCP tools
- RAG retrieval
- ML models
"""

import argparse
from typing import Optional

from src.llm.ollama_client import OllamaClient
from src.mcp.client import MCPClient
from src.mcp.tools import calculate_credit_score, assess_risk
from src.rag.retriever import RAGRetriever
from src.ml.credit_scoring import get_model
from src.config import PromptingConfig


def run_llm_inference(
    prompt: str,
    technique: str = "zero_shot",
    system: Optional[str] = None
) -> str:
    """
    Run LLM inference with various prompting techniques.
    
    Args:
        prompt: Input prompt
        technique: Prompting technique (zero_shot, cot, few_shot, cot_few_shot)
        system: System prompt
    
    Returns:
        Generated response
    """
    client = OllamaClient()
    
    if not client.check_connection():
        print("Warning: Ollama service not available. Using mock response.")
        return f"Mock response for: {prompt}"
    
    if technique == "zero_shot":
        result = client.zero_shot(
            prompt=prompt,
            task_description=PromptingConfig.SYSTEM_PROMPTS["zero_shot"]
        )
    elif technique == "cot":
        result = client.chain_of_thought(
            problem=prompt,
            system_prompt=PromptingConfig.SYSTEM_PROMPTS["cot"]
        )
    elif technique == "few_shot":
        examples = [
            {"input": "Client: High income, married, bachelor's degree", "output": "Approved"},
            {"input": "Client: Low income, single, HS-grad", "output": "Rejected"}
        ]
        result = client.few_shot(prompt=prompt, examples=examples)
    elif technique == "cot_few_shot":
        examples = [
            {
                "input": "Client: High income, married, bachelor's degree",
                "reasoning": "1) Age 35 - prime working age, 2) Income >50K - above threshold",
                "output": "Approved (1)"
            }
        ]
        result = client.cot_few_shot(prompt=prompt, examples=examples)
    else:
        result = client.generate(prompt=prompt, system=system)
    
    return result.get("response", "No response")


def run_mcp_tools_example():
    """Run examples of MCP tools."""
    print("\n" + "="*50)
    print("MCP TOOLS EXAMPLES")
    print("="*50)
    
    # Credit score calculation
    print("\n1. Credit Score Calculation:")
    credit_result = calculate_credit_score(
        age=35,
        income=75000,
        employment_years=10,
        education_level="Bachelor's",
        has_credit_card=True,
        has_mortgage=True
    )
    print(f"   Score: {credit_result['score']}")
    print(f"   Grade: {credit_result['grade']}")
    
    # Risk assessment
    print("\n2. Risk Assessment:")
    risk_result = assess_risk(
        age=35,
        marital_status="Married",
        education="Bachelor's",
        occupation="Tech",
        capital_gain=5000,
        hours_per_week=45
    )
    print(f"   Risk Level: {risk_result['risk_level']}")
    print(f"   Recommendation: {risk_result['recommendation']}")


def run_rag_example():
    """Run RAG retrieval example."""
    print("\n" + "="*50)
    print("RAG RETRIEVAL EXAMPLE")
    print("="*50)
    
    retriever = RAGRetriever()
    
    query = "35 year old married professional with high income"
    print(f"\nQuery: {query}")
    
    # Note: This requires the dataset to be indexed first
    # results = retriever.retrieve(query, top_k=3)
    # print(f"Found {len(results)} similar cases")
    
    print("Note: Run index_dataset first to enable retrieval")


def run_ml_example():
    """Run ML model example."""
    print("\n" + "="*50)
    print("ML MODEL EXAMPLE")
    print("="*50)
    
    # Example with mock data
    print("\nNote: Train models first using train.py for actual predictions")
    print("Mock prediction example:")
    print("   Input features would be processed through trained model")
    print("   Output: Credit decision (approve/reject)")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Lab 2 NLP Inference")
    parser.add_argument("--input", type=str, help="Input text for LLM")
    parser.add_argument("--technique", type=str, default="zero_shot",
                       choices=["zero_shot", "cot", "few_shot", "cot_few_shot"],
                       help="Prompting technique")
    parser.add_argument("--example", type=str, default="all",
                       choices=["llm", "mcp", "rag", "ml", "all"],
                       help="Example to run")
    
    args = parser.parse_args()
    
    print("Lab 2 NLP - Credit Scoring Inference")
    print("="*50)
    
    if args.example in ["llm", "all"]:
        if args.input:
            print(f"\nRunning LLM inference...")
            response = run_llm_inference(args.input, args.technique)
            print(f"Response: {response}")
        else:
            print("\nLLM Example (mock):")
            response = run_llm_inference(
                "Should we approve credit for a client with income 75000, age 35, married?",
                "cot"
            )
            print(f"Response: {response[:200]}...")
    
    if args.example in ["mcp", "all"]:
        run_mcp_tools_example()
    
    if args.example in ["rag", "all"]:
        run_rag_example()
    
    if args.example in ["ml", "all"]:
        run_ml_example()


if __name__ == "__main__":
    main()
