"""
MCP (Model Context Protocol) module for Lab 2 NLP.
"""

from src.mcp.client import MCPClient
from src.mcp.tools import (
    calculate_credit_score,
    assess_risk,
    retrieve_similar_cases,
    CreditScoringTool,
    RiskAssessmentTool,
    RAGRetrieverTool
)

__all__ = [
    "MCPClient",
    "calculate_credit_score",
    "assess_risk",
    "retrieve_similar_cases",
    "CreditScoringTool",
    "RiskAssessmentTool",
    "RAGRetrieverTool"
]
