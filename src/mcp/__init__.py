"""
MCP (Model Context Protocol) module for Lab 2 NLP.

This module provides MCP server and client implementations for
credit scoring analysis with tools like:
- Credit score calculation
- Risk assessment
- RAG retrieval
"""

from src.mcp.server import MCPServer, create_mcp_app
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
    "MCPServer",
    "create_mcp_app",
    "MCPClient",
    "calculate_credit_score",
    "assess_risk",
    "retrieve_similar_cases",
    "CreditScoringTool",
    "RiskAssessmentTool",
    "RAGRetrieverTool"
]
