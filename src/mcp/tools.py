"""
MCP Tools for credit scoring analysis.

This module provides tool implementations for the MCP server:
- Credit score calculation
- Risk assessment based on metadata
- RAG retrieval for similar cases
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import json


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================
@dataclass
class Tool:
    """Base class for MCP tools."""
    
    name: str
    description: str
    input_schema: Dict[str, Any]
    function: Callable
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "function": self.function
        }


# =============================================================================
# CREDIT SCORING TOOL
# =============================================================================
def calculate_credit_score(
    age: int,
    income: float,
    employment_years: int,
    education_level: str,
    has_credit_card: bool = True,
    has_mortgage: bool = False,
    has_loans: bool = False
) -> Dict[str, Any]:
    """
    Calculate credit score based on client characteristics.
    
    This function implements a simplified version of credit scoring
    based on various factors like age, income, employment, etc.
    
    Args:
        age (int): Client's age in years
        income (float): Annual income in dollars
        employment_years (int): Years of employment
        education_level (str): Education level (High School, Bachelor's, Master's, etc.)
        has_credit_card (bool): Whether client has a credit card
        has_mortgage (bool): Whether client has a mortgage
        has_loans (bool): Whether client has other loans
    
    Returns:
        dict: Credit score and breakdown
    
    Example:
        >>> result = calculate_credit_score(
        ...     age=35,
        ...     income=75000,
        ...     employment_years=10,
        ...     education_level="Bachelor's"
        ... )
        >>> print(result["score"])  # e.g., 720
    """
    score = 0
    factors = {}
    
    # Age factor (optimal age range: 35-55)
    if 35 <= age <= 55:
        age_score = 100
    elif 25 <= age < 35:
        age_score = 70 + (age - 25) * 3
    elif 55 < age <= 65:
        age_score = 100 - (age - 55) * 3
    else:
        age_score = max(30, min(60, age * 2))
    score += age_score
    factors["age"] = {"value": age, "score": age_score}
    
    # Income factor (normalized)
    if income < 20000:
        income_score = 50
    elif income < 40000:
        income_score = 70
    elif income < 60000:
        income_score = 85
    elif income < 100000:
        income_score = 100
    else:
        income_score = min(120, 100 + (income - 100000) / 10000)
    score += income_score
    factors["income"] = {"value": income, "score": income_score}
    
    # Employment factor
    employment_score = min(80, employment_years * 8)
    score += employment_score
    factors["employment"] = {"value": employment_years, "score": employment_score}
    
    # Education factor
    education_scores = {
        "Preschool": 20,
        "High School": 50,
        "Bachelor's": 80,
        "Master's": 90,
        "Doctorate": 100
    }
    education_score = education_scores.get(education_level, 50)
    score += education_score
    factors["education"] = {"value": education_level, "score": education_score}
    
    # Credit history factors
    credit_history_score = 0
    if has_credit_card:
        credit_history_score += 50
    if has_mortgage:
        credit_history_score += 30
    if has_loans:
        credit_history_score += 20
    score += credit_history_score
    factors["credit_history"] = {
        "has_credit_card": has_credit_card,
        "has_mortgage": has_mortgage,
        "has_loans": has_loans,
        "score": credit_history_score
    }
    
    # Normalize score to 300-850 range
    normalized_score = max(300, min(850, int(score * 2.5)))
    
    # Determine grade
    if normalized_score >= 800:
        grade = "Excellent"
    elif normalized_score >= 700:
        grade = "Good"
    elif normalized_score >= 650:
        grade = "Fair"
    elif normalized_score >= 550:
        grade = "Poor"
    else:
        grade = "Very Poor"
    
    return {
        "score": normalized_score,
        "grade": grade,
        "factors": factors,
        "max_score": 850,
        "min_score": 300
    }


class CreditScoringTool(Tool):
    """Tool for calculating credit scores."""
    
    def __init__(self):
        super().__init__(
            name="calculate_credit_score",
            description=(
                "Calculate credit score based on client demographics and financial information. "
                "Returns credit score (300-850), grade, and factor breakdown."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "age": {"type": "integer", "description": "Client's age in years"},
                    "income": {"type": "number", "description": "Annual income in dollars"},
                    "employment_years": {"type": "integer", "description": "Years of employment"},
                    "education_level": {
                        "type": "string",
                        "enum": ["Preschool", "High School", "Bachelor's", "Master's", "Doctorate"],
                        "description": "Education level"
                    },
                    "has_credit_card": {"type": "boolean", "description": "Has credit card"},
                    "has_mortgage": {"type": "boolean", "description": "Has mortgage"},
                    "has_loans": {"type": "boolean", "description": "Has other loans"}
                },
                "required": ["age", "income", "employment_years", "education_level"]
            },
            function=calculate_credit_score
        )


# =============================================================================
# RISK ASSESSMENT TOOL
# =============================================================================
def assess_risk(
    age: int,
    marital_status: str,
    education: str,
    occupation: str,
    capital_gain: float = 0,
    capital_loss: float = 0,
    hours_per_week: int = 40
) -> Dict[str, Any]:
    """
    Assess credit risk based on demographic and employment metadata.
    
    Args:
        age (int): Client's age
        marital_status (str): Marital status
        education (str): Education level
        occupation (str): Occupation type
        capital_gain (float): Capital gains
        capital_loss (float): Capital losses
        hours_per_week (int): Working hours per week
    
    Returns:
        dict: Risk assessment with level and factors
    
    Example:
        >>> result = assess_risk(
        ...     age=35,
        ...     marital_status="Married",
        ...     education="Bachelor's",
        ...     occupation="Tech"
        ... )
        >>> print(result["risk_level"])  # e.g., "low"
    """
    risk_score = 0
    factors = {}
    
    # Age risk
    if 25 <= age <= 55:
        age_risk = 0  # Low risk
    elif 18 <= age < 25:
        age_risk = 20  # Higher risk - young
    elif 55 < age <= 65:
        age_risk = 10  # Slight higher risk
    else:
        age_risk = 30  # Higher risk
    risk_score += age_risk
    factors["age"] = age_risk
    
    # Marital status risk
    marital_risk_scores = {
        "Married": 0,
        "Divorced": 10,
        "Separated": 15,
        "Widowed": 20,
        "Never Married": 15
    }
    marital_risk = marital_risk_scores.get(marital_status, 10)
    risk_score += marital_risk
    factors["marital_status"] = marital_risk
    
    # Education risk
    education_risk_scores = {
        "Doctorate": 0,
        "Master's": 5,
        "Bachelor's": 10,
        "HS-grad": 20,
        "Some-college": 25,
        "Less than HS": 35
    }
    education_risk = education_risk_scores.get(education, 20)
    risk_score += education_risk
    factors["education"] = education_risk
    
    # Occupation risk
    occupation_risk_scores = {
        "Tech": 5,
        "Finance": 5,
        "Healthcare": 10,
        "Education": 10,
        "Retail": 20,
        "Manual": 25,
        "Service": 20
    }
    occupation_risk = occupation_risk_scores.get(occupation, 15)
    risk_score += occupation_risk
    factors["occupation"] = occupation_risk
    
    # Capital gains/losses
    net_capital = capital_gain - capital_loss
    if net_capital > 10000:
        capital_risk = -20  # Reduces risk
    elif net_capital > 0:
        capital_risk = -10
    elif net_capital < -5000:
        capital_risk = 25
    else:
        capital_risk = 0
    risk_score += capital_risk
    factors["capital"] = capital_risk
    
    # Working hours risk
    if hours_per_week >= 40:
        hours_risk = 0
    elif hours_per_week >= 30:
        hours_risk = 10
    else:
        hours_risk = 20
    risk_score += hours_risk
    factors["hours_per_week"] = hours_risk
    
    # Determine risk level
    if risk_score <= 15:
        risk_level = "low"
    elif risk_score <= 35:
        risk_level = "medium"
    elif risk_score <= 55:
        risk_level = "high"
    else:
        risk_level = "very_high"
    
    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "factors": factors,
        "recommendation": _get_recommendation(risk_level)
    }


def _get_recommendation(risk_level: str) -> str:
    """Get recommendation based on risk level."""
    recommendations = {
        "low": "Approve with standard terms. Client shows stable profile.",
        "medium": "Approve with slightly higher interest rate. Consider additional verification.",
        "high": "Require additional documentation. Higher interest rate recommended.",
        "very_high": "Decline or require co-signer. High probability of default."
    }
    return recommendations.get(risk_level, "Review manually.")


class RiskAssessmentTool(Tool):
    """Tool for assessing credit risk."""
    
    def __init__(self):
        super().__init__(
            name="assess_risk",
            description=(
                "Assess credit risk based on demographic and employment metadata. "
                "Returns risk level (low/medium/high/very_high) and recommendation."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "age": {"type": "integer"},
                    "marital_status": {
                        "type": "string",
                        "enum": ["Married", "Divorced", "Separated", "Widowed", "Never Married"]
                    },
                    "education": {
                        "type": "string",
                        "enum": ["Doctorate", "Master's", "Bachelor's", "HS-grad", "Some-college", "Less than HS"]
                    },
                    "occupation": {
                        "type": "string",
                        "enum": ["Tech", "Finance", "Healthcare", "Education", "Retail", "Manual", "Service"]
                    },
                    "capital_gain": {"type": "number"},
                    "capital_loss": {"type": "number"},
                    "hours_per_week": {"type": "integer"}
                },
                "required": ["age", "marital_status", "education", "occupation"]
            },
            function=assess_risk
        )


# =============================================================================
# RAG RETRIEVER TOOL (Placeholder - actual implementation in rag module)
# =============================================================================
def retrieve_similar_cases(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Retrieve similar historical cases using RAG.
    
    Args:
        query (str): Query text describing the client
        top_k (int): Number of similar cases to retrieve
        similarity_threshold (float): Minimum similarity score
    
    Returns:
        dict: Retrieved similar cases with similarity scores
    """
    # This is a placeholder - actual implementation uses RAG module
    return {
        "query": query,
        "results": [],
        "message": "RAG retriever not initialized. Initialize with rag module."
    }


class RAGRetrieverTool(Tool):
    """Tool for retrieving similar cases using RAG."""
    
    def __init__(self):
        super().__init__(
            name="retrieve_similar_cases",
            description=(
                "Retrieve similar historical credit cases using RAG. "
                "Returns similar cases with their outcomes and similarity scores."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Client description query"},
                    "top_k": {"type": "integer", "description": "Number of results to return"},
                    "similarity_threshold": {"type": "number", "description": "Minimum similarity score"}
                },
                "required": ["query"]
            },
            function=retrieve_similar_cases
        )


# =============================================================================
# TOOL REGISTRY
# =============================================================================
class ToolRegistry:
    """Registry for all available MCP tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools."""
        self.register(CreditScoringTool())
        self.register(RiskAssessmentTool())
        self.register(RAGRetrieverTool())
    
    def register(self, tool: Tool):
        """Register a new tool."""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema
            }
            for tool in self.tools.values()
        ]
    
    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by name.
        
        Args:
            name (str): Tool name
            **kwargs: Tool arguments
        
        Returns:
            dict: Tool execution result
        """
        tool = self.get_tool(name)
        if tool is None:
            return {"error": f"Tool '{name}' not found"}
        
        try:
            result = tool.function(**kwargs)
            return result
        except Exception as e:
            return {"error": str(e)}


# Global tool registry
TOOL_REGISTRY = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return TOOL_REGISTRY
