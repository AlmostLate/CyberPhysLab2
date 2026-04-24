"""
MCP Client implementation for credit scoring.

This module provides a client for interacting with the MCP server
to execute credit scoring tools.
"""

import requests
from typing import Dict, Any, List, Optional
import json

from src.config import MCPServiceConfig, get_mcp_service_url


class MCPClient:
    """
    Client for interacting with MCP server.
    
    This client provides methods to:
    - List available tools
    - Execute tools
    - Get health status
    
    Attributes:
        base_url (str): Base URL for MCP service
        timeout (int): Request timeout in seconds
    
    Example:
        >>> client = MCPClient()
        >>> tools = client.list_tools()
        >>> result = client.execute_tool("calculate_credit_score", {"age": 35, "income": 75000})
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize MCP client.
        
        Args:
            base_url (str, optional): Base URL for MCP service
            timeout (int): Request timeout in seconds
        """
        self.base_url = base_url or MCPServiceConfig.HOST
        self.timeout = timeout
    
    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to MCP API.
        
        Args:
            endpoint (str): API endpoint path
            method (str): HTTP method
            data (dict, optional): Request body data
        
        Returns:
            dict: Response JSON
        
        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=self.timeout)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(
                f"MCP API request failed: {str(e)}"
            )
    
    def check_connection(self) -> bool:
        """
        Check if MCP service is running and accessible.
        
        Returns:
            bool: True if service is available, False otherwise
        """
        try:
            self._make_request("/health")
            return True
        except Exception:
            return False
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get health status of MCP service.
        
        Returns:
            dict: Health status information
        """
        return self._make_request("/health")
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools on MCP server.
        
        Returns:
            List of tool definitions with schemas
        """
        try:
            response = self._make_request("/tools")
            return response.get("tools", [])
        except Exception:
            return []
    
    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool on MCP server.
        
        Args:
            tool_name (str): Name of the tool to execute
            arguments (dict): Tool arguments
        
        Returns:
            dict: Tool execution result
        
        Example:
            >>> client = MCPClient()
            >>> result = client.execute_tool(
            ...     "calculate_credit_score",
            ...     {"age": 35, "income": 75000, "employment_years": 10}
            ... )
            >>> print(result["score"])  # e.g., 720
        """
        data = {
            "tool_name": tool_name,
            "arguments": arguments
        }
        
        return self._make_request("/tools/execute", method="POST", data=data)
    
    def calculate_credit_score(
        self,
        age: int,
        income: float,
        employment_years: int,
        education_level: str,
        has_credit_card: bool = True,
        has_mortgage: bool = False,
        has_loans: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate credit score using MCP tool.
        
        Args:
            age: Client's age
            income: Annual income
            employment_years: Years of employment
            education_level: Education level
            has_credit_card: Has credit card
            has_mortgage: Has mortgage
            has_loans: Has other loans
        
        Returns:
            dict: Credit score result
        """
        return self.execute_tool(
            "calculate_credit_score",
            {
                "age": age,
                "income": income,
                "employment_years": employment_years,
                "education_level": education_level,
                "has_credit_card": has_credit_card,
                "has_mortgage": has_mortgage,
                "has_loans": has_loans
            }
        )
    
    def assess_risk(
        self,
        age: int,
        marital_status: str,
        education: str,
        occupation: str,
        capital_gain: float = 0,
        capital_loss: float = 0,
        hours_per_week: int = 40
    ) -> Dict[str, Any]:
        """
        Assess credit risk using MCP tool.
        
        Args:
            age: Client's age
            marital_status: Marital status
            education: Education level
            occupation: Occupation
            capital_gain: Capital gains
            capital_loss: Capital losses
            hours_per_week: Working hours per week
        
        Returns:
            dict: Risk assessment result
        """
        return self.execute_tool(
            "assess_risk",
            {
                "age": age,
                "marital_status": marital_status,
                "education": education,
                "occupation": occupation,
                "capital_gain": capital_gain,
                "capital_loss": capital_loss,
                "hours_per_week": hours_per_week
            }
        )
    
    def retrieve_similar_cases(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Retrieve similar cases using RAG.
        
        Args:
            query: Query text
            top_k: Number of results
            similarity_threshold: Minimum similarity
        
        Returns:
            dict: Retrieved similar cases
        """
        return self.execute_tool(
            "retrieve_similar_cases",
            {
                "query": query,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold
            }
        )


def get_default_client() -> MCPClient:
    """
    Get a default MCP client with configured settings.
    
    Returns:
        MCPClient: Client with default configuration
    """
    return MCPClient()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Example usage
    print("Testing MCP client...")
    
    client = MCPClient()
    
    if client.check_connection():
        print("✓ MCP service is running")
        print(f"Available tools: {client.list_tools()}")
        
        # Test credit score calculation
        print("\nTesting credit score calculation...")
        result = client.calculate_credit_score(
            age=35,
            income=75000,
            employment_years=10,
            education_level="Bachelor's"
        )
        print(f"Credit Score Result: {result}")
    else:
        print("✗ MCP service is not available")
        print("Please ensure MCP server is running:")
        print("  python src/mcp/server.py")
