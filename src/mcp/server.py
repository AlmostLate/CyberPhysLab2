"""
MCP Server implementation for credit scoring.

This module provides a FastAPI server that implements the MCP protocol
for credit scoring analysis with multiple tools.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import uvicorn

from src.config import MCPServiceConfig, API
from src.mcp.tools import (
    TOOL_REGISTRY,
    CreditScoringTool,
    RiskAssessmentTool,
    RAGRetrieverTool,
    calculate_credit_score,
    assess_risk,
    retrieve_similar_cases
)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================
class ToolRequest(BaseModel):
    """Request model for tool execution."""
    
    tool_name: str = Field(..., description="Name of the tool to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class ToolResponse(BaseModel):
    """Response model for tool execution."""
    
    tool_name: str
    result: Dict[str, Any]
    success: bool


class ToolListResponse(BaseModel):
    """Response model for listing tools."""
    
    tools: List[Dict[str, Any]]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    tools_count: int


# =============================================================================
# MCP SERVER
# =============================================================================
class MCPServer:
    """
    MCP Server for credit scoring analysis.
    
    This server provides access to various tools for credit analysis:
    - Credit score calculation
    - Risk assessment
    - RAG retrieval
    """
    
    def __init__(self):
        """Initialize MCP server with tool registry."""
        self.tool_registry = TOOL_REGISTRY
        self._register_tools()
    
    def _register_tools(self):
        """Register available tools."""
        self.tool_registry.register(CreditScoringTool())
        self.tool_registry.register(RiskAssessmentTool())
        self.tool_registry.register(RAGRetrieverTool())
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return self.tool_registry.list_tools()
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given arguments."""
        return self.tool_registry.execute_tool(tool_name, **arguments)


# =============================================================================
# FASTAPI APP
# =============================================================================
def create_mcp_app() -> FastAPI:
    """
    Create and configure FastAPI application for MCP server.
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    app = FastAPI(
        title="MCP Server for Credit Scoring",
        description="Model Context Protocol server with credit scoring tools",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=API.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=API.CORS_METHODS,
        allow_headers=API.CORS_HEADERS,
    )
    
    # Initialize MCP server
    mcp_server = MCPServer()
    
    # =============================================================================
    # HEALTH CHECK
    # =============================================================================
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            tools_count=len(mcp_server.list_tools())
        )
    
    # =============================================================================
    # LIST TOOLS
    # =============================================================================
    @app.get("/tools", response_model=ToolListResponse, tags=["Tools"])
    async def list_tools():
        """
        List all available MCP tools.
        
        Returns:
            List of available tools with their schemas
        """
        tools = mcp_server.list_tools()
        return ToolListResponse(tools=tools, count=len(tools))
    
    # =============================================================================
    # EXECUTE TOOL
    # =============================================================================
    @app.post("/tools/execute", response_model=ToolResponse, tags=["Tools"])
    async def execute_tool(request: ToolRequest):
        """
        Execute an MCP tool.
        
        Args:
            request: Tool execution request with tool name and arguments
        
        Returns:
            Tool execution result
        """
        try:
            result = mcp_server.execute_tool(
                request.tool_name,
                request.arguments
            )
            
            success = "error" not in result
            
            return ToolResponse(
                tool_name=request.tool_name,
                result=result,
                success=success
            )
        
        except Exception as e:
            return ToolResponse(
                tool_name=request.tool_name,
                result={"error": str(e)},
                success=False
            )
    
    # =============================================================================
    # CREDIT SCORING ENDPOINT
    # =============================================================================
    @app.post("/credit-score", tags=["Credit Scoring"])
    async def calculate_credit_score_endpoint(
        age: int,
        income: float,
        employment_years: int,
        education_level: str,
        has_credit_card: bool = True,
        has_mortgage: bool = False,
        has_loans: bool = False
    ):
        """
        Calculate credit score for a client.
        
        Args:
            age: Client's age
            income: Annual income
            employment_years: Years of employment
            education_level: Education level
            has_credit_card: Has credit card
            has_mortgage: Has mortgage
            has_loans: Has other loans
        
        Returns:
            Credit score result
        """
        result = calculate_credit_score(
            age=age,
            income=income,
            employment_years=employment_years,
            education_level=education_level,
            has_credit_card=has_credit_card,
            has_mortgage=has_mortgage,
            has_loans=has_loans
        )
        
        return result
    
    # =============================================================================
    # RISK ASSESSMENT ENDPOINT
    # =============================================================================
    @app.post("/risk-assessment", tags=["Risk Assessment"])
    async def assess_risk_endpoint(
        age: int,
        marital_status: str,
        education: str,
        occupation: str,
        capital_gain: float = 0,
        capital_loss: float = 0,
        hours_per_week: int = 40
    ):
        """
        Assess credit risk for a client.
        
        Args:
            age: Client's age
            marital_status: Marital status
            education: Education level
            occupation: Occupation type
            capital_gain: Capital gains
            capital_loss: Capital losses
            hours_per_week: Working hours per week
        
        Returns:
            Risk assessment result
        """
        result = assess_risk(
            age=age,
            marital_status=marital_status,
            education=education,
            occupation=occupation,
            capital_gain=capital_gain,
            capital_loss=capital_loss,
            hours_per_week=hours_per_week
        )
        
        return result
    
    # =============================================================================
    # RAG RETRIEVAL ENDPOINT
    # =============================================================================
    @app.post("/retrieve-similar", tags=["RAG Retrieval"])
    async def retrieve_similar_cases_endpoint(
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Retrieve similar historical cases using RAG.
        
        Args:
            query: Query text describing the client
            top_k: Number of similar cases to retrieve
            similarity_threshold: Minimum similarity score
        
        Returns:
            Retrieved similar cases
        """
        result = retrieve_similar_cases(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        return result
    
    # Store server instance for access by other modules
    app.state.mcp_server = mcp_server
    
    return app


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def run_server():
    """
    Run the MCP FastAPI server.
    """
    app = create_mcp_app()
    
    print(f"Starting MCP Server on {MCPServiceConfig.HOST}:{MCPServiceConfig.PORT}")
    print(f"Available tools: credit_score, risk_assessment, retrieve_similar")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=MCPServiceConfig.PORT,
        reload=MCPServiceConfig.DEBUG
    )


if __name__ == "__main__":
    run_server()
