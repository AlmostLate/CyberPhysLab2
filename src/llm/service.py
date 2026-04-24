"""
FastAPI service for LLM inference.

This module provides a FastAPI wrapper around the Ollama LLM service,
exposing endpoints for various prompting techniques including:
- Zero-shot
- Chain-of-thought (CoT)
- Few-shot
- CoT + Few-shot
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn

from src.config import (
    LLMServiceConfig,
    PromptingConfig,
    API,
    get_ollama_url
)
from src.llm.ollama_client import OllamaClient


# =============================================================================
# PYDANTIC MODELS
# =============================================================================
class GenerateRequest(BaseModel):
    """Request model for text generation."""
    
    prompt: str = Field(..., description="Input prompt for generation")
    system: Optional[str] = Field(None, description="System prompt")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    max_tokens: int = Field(256, ge=1, le=4096, description="Maximum tokens to generate")


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    
    response: str
    model: str
    done: bool


class ChatMessage(BaseModel):
    """Chat message model."""
    
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat."""
    
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    """Response model for chat."""
    
    response: str
    model: str
    done: bool


class ZeroShotRequest(BaseModel):
    """Request model for zero-shot learning."""
    
    prompt: str = Field(..., description="Input prompt")
    task: str = Field(..., description="Task description")


class CoTRequest(BaseModel):
    """Request model for chain-of-thought reasoning."""
    
    problem: str = Field(..., description="Problem to solve")


class FewShotRequest(BaseModel):
    """Request model for few-shot learning."""
    
    prompt: str = Field(..., description="Input prompt")
    examples: List[Dict[str, str]] = Field(..., description="Examples with input/output")


class CoTFewShotRequest(BaseModel):
    """Request model for CoT + few-shot."""
    
    prompt: str = Field(..., description="Input prompt")
    examples: List[Dict[str, str]] = Field(..., description="Examples with reasoning")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    ollama_connected: bool
    model: str


# =============================================================================
# FASTAPI APP
# =============================================================================
def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    app = FastAPI(
        title="LLM Service API",
        description="FastAPI wrapper for Ollama LLM service with multiple prompting techniques",
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
    
    # Initialize Ollama client
    client = OllamaClient()
    
    # =============================================================================
    # HEALTH CHECK
    # =============================================================================
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """
        Health check endpoint.
        
        Returns service status and Ollama connection status.
        """
        ollama_connected = client.check_connection()
        
        return HealthResponse(
            status="healthy" if ollama_connected else "degraded",
            ollama_connected=ollama_connected,
            model=client.model_name
        )
    
    # =============================================================================
    # BASIC GENERATION
    # =============================================================================
    @app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
    async def generate(request: GenerateRequest):
        """
        Basic text generation endpoint.
        
        Args:
            request: Generation request with prompt and parameters
        
        Returns:
            Generated text response
        """
        try:
            result = client.generate(
                prompt=request.prompt,
                system=request.system,
                temperature=request.temperature,
                top_p=request.top_p,
                num_predict=request.max_tokens
            )
            
            return GenerateResponse(
                response=result.get("response", ""),
                model=client.model_name,
                done=result.get("done", True)
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # =============================================================================
    # CHAT
    # =============================================================================
    @app.post("/chat", response_model=ChatResponse, tags=["Chat"])
    async def chat(request: ChatRequest):
        """
        Chat endpoint using conversation history.
        
        Args:
            request: Chat request with message history
        
        Returns:
            Chat response
        """
        try:
            messages = [msg.dict() for msg in request.messages]
            
            result = client.chat(
                messages=messages,
                temperature=request.temperature
            )
            
            # Extract response from the last message
            response_text = ""
            if "message" in result:
                response_text = result["message"].get("content", "")
            
            return ChatResponse(
                response=response_text,
                model=client.model_name,
                done=result.get("done", True)
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # =============================================================================
    # ZERO-SHOT
    # =============================================================================
    @app.post("/zero-shot", response_model=GenerateResponse, tags=["Prompting"])
    async def zero_shot(request: ZeroShotRequest):
        """
        Zero-shot learning endpoint.
        
        Args:
            request: Zero-shot request with prompt and task description
        
        Returns:
            Generated response
        """
        try:
            result = client.zero_shot(
                prompt=request.prompt,
                task_description=request.task
            )
            
            return GenerateResponse(
                response=result.get("response", ""),
                model=client.model_name,
                done=result.get("done", True)
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # =============================================================================
    # CHAIN-OF-THOUGHT
    # =============================================================================
    @app.post("/cot", response_model=GenerateResponse, tags=["Prompting"])
    async def chain_of_thought(request: CoTRequest):
        """
        Chain-of-thought reasoning endpoint.
        
        Args:
            request: CoT request with problem statement
        
        Returns:
            Generated response with reasoning steps
        """
        try:
            result = client.chain_of_thought(
                problem=request.problem,
                system_prompt=PromptingConfig.SYSTEM_PROMPTS["cot"]
            )
            
            return GenerateResponse(
                response=result.get("response", ""),
                model=client.model_name,
                done=result.get("done", True)
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # =============================================================================
    # FEW-SHOT
    # =============================================================================
    @app.post("/few-shot", response_model=GenerateResponse, tags=["Prompting"])
    async def few_shot(request: FewShotRequest):
        """
        Few-shot learning endpoint.
        
        Args:
            request: Few-shot request with examples
        
        Returns:
            Generated response
        """
        try:
            result = client.few_shot(
                prompt=request.prompt,
                examples=request.examples
            )
            
            return GenerateResponse(
                response=result.get("response", ""),
                model=client.model_name,
                done=result.get("done", True)
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # =============================================================================
    # COT + FEW-SHOT
    # =============================================================================
    @app.post("/cot-few-shot", response_model=GenerateResponse, tags=["Prompting"])
    async def cot_few_shot(request: CoTFewShotRequest):
        """
        Chain-of-thought with few-shot examples endpoint.
        
        Args:
            request: CoT + few-shot request with examples
        
        Returns:
            Generated response with reasoning
        """
        try:
            result = client.cot_few_shot(
                prompt=request.prompt,
                examples=request.examples
            )
            
            return GenerateResponse(
                response=result.get("response", ""),
                model=client.model_name,
                done=result.get("done", True)
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def run_service():
    """
    Run the LLM FastAPI service.
    """
    app = create_app()
    
    print(f"Starting LLM Service on {LLMServiceConfig.HOST}:{LLMServiceConfig.PORT}")
    print(f"Ollama URL: {LLMServiceConfig.HOST}")
    print(f"Model: {LLMServiceConfig.MODEL_NAME}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=LLMServiceConfig.PORT,
        reload=LLMServiceConfig.DEBUG
    )


if __name__ == "__main__":
    run_service()
