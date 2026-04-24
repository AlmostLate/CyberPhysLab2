"""
Ollama API client for Lab 2 NLP.

This module provides a client for interacting with the Ollama LLM service,
which hosts models like Qwen2.5:0.5B locally.
"""

import requests
from typing import Optional, Dict, Any, List
import json

from src.config import OllamaConfig, get_ollama_url


class OllamaClient:
    """
    Client for interacting with Ollama LLM service.
    
    This client wraps the Ollama API to provide easier access to LLM
    inference capabilities. It supports various prompting techniques
    including zero-shot, chain-of-thought, and few-shot learning.
    
    Attributes:
        base_url (str): Base URL for Ollama API
        model_name (str): Name of the model to use
        timeout (int): Request timeout in seconds
    
    Example:
        >>> client = OllamaClient()
        >>> response = client.generate("What is credit scoring?")
        >>> print(response["response"])
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 120
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url (str, optional): Base URL for Ollama API
            model_name (str, optional): Name of the model to use
            timeout (int): Request timeout in seconds
        """
        self.base_url = base_url or OllamaConfig.HOST
        self.model_name = model_name or OllamaConfig.MODEL_NAME
        self.timeout = timeout
    
    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Ollama API.
        
        Args:
            endpoint (str): API endpoint path
            method (str): HTTP method (GET, POST)
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
                f"Ollama API request failed: {str(e)}"
            )
    
    def check_connection(self) -> bool:
        """
        Check if Ollama service is running and accessible.
        
        Returns:
            bool: True if service is available, False otherwise
        """
        try:
            self._make_request("/api/tags")
            return True
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """
        List available models in Ollama.
        
        Returns:
            List[str]: List of available model names
        """
        try:
            response = self._make_request("/api/tags")
            return [model["name"] for model in response.get("models", [])]
        except Exception:
            return []
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_predict: int = 256
    ) -> Dict[str, Any]:
        """
        Generate text using the LLM.
        
        Args:
            prompt (str): The input prompt for generation
            system (str, optional): System prompt for instruction tuning
            temperature (float): Sampling temperature (0.0 to 1.0)
            top_p (float): Nucleus sampling parameter
            num_predict (int): Maximum number of tokens to predict
        
        Returns:
            dict: Response containing 'response' field with generated text
        
        Example:
            >>> client = OllamaClient()
            >>> result = client.generate(
            ...     prompt="Explain credit scoring",
            ...     system="You are a financial advisor"
            ... )
            >>> print(result["response"])
        """
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
            "stream": False
        }
        
        if system:
            data["system"] = system
        
        return self._make_request("/api/generate", method="POST", data=data)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        Generate chat response using conversation history.
        
        Args:
            messages (List[Dict[str, str]]): List of message dicts with 'role' and 'content'
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter
        
        Returns:
            dict: Response with generated message
        
        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant"},
            ...     {"role": "user", "content": "What is credit?"}
            ... ]
            >>> result = client.chat(messages)
        """
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False
        }
        
        return self._make_request("/api/chat", method="POST", data=data)
    
    def zero_shot(
        self,
        prompt: str,
        task_description: str
    ) -> Dict[str, Any]:
        """
        Perform zero-shot task.
        
        Args:
            prompt (str): The input prompt
            task_description (str): Description of the task
        
        Returns:
            dict: Generated response
        """
        full_prompt = f"{task_description}\n\nInput: {prompt}"
        return self.generate(prompt=full_prompt)
    
    def chain_of_thought(
        self,
        problem: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform chain-of-thought reasoning.
        
        Args:
            problem (str): The problem to solve
            system_prompt (str, optional): Custom system prompt
        
        Returns:
            dict: Generated response with reasoning steps
        """
        cot_system = system_prompt or (
            "Think step by step about the problem. "
            "Break down your reasoning into clear steps "
            "and then provide your final answer."
        )
        
        return self.generate(
            prompt=problem,
            system=cot_system,
            temperature=0.3  # Lower temperature for more deterministic reasoning
        )
    
    def few_shot(
        self,
        prompt: str,
        examples: List[Dict[str, str]],
        include_answer: bool = True
    ) -> Dict[str, Any]:
        """
        Perform few-shot learning with examples.
        
        Args:
            prompt (str): The input prompt to classify/analyze
            examples (List[Dict[str, str]]): List of example dicts with 'input' and 'output'
            include_answer (bool): Whether to include answer in output
        
        Returns:
            dict: Generated response
        """
        examples_text = "\n\n".join([
            f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
            for i, ex in enumerate(examples)
        ])
        
        full_prompt = f"{examples_text}\n\nNow analyze this:\nInput: {prompt}\nOutput:"
        
        return self.generate(prompt=full_prompt, temperature=0.5)
    
    def cot_few_shot(
        self,
        prompt: str,
        examples: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Perform chain-of-thought with few-shot examples.
        
        Args:
            prompt (str): The input prompt
            examples (List[Dict[str, str]]): List of examples with 'input' and 'reasoning' and 'output'
        
        Returns:
            dict: Generated response
        """
        examples_text = "\n\n".join([
            f"Example {i+1}:\nInput: {ex['input']}\n"
            f"Reasoning: {ex.get('reasoning', 'N/A')}\n"
            f"Output: {ex['output']}"
            for i, ex in enumerate(examples)
        ])
        
        full_prompt = (
            f"{examples_text}\n\n"
            f"Now think step by step about this:\n"
            f"Input: {prompt}\n"
            f"Reasoning:"
        )
        
        return self.generate(
            prompt=full_prompt,
            temperature=0.3,
            num_predict=512  # Allow longer responses for reasoning
        )


def get_default_client() -> OllamaClient:
    """
    Get a default Ollama client with configured settings.
    
    Returns:
        OllamaClient: Client with default configuration
    """
    return OllamaClient()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Example usage
    print("Testing Ollama client...")
    
    client = OllamaClient()
    
    if client.check_connection():
        print("✓ Ollama service is running")
        print(f"Available models: {client.list_models()}")
        
        # Test generation
        print("\nTesting text generation...")
        response = client.generate("What is credit scoring?")
        print(f"Response: {response.get('response', 'No response')[:200]}...")
    else:
        print("✗ Ollama service is not available")
        print("Please ensure Ollama is running:")
        print("  - Install: https://ollama.ai/")
        print("  - Run: ollama serve")
        print("  - Pull model: ollama pull qwen2.5:0.5b")
