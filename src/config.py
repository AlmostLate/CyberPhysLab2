"""
Configuration settings for Lab 2 NLP - LLM with MCP and RAG for Credit Scoring.

This module contains all configurable parameters for the project including:
- Ollama/LLM settings
- MCP service configuration
- RAG configuration
- ML model parameters
- Dataset settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, EXPERIMENTS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# OLLAMA / LLM CONFIGURATION
# =============================================================================
class OllamaConfig:
    """Configuration for Ollama LLM service."""
    
    HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
    TIMEOUT = 120  # seconds
    STREAM = False


# =============================================================================
# LLM SERVICE CONFIGURATION
# =============================================================================
class LLMServiceConfig:
    """Configuration for LLM FastAPI service."""
    
    HOST = os.getenv("LLM_SERVICE_URL", "http://localhost:8000")
    PORT = int(os.getenv("LLM_SERVICE_PORT", "8000"))
    API_PREFIX = "/api/v1"
    DEBUG = True


# =============================================================================
# MCP SERVICE CONFIGURATION
# =============================================================================
class MCPServiceConfig:
    """Configuration for MCP service."""
    
    HOST = os.getenv("MCP_SERVICE_URL", "http://localhost:8001")
    PORT = int(os.getenv("MCP_SERVICE_PORT", "8001"))
    API_PREFIX = "/api/v1/mcp"
    DEBUG = True


# =============================================================================
# RAG CONFIGURATION
# =============================================================================
class RAGConfig:
    """Configuration for RAG (Retrieval-Augmented Generation)."""
    
    # Embedding model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = 384  # for all-MiniLM-L6-v2
    
    # FAISS index
    INDEX_PATH = os.getenv("FAISS_INDEX_PATH", str(DATA_DIR / "faiss_index"))
    INDEX_TYPE = "Flat"  # Flat, IVF, HNSW
    
    # Retrieval settings
    TOP_K = 5  # Number of results to retrieve
    SIMILARITY_THRESHOLD = 0.7


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
class DatasetConfig:
    """Configuration for Adult Income dataset (Credit Scoring)."""
    
    NAME = "Adult Income Dataset"
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    PATH = str(DATA_DIR / "adult.csv")
    
    # Feature columns
    FEATURE_COLUMNS = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country"
    ]
    
    # Target column
    TARGET_COLUMN = "income"
    TARGET_VALUES = [" <=50K", " >50K"]  # Note: values have leading space
    
    # Categorical columns
    CATEGORICAL_COLUMNS = [
        "workclass", "education", "marital_status", "occupation",
        "relationship", "race", "sex", "native_country"
    ]
    
    # Numerical columns
    NUMERICAL_COLUMNS = [
        "age", "fnlwgt", "education_num", "capital_gain", 
        "capital_loss", "hours_per_week"
    ]
    
    # Test split
    TEST_SIZE = 0.2
    RANDOM_SEED = 42


# =============================================================================
# ML MODEL CONFIGURATION
# =============================================================================
class MLConfig:
    """Configuration for ML credit scoring models."""
    
    # Model types
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    
    # Model paths
    MODEL_PATH = str(MODELS_DIR)
    
    # Hyperparameters for Logistic Regression
    LOGISTIC_REGRESSION_PARAMS = {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
        "solver": "lbfgs"
    }
    
    # Hyperparameters for Random Forest
    RANDOM_FOREST_PARAMS = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    }
    
    # Hyperparameters for Gradient Boosting
    GRADIENT_BOOSTING_PARAMS = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "random_state": 42
    }
    
    # Preprocessing
    SCALE_NUMERICAL = True


# =============================================================================
# MCP TOOLS CONFIGURATION
# =============================================================================
class MCPToolsConfig:
    """Configuration for MCP tools."""
    
    # Credit scoring thresholds
    CREDIT_SCORE_MIN = 300
    CREDIT_SCORE_MAX = 850
    
    # Risk levels
    RISK_LEVELS = ["low", "medium", "high", "very_high"]
    
    # Age groups
    AGE_GROUPS = {
        "young": (18, 25),
        "adult": (26, 40),
        "middle_aged": (41, 55),
        "senior": (56, 100)
    }
    
    # Education levels (ordered)
    EDUCATION_LEVELS = [
        "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th",
        "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters",
        "Prof-school", "Doctorate"
    ]


# =============================================================================
# PROMPTING CONFIGURATIONS
# =============================================================================
class PromptingConfig:
    """Configuration for different prompting techniques."""
    
    # System prompts for different techniques
    SYSTEM_PROMPTS = {
        "zero_shot": "You are a credit scoring assistant. Analyze the client information and provide a credit decision.",
        
        "cot": """You are a credit scoring assistant. Think step by step about the client's profile:
1. Analyze the client's demographic information
2. Evaluate financial indicators
3. Consider employment status
4. Assess risk factors
5. Provide a final credit decision with reasoning.
Output format: {{"reasoning": "<your step-by-step analysis>", "verdict": <0 or 1>}}""",
        
        "few_shot": """You are a credit scoring assistant. Here are examples of credit decisions:

Example 1:
Client: High income, married, with bachelor's degree
Decision: Approved (1)
Reasoning: Strong financial indicators and stable family situation

Example 2:
Client: Low income, single, with HS-grad education
Decision: Rejected (0)
Reasoning: Limited financial capacity and unstable profile

Now analyze this client and provide a decision.""",
        
        "cot_few_shot": """You are a credit scoring assistant. Think step by step like in CoT, using similar examples.

Example:
Client: High income, married, with bachelor's degree
Reasoning: 1) Age 35 - prime working age, 2) Income >50K - above threshold, 3) Married - stable situation, 4) Bachelor's - higher education indicates responsibility
Verdict: 1

Now analyze this client:"""
    }
    
    # Output format for structured responses
    JSON_OUTPUT_FORMAT = {
        "reasoning": "string",
        "verdict": "0 or 1",
        "confidence": "0.0 to 1.0"
    }


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
class ExperimentConfig:
    """Configuration for experiments."""
    
    # Experiment names
    LLM_BASELINE = "llm_baseline"
    LLM_COT = "llm_cot"
    LLM_FEW_SHOT = "llm_few_shot"
    LLM_COT_FEW_SHOT = "llm_cot_few_shot"
    ML_MODELS = "ml_models"
    RAG_RETRIEVAL = "rag_retrieval"
    
    # Results files
    LLM_RESULTS_FILE = "llm_results.md"
    MCP_RESULTS_FILE = "mcp_results.md"
    ML_RESULTS_FILE = "ml_results.md"


# =============================================================================
# API CONFIGURATION
# =============================================================================
class APIConfig:
    """Configuration for FastAPI endpoints."""
    
    # CORS settings
    CORS_ORIGINS = ["*"]
    CORS_METHODS = ["*"]
    CORS_HEADERS = ["*"]
    
    # Rate limiting
    RATE_LIMIT = 100  # requests per minute
    RATE_LIMIT_PERIOD = 60  # seconds


# =============================================================================
# CONFIGURATION INSTANCES
# =============================================================================
OLLAMA = OllamaConfig()
LLM_SERVICE = LLMServiceConfig()
MCP_SERVICE = MCPServiceConfig()
RAG = RAGConfig()
DATASET = DatasetConfig()
ML = MLConfig()
MCP_TOOLS = MCPToolsConfig()
PROMPTING = PromptingConfig()
EXPERIMENT = ExperimentConfig()
API = APIConfig()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_ollama_url(endpoint: str = "") -> str:
    """
    Get full Ollama API URL.
    
    Args:
        endpoint (str): API endpoint path
    
    Returns:
        str: Full URL to Ollama endpoint
    """
    return f"{OLLAMA.HOST}/api/{endpoint}"


def get_llm_service_url(endpoint: str = "") -> str:
    """
    Get full LLM service URL.
    
    Args:
        endpoint (str): API endpoint path
    
    Returns:
        str: Full URL to LLM service endpoint
    """
    base = LLM_SERVICE.HOST.rstrip("/")
    prefix = LLM_SERVICE.API_PREFIX.lstrip("/")
    endpoint = endpoint.lstrip("/")
    
    if endpoint:
        return f"{base}/{prefix}/{endpoint}"
    return f"{base}/{prefix}"


def get_mcp_service_url(endpoint: str = "") -> str:
    """
    Get full MCP service URL.
    
    Args:
        endpoint (str): API endpoint path
    
    Returns:
        str: Full URL to MCP service endpoint
    """
    base = MCP_SERVICE.HOST.rstrip("/")
    prefix = MCP_SERVICE.API_PREFIX.lstrip("/")
    endpoint = endpoint.lstrip("/")
    
    if endpoint:
        return f"{base}/{prefix}/{endpoint}"
    return f"{base}/{prefix}"
