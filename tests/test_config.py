"""
Basic unit tests for the LLMOps pipeline.
Add more comprehensive tests as needed.
"""
import pytest
from src.config import Config


def test_config_loaded():
    """Test that configuration values are loaded."""
    assert Config.AZURE_OPENAI_API_KEY is not None
    assert Config.AZURE_OPENAI_ENDPOINT is not None


def test_vector_store_type():
    """Test vector store type configuration."""
    assert Config.VECTOR_STORE_TYPE in ["chroma", "azure_search"]


def test_mlflow_config():
    """Test MLflow configuration."""
    assert Config.MLFLOW_TRACKING_URI is not None
    assert Config.MLFLOW_EXPERIMENT_NAME is not None
