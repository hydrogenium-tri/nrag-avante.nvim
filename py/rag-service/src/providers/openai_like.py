# src/providers/openai_like.py
"""
OpenAI-like provider for accessing non-OpenAI models through OpenAI-compatible APIs.

This provider uses llama_index.llms.openai_like.OpenAILike to connect to any service
that implements the OpenAI API protocol, such as:
- Alibaba DashScope (Qwen series)
- Zhipu AI (GLM series)
- Moonshot AI (Kimi)
- DeepSeek
- 01.AI (Yi series)
- Local deployments (Ollama, vLLM, etc.)
"""

from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike


def initialize_embed_model(
    embed_endpoint: str,
    embed_api_key: str,
    embed_model: str,
    **embed_extra: Any,  # noqa: ANN401
) -> BaseEmbedding:
    """
    Create OpenAI-compatible embedding model.

    This uses the standard OpenAI embedding client with a custom endpoint,
    which works with any OpenAI-compatible embedding API.

    Args:
        embed_endpoint: The API endpoint for the OpenAI-compatible API.
        embed_api_key: The API key for the service.
        embed_model: The name of the embedding model.
        embed_extra: Extra parameters for the embedding model.

    Returns:
        The initialized embed_model.

    """
    # Use OpenAIEmbedding with custom endpoint for compatibility
    # This works with any OpenAI-compatible embedding API
    return OpenAIEmbedding(
        model=embed_model,
        api_base=embed_endpoint,
        api_key=embed_api_key,
        **embed_extra,
    )


def initialize_llm_model(
    llm_endpoint: str,
    llm_api_key: str,
    llm_model: str,
    **llm_extra: Any,  # noqa: ANN401
) -> LLM:
    """
    Create OpenAI-like LLM model for non-OpenAI providers.

    This uses OpenAILike which is specifically designed for OpenAI-compatible
    APIs from other providers. It automatically handles model validation and
    works with any service implementing the OpenAI protocol.

    Args:
        llm_endpoint: The API endpoint for the OpenAI-compatible API.
        llm_api_key: The API key for the service.
        llm_model: The name of the LLM model.
        llm_extra: Extra parameters for the LLM model.

    Returns:
        The initialized llm_model.

    Examples:
        ```python
        # Zhipu GLM
        llm = initialize_llm_model(
            llm_endpoint="https://open.bigmodel.cn/api/paas/v4",
            llm_api_key="your-glm-api-key",
            llm_model="glm-4-flash",
        )

        # Alibaba Qwen
        llm = initialize_llm_model(
            llm_endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
            llm_api_key="your-dashscope-api-key",
            llm_model="qwen-plus",
        )

        # Moonshot Kimi
        llm = initialize_llm_model(
            llm_endpoint="https://api.moonshot.cn/v1",
            llm_api_key="your-moonshot-api-key",
            llm_model="moonshot-v1-8k",
        )
        ```

    """
    # Extract context_window if provided, default to 32k for unknown models
    context_window = llm_extra.pop("context_window", 32768)

    # Use OpenAILike which is designed for OpenAI-compatible APIs
    # This automatically handles model validation and works with any
    # service implementing the OpenAI protocol
    return OpenAILike(
        model=llm_model,
        api_base=llm_endpoint,
        api_key=llm_api_key,
        context_window=context_window,
        is_chat_model=True,  # Most modern LLMs are chat models
        **llm_extra,
    )

