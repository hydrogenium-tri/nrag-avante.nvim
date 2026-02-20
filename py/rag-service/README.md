# RAG Service Configuration

This document describes how to configure the RAG service, including setting up Language Model (LLM) and Embedding providers.

## Provider Support Matrix

The following table shows which model types are supported by each provider:

| Provider   | LLM Support | Embedding Support |
| ---------- | ----------- | ----------------- |
| dashscope  | Yes         | Yes               |
| ollama     | Yes         | Yes               |
| openai     | Yes         | Yes               |
| openrouter | Yes | No |
| openai_like | Yes | Yes |

## LLM Provider Configuration

The `llm` section in the configuration file is used to configure the Language Model (LLM) used by the RAG service.

Here are the configuration examples for each supported LLM provider:

### OpenAI LLM Configuration

[See more configurations](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/base.py#L130)

```lua
llm = { -- Configuration for the Language Model (LLM) used by the RAG service
  provider = "openai", -- The LLM provider ("openai")
  endpoint = "https://api.openai.com/v1", -- The LLM API endpoint
  api_key = "OPENAI_API_KEY", -- The environment variable name for the LLM API key
  model = "gpt-4o-mini", -- The LLM model name (e.g., "gpt-4o-mini", "gpt-3.5-turbo")
  extra = {-- Extra configuration options for the LLM (optional)
    temperature = 0.7, -- Controls the randomness of the output. Lower values make it more deterministic.
    max_tokens = 512, -- The maximum number of tokens to generate in the completion.
    -- system_prompt = "You are a helpful assistant.", -- A system prompt to guide the model's behavior.
    -- timeout = 120, -- Request timeout in seconds.
  },
},
```

### DashScope LLM Configuration

[See more configurations](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/llms/llama-index-llms-dashscope/llama_index/llms/dashscope/base.py#L155)

```lua
llm = { -- Configuration for the Language Model (LLM) used by the RAG service
  provider = "dashscope", -- The LLM provider ("dashscope")
  endpoint = "", -- The LLM API endpoint (DashScope typically uses default or environment variables)
  api_key = "DASHSCOPE_API_KEY", -- The environment variable name for the LLM API key
  model = "qwen-plus", -- The LLM model name (e.g., "qwen-plus", "qwen-max")
  extra = nil, -- Extra configuration options for the LLM (optional)
},
```

### Ollama LLM Configuration

[See more configurations](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py#L65)

```lua
llm = { -- Configuration for the Language Model (LLM) used by the RAG service
  provider = "ollama", -- The LLM provider ("ollama")
  endpoint = "http://localhost:11434", -- The LLM API endpoint for Ollama
  api_key = "", -- Ollama typically does not require an API key
  model = "llama2", -- The LLM model name (e.g., "llama2", "mistral")
  extra = nil, -- Extra configuration options for the LLM (optional) Kristin", -- Extra configuration options for the LLM (optional)
},
```

### OpenRouter LLM Configuration

[See more configurations](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/llms/llama-index-llms-openrouter/llama_index/llms/openrouter/base.py#L17)

```lua
llm = { -- Configuration for the Language Model (LLM) used by the RAG service
  provider = "openrouter", -- The LLM provider ("openrouter")
  endpoint = "https://openrouter.ai/api/v1", -- The LLM API endpoint for OpenRouter
  api_key = "OPENROUTER_API_KEY", -- The environment variable name for the LLM API key
  model = "openai/gpt-4o-mini", -- The LLM model name (e.g., "openai/gpt-4o-mini", "mistralai/mistral-7b-instruct")
  extra = nil, -- Extra configuration options for the LLM (optional)
},
```

### OpenAI-Like LLM Configuration (For OpenAI-Compatible APIs)

Use this provider to access any service that implements the OpenAI API protocol, including:
- **Zhipu AI (GLM series)**: glm-4-flash, glm-4, etc.
- **Alibaba DashScope (Qwen series)**: qwen-plus, qwen-max, etc.
- **Moonshot AI (Kimi)**: moonshot-v1-8k, moonshot-v1-32k, etc.
- **DeepSeek**: deepseek-chat, deepseek-coder, etc.
- **01.AI (Yi series)**: yi-large, yi-medium, etc.
- **Local deployments**: Ollama, vLLM, etc.

[See more configurations](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/llms/llama-index-llms-openai-like/llama_index/llms/openai_like/base.py#L17)

#### Example: Zhipu GLM-4-Flash

```lua
llm = { -- Configuration for the Language Model (LLM) used by the RAG service
 provider = "openai_like", -- The LLM provider ("openai_like")
 endpoint = "https://open.bigmodel.cn/api/paas/v4", -- GLM OpenAI-compatible endpoint
 api_key = "GLM_API_KEY", -- Environment variable name for GLM API key
 model = "glm-4-flash", -- The GLM model name
 extra = {
 temperature = 0.7,
 max_tokens = 4096,
 context_window = 32768, -- GLM-4-Flash context window
 },
},
```

#### Example: Alibaba Qwen (DashScope)

```lua
llm = { -- Configuration for the Language Model (LLM) used by the RAG service
 provider = "openai_like", -- The LLM provider ("openai_like")
 endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1", -- DashScope OpenAI-compatible endpoint
 api_key = "DASHSCOPE_API_KEY", -- Environment variable name for DashScope API key
 model = "qwen-plus", -- The Qwen model name
 extra = {
 temperature = 0.7,
 max_tokens = 4096,
 context_window = 32768,
 },
},
```

#### Example: Moonshot Kimi

```lua
llm = { -- Configuration for the Language Model (LLM) used by the RAG service
 provider = "openai_like", -- The LLM provider ("openai_like")
 endpoint = "https://api.moonshot.cn/v1", -- Moonshot OpenAI-compatible endpoint
 api_key = "MOONSHOT_API_KEY", -- Environment variable name for Moonshot API key
 model = "moonshot-v1-8k", -- The Moonshot model name
 extra = {
 temperature = 0.7,
 max_tokens = 4096,
 context_window = 8192,
 },
},
```

## Embedding Provider Configuration

The `embedding` section in the configuration file is used to configure the Embedding Model used by the RAG service.

Here are the configuration examples for each supported Embedding provider:

### OpenAI Embedding Configuration

[See more configurations](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/embeddings/llama-index-embeddings-openai/llama_index/embeddings/openai/base.py#L214)

```lua
embed = { -- Configuration for the Embedding Model used by the RAG service
  provider = "openai", -- The Embedding provider ("openai")
  endpoint = "https://api.openai.com/v1", -- The Embedding API endpoint
  api_key = "OPENAI_API_KEY", -- The environment variable name for the Embedding API key
  model = "text-embedding-3-large", -- The Embedding model name (e.g., "text-embedding-3-small", "text-embedding-3-large")
  extra = {-- Extra configuration options for the Embedding model (optional)
    dimensions = nil,
  },
},
```

### DashScope Embedding Configuration

[See more configurations](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/embeddings/llama-index-embeddings-dashscope/llama_index/embeddings/dashscope/base.py#L156)

```lua
embed = { -- Configuration for the Embedding Model used by the RAG service
  provider = "dashscope", -- The Embedding provider ("dashscope")
  endpoint = "", -- The Embedding API endpoint (DashScope typically uses default or environment variables)
  api_key = "DASHSCOPE_API_KEY", -- The environment variable name for the Embedding API key
  model = "text-embedding-v3", -- The Embedding model name (e.g., "text-embedding-v2")
  extra = { -- Extra configuration options for the Embedding model (optional)
    embed_batch_size = 10,
  },
},
```

### Ollama Embedding Configuration

[See more configurations](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/embeddings/llama-index-embeddings-ollama/llama_index/embeddings/ollama/base.py#L12)

```lua
embed = { -- Configuration for the Embedding Model used by the RAG service
  provider = "ollama", -- The Embedding provider ("ollama")
  endpoint = "http://localhost:11434", -- The Embedding API endpoint for Ollama
  api_key = "", -- Ollama typically does not require an API key
  model = "nomic-embed-text", -- The Embedding model name (e.g., "nomic-embed-text")
  extra = { -- Extra configuration options for the Embedding model (optional)
    embed_batch_size = 10，
  },
},
```
