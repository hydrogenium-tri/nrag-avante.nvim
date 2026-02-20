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
 embed_batch_size = 10,
 },
},
```

### OpenAI-Like Embedding Configuration (For OpenAI-Compatible APIs)

Use this provider to access any service that implements the OpenAI Embeddings API protocol, including:
- **Gitee AI**: Qwen3-Embedding series, bge-large-zh-v1.5, etc.
- **Zhipu AI**: embedding-2, embedding-3, etc.
- **Alibaba DashScope**: text-embedding-v1, text-embedding-v2, etc.
- **Moonshot AI**: moonshot-embedding-v1, etc.
- **Local deployments**: Ollama, vLLM with embedding models, etc.

[See more configurations](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/embeddings/llama-index-embeddings-openai-like/llama_index/embeddings/openai_like/base.py)

#### Example: Gitee AI Qwen3-Embedding-8B

```lua
embed = { -- Configuration for the Embedding Model used by the RAG service
 provider = "openai_like", -- The Embedding provider ("openai_like")
 endpoint = "https://ai.gitee.com/v1", -- Gitee AI OpenAI-compatible endpoint
 api_key = "GITEE_AI_API_KEY", -- Environment variable name for Gitee AI API key
 model = "Qwen3-Embedding-8B", -- The Gitee AI embedding model name
 extra = {-- Extra configuration options (optional)
 dimensions = 1024, -- Output embedding dimensions (model-specific)
 max_retries = 3, -- Number of retry attempts for failed requests
 },
},
```

#### Example: Zhipu AI Embedding-2

```lua
embed = { -- Configuration for the Embedding Model used by the RAG service
 provider = "openai_like", -- The Embedding provider ("openai_like")
 endpoint = "https://open.bigmodel.cn/api/paas/v4", -- Zhipu AI OpenAI-compatible endpoint
 api_key = "GLM_API_KEY", -- Environment variable name for Zhipu AI API key
 model = "embedding-2", -- The Zhipu AI embedding model name
 extra = {-- Extra configuration options (optional)
 dimensions = 1024,
 max_retries = 3,
 },
},
```

## Complete Configuration Examples

### Example 1: Zhipu AI LLM + Gitee AI Embedding (Recommended for Chinese Users)

This configuration uses Zhipu AI's GLM-4.7-Flash for language tasks and Gitee AI's Qwen3-Embedding-8B for embeddings:

```lua
rag_service = {
 enabled = true,
 host_mount = os.getenv("HOME"),
 runner = "docker",
 image = "avante-rag-service:local", -- Use local image for custom providers
 llm = { -- Zhipu AI GLM-4.7-Flash for language tasks
  provider = "openai_like",
  endpoint = "https://open.bigmodel.cn/api/paas/v4",
  api_key = "GLM_API_KEY", -- Set export GLM_API_KEY=your_zhipu_api_key
  model = "glm-4.7-flash", -- Latest Zhipu GLM model
  extra = {
   temperature = 0.7,
   max_tokens = 4096,
   context_window = 32768,
  },
 },
 embed = { -- Gitee AI Qwen3-Embedding-8B for embeddings
  provider = "openai_like",
  endpoint = "https://ai.gitee.com/v1",
  api_key = "GITEE_AI_API_KEY", -- Set export GITEE_AI_API_KEY=your_gitee_api_key
  model = "Qwen3-Embedding-8B", -- Gitee AI embedding model
  extra = {
   dimensions = 1024,
   max_retries = 3,
  },
 },
 docker_extra_args = "",
},
```

**Environment Variables Setup:**
```bash
# Zhipu AI API Key (for LLM)
export GLM_API_KEY="your_zhipu_api_key_here"

# Gitee AI API Key (for Embedding)
export GITEE_AI_API_KEY="your_gitee_ai_api_key_here"
```

### Example 2: OpenAI GPT-4 + OpenAI Embeddings

```lua
rag_service = {
 enabled = true,
 host_mount = os.getenv("HOME"),
 runner = "docker",
 llm = {
  provider = "openai",
  endpoint = "https://api.openai.com/v1",
  api_key = "OPENAI_API_KEY",
  model = "gpt-4o-mini",
  extra = {
   temperature = 0.7,
   max_tokens = 512,
  },
 },
 embed = {
  provider = "openai",
  endpoint = "https://api.openai.com/v1",
  api_key = "OPENAI_API_KEY",
  model = "text-embedding-3-large",
  extra = {
   dimensions = 1024,
  },
 },
},
```

### Example 3: Ollama Local LLM + Ollama Local Embedding

```lua
rag_service = {
 enabled = true,
 host_mount = os.getenv("HOME"),
 runner = "docker",
 llm = {
  provider = "ollama",
  endpoint = "http://host.docker.internal:11434", -- Use host.docker.internal to access host from Docker
  api_key = "",
  model = "llama3.1",
  extra = nil,
 },
 embed = {
  provider = "ollama",
  endpoint = "http://host.docker.internal:11434",
  api_key = "",
  model = "nomic-embed-text",
  extra = {
   embed_batch_size = 10,
  },
 },
},
```

## Important Notes

### LLM vs Embedding Configuration Differences

1. **LLM Configuration** (`llm` section):
   - Used for generating responses, understanding queries, and reasoning
   - Requires more computational power
   - Models: GPT-4, GLM-4, Qwen, Llama, etc.
   - Parameters: `temperature`, `max_tokens`, `context_window`

2. **Embedding Configuration** (`embed` section):
   - Used for converting text to vector representations
   - Used for semantic search and retrieval
   - Models: text-embedding-3-large, Qwen3-Embedding-8B, etc.
   - Parameters: `dimensions` (output vector size), `embed_batch_size`

### Key Points

- **Provider Independence**: You can mix and match different providers for LLM and Embedding
  - Example: Zhipu AI for LLM + Gitee AI for Embedding
  - Example: Ollama for LLM + OpenAI for Embedding

- **API Key Security**: Never hardcode API keys in your configuration!
  - Always use environment variables: `api_key = "ENV_VAR_NAME"`
  - The RAG service will read the value from the environment

- **Context Window**: For LLM, set `context_window` appropriately based on the model's capabilities
  - GLM-4.7-Flash: 32768 tokens
  - GPT-4o-mini: 16384 tokens
  - Qwen-Plus: 32768 tokens

- **Embedding Dimensions**: Different embedding models output different vector sizes
  - Qwen3-Embedding-8B: 1024 dimensions
  - text-embedding-3-large: 3072 dimensions (configurable)
  - embedding-2: 1024 dimensions

### Troubleshooting

**Error: "No schema matches" (400 Bad Request)**
- This is usually caused by invalid content being sent to the embedding API
- The RAG service automatically skips invalid documents and continues
- Error rate < 1% is considered normal and acceptable

**Error: "Failed to load index from storage"**
- This appears on first run when no index exists yet
- The service will automatically create a new index
- Safe to ignore

**Building Local Image**:
If you're using custom providers (like `openai_like`), you need to build a local Docker image:
```bash
cd py/rag-service
docker build -t avante-rag-service:local .
```

Then update your Neovim config:
```lua
image = "avante-rag-service:local",
```
