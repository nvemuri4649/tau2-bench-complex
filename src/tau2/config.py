# SIMULATION
DEFAULT_MAX_STEPS = 200
DEFAULT_MAX_ERRORS = 10
DEFAULT_SEED = 300
DEFAULT_MAX_CONCURRENCY = 3
DEFAULT_NUM_TRIALS = 1
DEFAULT_SAVE_TO = None
DEFAULT_LOG_LEVEL = "ERROR"

# LLM
# Available models:
#   OpenAI: gpt-4.1, gpt-4o, gpt-4o-mini, gpt-5.2, gpt-5.2-chat-latest, gpt-5.2-pro
#   Anthropic: claude-3-5-sonnet-20241022, claude-3-opus-20240229
# 
# GPT-5.2 Notes (released Dec 2025):
#   - gpt-5.2: Thinking model, 400K context, best for complex reasoning
#   - gpt-5.2-chat-latest: Instant model, 128K context, optimized for everyday tasks  
#   - gpt-5.2-pro: Pro model, 400K context, highest accuracy
#   - Pricing: $1.75/M input, $14/M output (gpt-5.2/chat-latest)
#   - Pricing: $21/M input, $168/M output (gpt-5.2-pro)

DEFAULT_AGENT_IMPLEMENTATION = "llm_agent"
DEFAULT_USER_IMPLEMENTATION = "user_simulator"
DEFAULT_LLM_AGENT = "gpt-4.1"
DEFAULT_LLM_USER = "gpt-4.1"
DEFAULT_LLM_TEMPERATURE_AGENT = 0.0
DEFAULT_LLM_TEMPERATURE_USER = 0.0
DEFAULT_LLM_ARGS_AGENT = {"temperature": DEFAULT_LLM_TEMPERATURE_AGENT}
DEFAULT_LLM_ARGS_USER = {"temperature": DEFAULT_LLM_TEMPERATURE_USER}

# For complex tasks with long tool responses, GPT-5.2 is recommended
DEFAULT_LLM_AGENT_COMPLEX = "gpt-5.2"
DEFAULT_LLM_USER_COMPLEX = "gpt-5.2-chat-latest"

DEFAULT_LLM_NL_ASSERTIONS = "gpt-4o-mini"
DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE = 0.0
DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE}

DEFAULT_LLM_ENV_INTERFACE = "gpt-4.1"
DEFAULT_LLM_ENV_INTERFACE_TEMPERATURE = 0.0
DEFAULT_LLM_ENV_INTERFACE_ARGS = {"temperature": DEFAULT_LLM_ENV_INTERFACE_TEMPERATURE}

# LITELLM
DEFAULT_MAX_RETRIES = 3
LLM_CACHE_ENABLED = False
DEFAULT_LLM_CACHE_TYPE = "redis"

# REDIS CACHE
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = ""
REDIS_PREFIX = "tau2"
REDIS_CACHE_VERSION = "v1"
REDIS_CACHE_TTL = 60 * 60 * 24 * 30

# LANGFUSE
USE_LANGFUSE = False  # If True, make sure all the env variables are set for langfuse.

# API
API_PORT = 8000
