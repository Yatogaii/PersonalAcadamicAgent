from settings import settings

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

def init_chat_model_from_modelscope(model_name="deepseek-ai/DeepSeek-V3.2-Exp") -> BaseChatModel:
    '''
    Initialize a chat model from ModelScope.

    Args:
        model_name: the name of the model.
    Return:
        the initialized chat model.
    '''
    model = init_chat_model(model=model_name,
                             model_provider="openai",
                             base_url="https://api-inference.modelscope.cn/v1",
                             api_key=settings.OPENAI_API_KEY)
    return model

def init_kimi_k2() -> BaseChatModel:
    model = init_chat_model(model="kimi-k2-0905-preview",
                             model_provider="openai",
                             base_url="https://api.moonshot.cn/v1",
                             api_key=settings.KIMI_API_KEY)
    return model

def init_ollama_model(model_name="qwen3:8b"):
    model = init_chat_model(model=model_name,
                             model_provider="ollama",
                             base_url=settings.OLLAMA_API_URL)
    return model

def init_deepseek(model_name="deepseek-chat"):
    model = init_chat_model(model=model_name,
                             model_provider="openai",
                             base_url="https://api.deepseek.com/v1",
                             api_key=settings.DEEPSEEK_KEY)
    return model


def get_llm_by_usage(usage: str = "evaluation", model_name: str | None = None) -> BaseChatModel:
    """Return a chat LLM tailored for a specific usage.

    Args:
        usage: One of 'agentic', 'evaluation', 'contextual'.
        model_name: Optional model name override for providers that support it.

    Mapping:
        - 'agentic' -> KIMI K2 (init_kimi_k2)
        - 'evaluation' -> GPT API (init_chat_model_from_gptapi)
        - 'contextual' -> GPT API (init_chat_model_from_gptapi)
        - fallback -> ModelScope (init_chat_model_from_modelscope)
    """
    u = (usage or "").lower()
    try:
        if u == "agentic":
            return init_kimi_k2()
        if u in ("evaluation", "contextual"):
            if model_name:
                return init_deepseek(model_name=model_name)
            return init_deepseek()
    except Exception as e:
        # If specific provider fails, try falling back to ModelScope
        try:
            return init_chat_model_from_modelscope() if model_name is None else init_chat_model_from_modelscope(model_name)
        except Exception:
            # Last resort: raise original exception
            raise e

    # Default fallback: ModelScope
    return init_chat_model_from_modelscope(model_name or "deepseek-ai/DeepSeek-V3.2-Exp")