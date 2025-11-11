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
                             base_url="https://api-inference.modelscope.cn/v1")
    return model

def init_kimi_k2() -> BaseChatModel:
    model = init_chat_model(model="kimi-k2-0905-preview",
                             model_provider="openai",
                             base_url="https://api.moonshot.cn/v1")
    return model
