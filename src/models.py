from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from settings import settings

MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

DEFAULT_MODELSCOPE_CHAT_MODEL = "deepseek-ai/DeepSeek-V3.2-Exp"
DEFAULT_KIMI_CHAT_MODEL = "kimi-k2-0905-preview"
DEFAULT_DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEFAULT_OPENAI_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


class OpenAIEmbeddingClient:
    """OpenAI-compatible embedding client with the same surface as FeatureExtractor."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        encoding_format: str = "float",
    ) -> None:
        if not api_key:
            raise ValueError("Embedding API key is empty.")
        if not base_url:
            raise ValueError("Embedding base_url is empty.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("`openai` package is required for OpenAI-compatible embeddings.") from exc

        self.model_name = model_name
        self.encoding_format = encoding_format
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def embed_query(self, text: str) -> list[float]:
        payload = text if text is not None else ""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=payload,
            encoding_format=self.encoding_format,
        )
        return response.data[0].embedding

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            encoding_format=self.encoding_format,
        )
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_text(texts)


def _normalize_usage(usage: str) -> str:
    return (usage or "").strip().lower()


def _resolve_embedding_api_key() -> str:
    return (settings.embedding_model_api_key or settings.OPENAI_API_KEY or "").strip()


def init_chat_model_from_modelscope(model_name: str = DEFAULT_MODELSCOPE_CHAT_MODEL) -> BaseChatModel:
    return init_chat_model(
        model=model_name,
        model_provider="openai",
        base_url=MODELSCOPE_BASE_URL,
        api_key=settings.OPENAI_API_KEY,
    )


def init_kimi_k2(model_name: str = DEFAULT_KIMI_CHAT_MODEL) -> BaseChatModel:
    return init_chat_model(
        model=model_name,
        model_provider="openai",
        base_url=MOONSHOT_BASE_URL,
        api_key=settings.KIMI_API_KEY,
    )


def init_ollama_model(model_name: str = "qwen3:8b") -> BaseChatModel:
    return init_chat_model(
        model=model_name,
        model_provider="ollama",
        base_url=settings.OLLAMA_API_URL,
    )


def init_deepseek(model_name: str = DEFAULT_DEEPSEEK_CHAT_MODEL) -> BaseChatModel:
    return init_chat_model(
        model=model_name,
        model_provider="openai",
        base_url=DEEPSEEK_BASE_URL,
        api_key=settings.DEEPSEEK_KEY,
    )


def init_openai_embedding_model(model_name: str | None = None) -> OpenAIEmbeddingClient:
    return OpenAIEmbeddingClient(
        model_name=model_name or settings.embedding_model or DEFAULT_OPENAI_EMBEDDING_MODEL,
        base_url=settings.embedding_model_base_url or MODELSCOPE_BASE_URL,
        api_key=_resolve_embedding_api_key(),
    )


def get_llm_by_usage(
    usage: str = "evaluation",
    model_name: str | None = None,
    model_type: str = "chat",
):
    """Return chat or embedding model by usage.

    Args:
        usage: One of 'agentic', 'evaluation', 'contextual'.
        model_name: Optional override model name.
        model_type: 'chat' or 'embedding'.
    """
    u = _normalize_usage(usage)
    t = (model_type or "chat").strip().lower()

    if t == "embedding":
        provider = (settings.embedding_provider or "openai").strip().lower()
        if provider == "openai":
            return init_openai_embedding_model(model_name)

        # Keep compatibility for non-openai providers if users still configure them.
        from rag.feature_extractor import FeatureExtractor

        if provider == "huggingface":
            api_key = settings.embedding_model_api_key or settings.HF_TOKEN
        else:
            api_key = settings.embedding_model_api_key

        return FeatureExtractor(
            provider=provider,
            api_key=api_key,
            model=model_name or settings.embedding_model,
        )

    try:
        if u == "agentic":
            return init_kimi_k2(model_name or DEFAULT_KIMI_CHAT_MODEL)
        if u in ("evaluation", "contextual"):
            return init_deepseek(model_name or DEFAULT_DEEPSEEK_CHAT_MODEL)
    except Exception as exc:
        try:
            return init_chat_model_from_modelscope(model_name or DEFAULT_MODELSCOPE_CHAT_MODEL)
        except Exception:
            raise exc

    return init_chat_model_from_modelscope(model_name or DEFAULT_MODELSCOPE_CHAT_MODEL)


def get_embedding_by_usage(usage: str = "contextual", model_name: str | None = None):
    return get_llm_by_usage(usage=usage, model_name=model_name, model_type="embedding")
