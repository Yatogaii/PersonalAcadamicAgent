from typing import Any, Callable, Dict

from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Use register to get class

class FeatureExtractor:
    def __init__(self,
                 provider: str,
                 api_key: str,
                 model: str,
                ) -> None:
        factory = self._providers.get(provider)
        if factory is None:
            providers = ', '.join(sorted(self._providers.keys()))
            raise RuntimeError(
                f"Unimplemented Feature Extractor Provider '{provider}'! Available: {providers}"
            )

        # factory returns a client instance
        self.client = factory(api_key, model)

    # Registry of provider name -> factory(api_key, model) callable
    _providers: Dict[str, Callable[[str, str], Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[[str, str], Any]], Callable[[str, str], Any]]:
        """Decorator to register a provider factory.

        The decorated callable must accept (api_key, model) and return a client instance.
        """

        def decorator(func: Callable[[str, str], Any]) -> Callable[[str, str], Any]:
            cls._providers[name] = func
            return func

        return decorator

    @classmethod
    def get_supported_providers(cls) -> list:
        """Return a list of supported provider names."""
        return list(cls._providers.keys())
    
    def embed_text(self, text:list[str]): 
        return self.client.embed_text(text)

    def embed_query(self, text:str):
        return self.client.embed_query(text)

# Register default provider(s)
@FeatureExtractor.register("huggingface")
def _register_huggingface(api_key: str, model: str):
    return HuggingFaceEndpointEmbeddings(
        model=model,
        huggingfacehub_api_token=api_key
    )