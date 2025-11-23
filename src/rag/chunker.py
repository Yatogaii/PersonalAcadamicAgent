from settings import settings
from models import init_chat_model_from_modelscope

class Chunker:
    def __init__(self):
        self.strategy = settings.chunk_strategy

    def chunk(self, text: str) -> list[str]:
        '''
        Chunk the input text based on the specified strategy.
        '''
        if self.strategy == "paragraph":
            return self._chunk_by_paragraph(text)
        elif self.strategy == "fixed_size":
            return self._chunk_by_fixed_size(text, size=settings.chunk_fixed_size)
        elif self.strategy == "sentence":
            return self._chunk_by_sentence(text)
        elif self.strategy == "contextual":
            return self._chunk_by_contextual(text)
        else:
            raise ValueError(f"Unsupported chunk strategy: {self.strategy}")

    def _chunk_by_paragraph(self, text: str) -> list[str]:
        return [para.strip() for para in text.split('\n\n') if para.strip()]

    def _chunk_by_fixed_size(self, text: str, size: int) -> list[str]:
        return [text[i:i+size] for i in range(0, len(text), size)]

    def _chunk_by_sentence(self, text: str) -> list[str]:
        paras = self._chunk_by_paragraph(text)
        return [sentence for para in paras if para.strip() for sentence in para.split('.') if sentence.strip()]
        
    def _chunk_by_contextual(self, text: str) -> list[str]:
        context_model = init_chat_model_from_modelscope("deepseek-ai/DeepSeek-V3.2-Exp")

        return []