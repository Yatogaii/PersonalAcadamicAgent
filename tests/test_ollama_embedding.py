import dotenv
import os
from langchain_ollama import OllamaEmbeddings
dotenv.load_dotenv()


embed = OllamaEmbeddings(
    model="qwen3-embedding:4b",
)


print(embed.embed_query("HELLO"))