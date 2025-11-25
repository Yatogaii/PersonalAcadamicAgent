import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
from pymilvus import MilvusClient
from src.settings import settings
from langchain_ollama import OllamaEmbeddings
from pprint import pprint

client = MilvusClient(settings.milvus_uri)
embed = OllamaEmbeddings(model=settings.embedding_model)

res = client.search(
    settings.milvus_collection,
    data=[embed.embed_query("Symbolic execution")],
    output_fields=[
        settings.milvus_text_field,
        settings.milvus_title_field
    ]
)

for each in res[0]:
    pprint(each)