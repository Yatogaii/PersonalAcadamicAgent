import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from settings import settings
# Enable Agentic RAG for this test
settings.enable_agentic_rag = True
settings.milvus_uri = "./milvus_demo.db" 

from agents.searcher import Searcher

def test_agentic_searcher():
    print("Testing Agentic Searcher...")
    
    # Mock dependencies
    with patch('agents.searcher.get_rag_client_by_provider') as MockGetClient, \
         patch('agents.searcher.init_kimi_k2') as MockInitLLM, \
         patch('agents.searcher.apply_prompt_template') as MockApplyPrompt:
        
        # Setup Mock RAG Client
        mock_rag = MockGetClient.return_value
        mock_rag.query_relevant_documents.return_value = [
            {"title": "Paper A", "abstract": "Abstract A", "doc_id": "1", "score": 0.9}
        ]
        
        # Setup Mock LLM
        mock_llm = MockInitLLM.return_value
        mock_response = MagicMock()
        mock_response.content = "Refined Query Keyword"
        mock_llm.invoke.return_value = mock_response
        
        # Setup Mock Prompt
        MockApplyPrompt.return_value = [{"role": "system", "content": "Prompt"}]
        
        # Initialize Searcher
        searcher = Searcher()
        
        # Test Search
        original_query = "tell me about RAG"
        print(f"Original Query: {original_query}")
        
        hits = searcher.search(original_query)
        
        # Verify LLM was called for refinement
        assert mock_llm.invoke.call_count == 1
        print("LLM invoked for refinement.")
        
        # Verify RAG client was called with REFINED query
        mock_rag.query_relevant_documents.assert_called_with("Refined Query Keyword")
        print(f"RAG Client called with: {mock_rag.query_relevant_documents.call_args[0][0]}")
        
        assert len(hits) == 1
        print("Hits retrieved successfully.")
        print("Test Passed!")

if __name__ == "__main__":
    test_agentic_searcher()
