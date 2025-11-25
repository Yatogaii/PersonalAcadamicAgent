import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from settings import settings
# Enable Agentic RAG for this test
settings.enable_agentic_rag = True

from agents.searcher import (
    Searcher, 
    search_abstracts, 
    search_by_section, 
    get_context_window, 
    get_paper_introduction
)

def test_tools_exist():
    """Test that all tools are properly defined."""
    print("Testing tool definitions...")
    
    # Check tool names and docstrings
    assert search_abstracts.name == "search_abstracts"
    assert "abstract" in search_abstracts.description.lower()
    
    assert search_by_section.name == "search_by_section"
    assert "section" in search_by_section.description.lower()
    
    assert get_context_window.name == "get_context_window"
    assert "context" in get_context_window.description.lower()
    
    assert get_paper_introduction.name == "get_paper_introduction"
    assert "introduction" in get_paper_introduction.description.lower()
    
    print("All tools properly defined!")

def test_agentic_searcher_init():
    """Test that Agentic Searcher initializes with tools."""
    print("\nTesting Agentic Searcher initialization...")
    
    with patch('agents.searcher.get_rag_client_by_provider') as MockGetClient, \
         patch('agents.searcher.init_kimi_k2') as MockInitLLM:
        
        mock_rag = MockGetClient.return_value
        mock_llm = MockInitLLM.return_value
        
        searcher = Searcher()
        
        # Check that tools are set up
        assert hasattr(searcher, 'tools')
        assert len(searcher.tools) == 4
        
        # Check that system prompt is loaded
        assert hasattr(searcher, 'system_prompt')
        assert "retrieval agent" in searcher.system_prompt.lower()
        
        print("Agentic Searcher initialized successfully!")
        print(f"Number of tools: {len(searcher.tools)}")
        print(f"Tool names: {[t.name for t in searcher.tools]}")

def test_search_abstracts_tool():
    """Test the search_abstracts tool directly."""
    print("\nTesting search_abstracts tool...")
    
    with patch('agents.searcher._get_rag_client') as MockGetClient:
        mock_rag = MockGetClient.return_value
        mock_rag.search_abstracts.return_value = [
            {
                "title": "Test Paper 1",
                "abstract": "This is a test abstract about RAG systems.",
                "doc_id": "doc_001",
                "url": "http://example.com/1",
                "pdf_url": "",
                "conference_name": "USENIX",
                "conference_year": 2024,
                "score": 0.95
            }
        ]
        
        result = search_abstracts.invoke({"query": "RAG systems", "k": 5})
        
        assert "Test Paper 1" in result
        assert "doc_001" in result
        print("search_abstracts result:")
        print(result[:300])

def test_search_by_section_tool():
    """Test the search_by_section tool directly."""
    print("\nTesting search_by_section tool...")
    
    with patch('agents.searcher._get_rag_client') as MockGetClient:
        mock_rag = MockGetClient.return_value
        mock_rag.search_by_section.return_value = [
            {
                "title": "Test Paper",
                "text": "We propose a novel method for improving RAG performance...",
                "doc_id": "doc_001",
                "chunk_id": 42,
                "section_category": 2,  # METHOD
                "parent_section": "3. Methodology",
                "page_number": 5,
                "score": 0.9
            }
        ]
        
        result = search_by_section.invoke({
            "query": "RAG improvement method", 
            "doc_id": "doc_001", 
            "category": 2,
            "k": 5
        })
        
        assert "doc_001" in result
        assert "chunk_id: 42" in result
        assert "METHOD" in result
        print("search_by_section result:")
        print(result[:400])

if __name__ == "__main__":
    test_tools_exist()
    test_agentic_searcher_init()
    test_search_abstracts_tool()
    test_search_by_section_tool()
    print("\n" + "="*50)
    print("All tests passed!")
