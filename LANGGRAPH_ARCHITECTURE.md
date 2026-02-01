# LangGraph Architecture Diagram

Generated for PaperCollector project - feat/migration-to-langgraph branch

## Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Coordinator Graph                                  │
│  ┌──────────┐     ┌─────────────┐     ┌─────────────────────────────────┐  │
│  │  START   │────▶│ coordinator │────▶│  Conditional Routing            │  │
│  └──────────┘     │   (router)  │     │  Based on keywords in query     │  │
│                   └─────────────┘     └─────────────────────────────────┘  │
│                                                 │                           │
│                    ┌────────────────────────────┼────────────────────────┐  │
│                    │                            │                        │  │
│                    ▼                            ▼                        │  │
│           ┌─────────────┐              ┌─────────────┐                   │  │
│           │  collector  │              │  searcher   │                   │  │
│           │  subgraph   │              │  subgraph   │                   │  │
│           └──────┬──────┘              └──────┬──────┘                   │  │
│                  │                            │                          │  │
│                  └────────────┬───────────────┘                          │  │
│                               ▼                                          │  │
│                         ┌──────────┐                                     │  │
│                         │   END    │                                     │  │
│                         └──────────┘                                     │  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Coordinator Router Logic

```python
def coordinator_router(state):
    content = state["messages"][-1].content
    
    # Intent Classification (Keyword-based)
    collect_keywords = ["收集", "collect", "conference", "会议", ...]
    search_keywords = ["搜索", "search", "query", "查找", ...]
    
    if any(kw in content for kw in collect_keywords):
        return Command(goto="collector", update={
            "intent": "collect",
            "original_query": content,
            "goto": "collector"
        })
    else:
        return Command(goto="searcher", update={
            "intent": "search", 
            "original_query": content,
            "goto": "searcher"
        })
```

## Collector Subgraph (Paper Collection)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Collector Subgraph                                 │
│                                                                              │
│  ┌────────────────┐                                                         │
│  │  START/Input   │  Input: {conference, year, round}                        │
│  └───────┬────────┘                                                         │
│          ▼                                                                   │
│  ┌─────────────────────────┐                                                │
│  │   extract_parameters    │  Parse conference name, year, round            │
│  └───────────┬─────────────┘                                                │
│              ▼                                                               │
│  ┌─────────────────────────┐                                                │
│  │   check_existing        │  Check DB for existing rounds                  │
│  └───────────┬─────────────┘                                                │
│              │                                                               │
│     ┌────────┴────────┐                                                     │
│     │                 │                                                     │
│     ▼                 ▼                                                     │
│ ┌─────────────┐  ┌──────────────┐                                          │
│ │discover_    │  │  skip if     │  Round = "unspecified"?                   │
│ │rounds       │  │  exists      │  Auto-discover all rounds                 │
│ │(DDG search)  │  └──────────────┘                                          │
│ └──────┬──────┘                                                            │
│        │                                                                     │
│        ▼                                                                     │
│ ┌─────────────────────────┐                                                │
│ │    parse_papers         │  For each round:                               │
│ │                         │  - Search DDG for URL                          │
│ │  ┌─────────────────┐    │  - Parse HTML with selectors                   │
│ │  │  get_parsed_html│    │  - Save to JSON                                │
│ │  │  (Tool call)    │    │                                                │
│ │  └─────────────────┘    │                                                │
│ └───────────┬─────────────┘                                                │
│             ▼                                                                │
│ ┌─────────────────────────┐                                                │
│ │   enrich_papers         │  Add PDF URLs and missing abstracts            │
│ │                         │  Tool: enrich_papers_with_details              │
│ └───────────┬─────────────┘                                                │
│             ▼                                                                │
│ ┌─────────────────────────┐                                                │
│ │    save_to_db           │  Insert into RAG (Milvus/PGVector)             │
│ │                         │  Store: title, abstract, url, pdf_url          │
│ └───────────┬─────────────┘                                                │
│             ▼                                                                │
│       ┌──────────┐                                                          │
│       │   END    │  Output: {status, papers_collected, message}             │
│       └──────────┘                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Searcher Subgraph (Agentic RAG)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              Searcher Subgraph                                  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                           Phase 1: Query Analysis                        │   │
│  │  ┌──────────────┐    ┌─────────────────────┐    ┌─────────────────┐    │   │
│  │  │analyze_query │───▶│ generate_hyde       │───▶│ search_abstracts│    │   │
│  │  │              │    │ (if enabled)        │    │                 │    │   │
│  │  │- Query type  │    │- Hypothetical       │    │- Vector search  │    │   │
│  │  │- Sub-queries │    │  document           │    │- Top-k results  │    │   │
│  │  │- Complexity  │    │                     │    │                 │    │   │
│  │  └──────────────┘    └─────────────────────┘    └────────┬────────┘    │   │
│  └───────────────────────────────────────────────────────────┼────────────┘   │
│                                                              │                 │
│  ┌───────────────────────────────────────────────────────────┼────────────┐   │
│  │                      Phase 2-4: Iterative Retrieval        │            │   │
│  │                           (Multi-round loop)               │            │   │
│  │                                                             │            │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │            │   │
│  │  │search_content│───▶│   evaluate   │───▶│   rerank     │  │            │   │
│  │  │              │    │   progress   │    │   results    │  │            │   │
│  │  │- Deep search │    │              │    │              │  │            │   │
│  │  │- Section     │    │- Sufficient? │    │- LLM scoring │  │            │   │
│  │  │  filtering   │    │- Continue?   │    │- Top papers  │  │            │   │
│  │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │            │   │
│  │         │                   │                    │          │            │   │
│  │         │         ┌─────────┴─────────┐          │          │            │   │
│  │         │         │                   │          │          │            │   │
│  │         │    ┌────┴────┐         ┌────┴────┐     │          │            │   │
│  │         └───▶│ Continue│         │   Stop  │◀────┘          │            │   │
│  │              │ Loop    │         │         │                 │            │   │
│  │              └────┬────┘         └────┬────┘                 │            │   │
│  └───────────────────┼───────────────────┼──────────────────────┘            │   │
│                      │                   │                                    │   │
│  ┌───────────────────┴───────────────────┴──────────────────────────────┐    │   │
│  │                      Phase 5: Answer Generation                       │    │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │    │   │
│  │  │                      generate_answer                             │  │    │   │
│  │  │                                                                  │  │    │   │
│  │  │  - Synthesize information from top-ranked papers                │  │    │   │
│  │  │  - Generate citations                                           │  │    │   │
│  │  │  - Format final response                                        │  │    │   │
│  │  └───────────────────────────┬─────────────────────────────────────┘  │    │   │
│  │                              ▼                                        │    │   │
│  │                        ┌──────────┐                                   │    │   │
│  │                        │   END    │  Output: {answer, papers_found}   │    │   │
│  │                        └──────────┘                                   │    │   │
│  └───────────────────────────────────────────────────────────────────────┘    │   │
└────────────────────────────────────────────────────────────────────────────────┘
```

## State Flow

### CoordinatorState (Main Graph)
```python
class CoordinatorState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    intent: str                              # "collect" | "search"
    original_query: str                      # User's raw input
    goto: str                               # Routing target
    error: str                              # Error message if any
```

### CollectorState (Collector Subgraph)
```python
class CollectorState(TypedDict):
    conference_name: str
    year: int
    round: str
    discovered_rounds: List[str]
    search_results: List[dict]
    parsed_paths: List[Path]
    papers_collected: int
    status: str
    message: str
```

### SearcherState (Searcher Subgraph)
```python
class SearcherState(TypedDict):
    query: str
    query_analysis: Dict[str, Any]
    sub_queries: List[str]
    current_sub_query_index: int
    use_hyde: bool
    hyde_document: str
    retrieval_round: int
    candidate_papers: List[Dict]
    loaded_doc_ids: List[str]
    search_results: List[Dict]
    is_sufficient: bool
    coverage_score: float
    reranked_results: List[Dict]
    answer: str
    status: str
```

## Key Design Decisions

1. **Explicit vs Implicit**: Replaced `create_agent` with explicit `StateGraph`
2. **Type Safety**: All states use `TypedDict` with proper type annotations
3. **Message Reducer**: Uses `add_messages` to automatically merge message lists
4. **Conditional Routing**: `add_conditional_edges` with `Command` pattern
5. **Subgraph Pattern**: Collector and Searcher are independent compiled subgraphs
6. **Tool Reuse**: All original LangChain tools remain unchanged

## File Structure

```
src/graphs/
├── __init__.py              # Exports coordinator_graph
├── states.py               # CoordinatorState definition
├── coordinator_graph.py    # Main orchestration (4 nodes)
├── collector_graph.py      # Collection workflow (6 nodes)
├── searcher_graph.py       # Agentic RAG (9 nodes)
└── nodes/
    ├── __init__.py
    ├── collector_node.py   # Invokes collector_subgraph
    └── searcher_node.py    # Invokes searcher_subgraph
```

## Usage Flow

```
User Query
    │
    ▼
┌──────────────┐
│ main_langgraph│◀── Entry point (alternative to main.py)
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ coordinator_graph │◀── Routes to collector or searcher
│   .invoke()       │
└──────┬───────────┘
       │
       ├──────────────┐
       │              │
       ▼              ▼
┌───────────┐  ┌───────────┐
│ collector │  │ searcher  │
│ _subgraph │  │ _subgraph │
│  .invoke()│  │  .invoke()│
└─────┬─────┘  └─────┬─────┘
      │              │
      ▼              ▼
┌───────────┐  ┌───────────┐
│ Save to   │  │ Return    │
│ Database  │  │ Answer    │
└───────────┘  └───────────┘
```

---

Generated: 2026-02-01
Branch: feat/migration-to-langgraph
