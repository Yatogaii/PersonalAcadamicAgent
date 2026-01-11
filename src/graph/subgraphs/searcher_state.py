from typing import Any, Dict, List, TypedDict


class SearcherState(TypedDict, total=False):
    original_query: str
    analysis: Dict[str, Any]
    sub_queries: List[str]
    current_round: int
    current_query: str
    search_query: str
    should_use_hyde: bool
    hyde_document: str
    max_rounds: int
    all_results: List[Dict[str, Any]]
    round_results: List[Dict[str, Any]]
    deep_results: List[Dict[str, Any]]
    evaluation: Dict[str, Any]
    should_continue: bool
    use_deep_search: bool
    pdf_load_report: Any
    errors: List[str]
