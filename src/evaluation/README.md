# RAG 评估框架文档

## 目录
1. [快速开始](#快速开始)
2. [问题生成策略](#问题生成策略)
3. [架构概览](#架构概览)
4. [评估指标详解](#评估指标详解)
5. [对比实验配置](#对比实验配置)
6. [数据流程](#数据流程)
7. [API 参考](#api-参考)

---

## 快速开始

### 1. 运行评估（使用已有数据）

```bash
# L1 + L2 评估（快速，约 5 秒）
uv run python tests/test_evaluation_runner.py

# L1 + L2 + L3 评估（需要 LLM，约 20-30 分钟）
uv run python tests/test_evaluation_runner.py --full

# 只运行 L1 Paper Discovery
uv run python tests/test_evaluation_runner.py --l1-only

# 只运行 L3 End-to-End
uv run python tests/test_evaluation_runner.py --l3-only
```

### 2. 查看数据状态

```bash
uv run python scripts/run_full_evaluation.py --status
```

输出示例：
```
📄 Source Papers: 821 篇
📦 Chunks: 8 篇论文, 1486 个 chunks
❓ Ground Truth: 5 个 QA pairs
🗄️ Eval Collection: papers_eval_paragraph (1494 条记录)
📊 Reports: 3 个报告
```

### 3. 生成更多测试数据

```bash
# 从现有 chunks 生成更多 QA（不需要重新处理 PDF）
uv run python scripts/run_full_evaluation.py --generate-qa --num-questions 30

# 从业务库抽样更多论文（需要下载 PDF）
uv run python scripts/run_full_evaluation.py --full --sample 20 --num-questions 30
```

---

## 完整对比实验流程

使用 `--compare` 模式可以一键运行 **12 种配置**的对比实验：

```bash
uv run python scripts/run_full_evaluation.py --compare --sample 20 --num-questions 30
```

### 流程概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           完整对比实验流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 1: 数据准备                                                            │
│  ────────────────                                                            │
│  • 从业务库 (papers_rag) 抽样论文元数据                                        │
│  • 下载 PDF 文件到本地                                                        │
│  • 使用两种策略分别处理 chunks:                                                │
│      - paragraph: 直接按段落分块                                              │
│      - contextual: LLM 增强上下文分块                                         │
│  • 创建 2 个评估 collection:                                                  │
│      - papers_eval_paragraph                                                 │
│      - papers_eval_contextual                                                │
│                                                                              │
│  Step 2: QA 生成                                                             │
│  ──────────────                                                              │
│  • 从 chunks 中用 LLM 生成问答对                                              │
│  • 每个 QA 包含: 问题、答案、来源论文、来源章节、期望 chunks                      │
│  • 保存到 ground_truth.json                                                  │
│                                                                              │
│  Step 3: 运行 12 种实验                                                       │
│  ─────────────────────                                                       │
│  对于每种配置组合:                                                             │
│                                                                              │
│    2 Chunk 策略   ×   3 Index 类型   ×   2 RAG 模式   =   12 种配置           │
│    ─────────────     ─────────────      ───────────                          │
│    • paragraph       • FLAT (精确)      • basic                              │
│    • contextual      • HNSW (近似)      • agentic                            │
│                      • IVF_FLAT                                              │
│                                                                              │
│  每种配置执行:                                                                 │
│    1. 切换到对应的 collection                                                 │
│    2. 重建对应类型的索引                                                       │
│    3. 运行 L1 评估 (Paper Discovery)                                          │
│    4. 运行 L2 评估 (Section Retrieval)                                        │
│    5. 运行 L3 评估 (End-to-End QA, 可选)                                      │
│    6. 记录指标                                                                │
│                                                                              │
│  Step 4: 生成对比报告                                                         │
│  ────────────────────                                                        │
│  • 汇总所有配置的指标                                                          │
│  • 生成 Markdown 对比表格                                                     │
│  • 保存 JSON 详细报告                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 资源说明

| 资源 | 数量 | 说明 |
|------|------|------|
| **Eval Tables/Collections** | 2 个 | sqlite-vec 表或 Milvus collection：`papers_eval_paragraph`, `papers_eval_contextual` |
| **索引重建** | 6 次 | 每个 collection 重建 3 种索引类型 |
| **评估运行** | 12 次 | 2 chunk × 3 index × 2 agentic |
| **LLM 调用** | ~30×12 次 (L3) | 每个 QA 需要 LLM 评判 |

### 快速模式

```bash
# 跳过 L3 评估（快很多，只比较检索性能）
uv run python scripts/run_full_evaluation.py --compare --sample 20 --no-l3

# 使用已有数据（跳过 Step 1 和 Step 2）
uv run python scripts/run_full_evaluation.py --compare --skip-data-prep
```

---

## 问题生成策略

### 设计理念

传统的 RAG 评估问题往往过于简单（单论文、关键词匹配），无法真正测试系统的综合能力。我们设计了 **4 层难度** 的问题生成策略，重点测试跨论文综合能力。

### 4 层难度体系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              问题生成策略 (V2)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Level 1: 单论文精确题 (Easy) - 20%                                          │
│  ─────────────────────────────────                                           │
│  • 问题: "哪篇论文提出了 TEE-based bot detection?"                            │
│  • 特点: 包含论文关键词，直接匹配即可找到                                       │
│  • 答案来源: Abstract                                                        │
│  • 测试能力: 基础检索                                                         │
│                                                                              │
│  Level 2: 单论文推理题 (Medium) - 30%                                        │
│  ─────────────────────────────────                                           │
│  • 问题: "如何在保证安全的前提下减少 bot 验证对用户体验的影响?"                   │
│  • 特点: 不包含明显关键词，需要语义理解                                         │
│  • 答案来源: Method + Evaluation sections                                    │
│  • 测试能力: 语义检索 + 理解方法论                                             │
│                                                                              │
│  Level 3: 跨论文比较题 (Hard) - 30%                                          │
│  ─────────────────────────────────                                           │
│  • 问题: "不同 side-channel 防御方法在性能和安全性上有什么权衡?"                 │
│  • 特点: 需要比较 2-3 篇相关论文，理解方法差异                                   │
│  • 答案来源: 多篇论文的 Method sections                                        │
│  • 测试能力: 多文档检索 + 比较分析                                             │
│                                                                              │
│  Level 4: 领域综述题 (Expert) - 20%                                          │
│  ─────────────────────────────────                                           │
│  • 问题: "当前安全研究中常用的用户研究方法论有哪些模式?"                          │
│  • 特点: 需要综合 3+ 篇论文，总结研究趋势                                       │
│  • 答案来源: 多篇论文的综合信息                                                │
│  • 测试能力: 全面检索 + 知识综合                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 论文聚类

为了生成高质量的跨论文问题（Level 3 和 Level 4），系统首先对论文进行聚类：

1. **关键词提取**: 从每篇论文的 title + abstract 提取关键词
2. **相似度计算**: 使用 Jaccard 相似度计算论文间的关联
3. **聚类分组**: 将相关论文分到同一组
4. **主题生成**: 使用 LLM 为每个聚类生成主题描述

```python
# 聚类示例
Cluster 0: "Side-channel attack detection"
  - Paper A: Timing attack detection with bounded error
  - Paper B: Hardware-based side-channel mitigation
  - Paper C: Constant-time implementation verification

Cluster 1: "User security behavior studies"  
  - Paper D: VPN user mental models
  - Paper E: Security tool usability evaluation
```

### 问题质量对比

**旧版问题 (V1)**:
```json
{
  "question": "What techniques can be used to detect timing vulnerabilities?",
  "expected_doc_ids": ["paper_a"],
  "difficulty": "medium"
}
```
❌ 问题：单论文、关键词明显、答案唯一

**新版问题 (V2)**:
```json
{
  "question": "How do different approaches to detecting timing vulnerabilities compare in terms of accuracy vs. performance overhead?",
  "expected_doc_ids": ["paper_a", "paper_b", "paper_c"],
  "difficulty": "hard",
  "is_multi_paper": true
}
```
✅ 改进：跨论文比较、需要综合理解、测试真实 RAG 能力

### 使用方法

```bash
# 使用新策略生成问题（默认 V2）
uv run python scripts/run_full_evaluation.py --generate-qa --num-questions 50

# 自定义难度分布
# 在代码中设置:
# difficulty_distribution = {
#     "easy": 0.1,    # 10% Level 1
#     "medium": 0.2,  # 20% Level 2
#     "hard": 0.4,    # 40% Level 3 (比较题)
#     "expert": 0.3   # 30% Level 4 (综述题)
# }
```

---

## 架构概览

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Evaluation Framework                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │ Data Preparation │    │  QA Generation  │    │   Runner     │ │
│  │                  │    │                 │    │              │ │
│  │ • DataExporter   │───▶│ • QAGenerator   │───▶│ • L1 Eval    │ │
│  │ • PDFLoader      │    │ • PaperClusterer│    │ • L2 Eval    │ │
│  │ • CollectionBuilder│  │ • Prompts V2    │    │ • L3 Eval    │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                      │                     │         │
│           ▼                      ▼                     ▼         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │ evaluation/data/ │    │ ground_truth.json│   │ reports/*.json│ │
│  │ • chunks/        │    │                 │    │              │ │
│  │ • pdfs/          │    │                 │    │              │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RAG Components                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │  RAG Provider   │    │   Embedding     │    │     LLM      │ │
│  │                 │    │                 │    │              │ │
│  │ • search_abstracts│  │ • qwen3-embedding│   │ • qwen3:8b   │ │
│  │ • search_by_section│ │ • dim=2560      │    │              │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 核心模块

| 模块 | 路径 | 职责 |
|------|------|------|
| **EvaluationRunner** | `src/evaluation/runner.py` | 执行 L1/L2/L3 评估，生成报告 |
| **QAGenerator** | `src/evaluation/qa_generation/qa_generator.py` | 4 级难度问题生成 |
| **PaperClusterer** | `src/evaluation/qa_generation/paper_clustering.py` | 论文聚类（用于跨论文问题） |
| **DataPreparationPipeline** | `src/evaluation/data_preparation/pipeline.py` | 数据导出、PDF处理、chunk保存 |
| **CollectionBuilder** | `src/evaluation/data_preparation/collection_builder.py` | 管理评估 collection，支持策略切换 |
| **EvaluationConfig** | `src/evaluation/config.py` | 配置管理（路径、策略、参数） |

---

## 评估指标详解

### L1: Paper Discovery（论文发现）

测试目标：给定一个问题，能否找到相关的论文？

**使用的 RAG 方法**: `search_abstracts(query, k=10)`

| 指标 | 公式 | 说明 |
|------|------|------|
| **Precision@K** | $P@K = \frac{\|Retrieved_K \cap Relevant\|}{K}$ | 返回的 K 个结果中有多少是相关的 |
| **Recall@K** | $R@K = \frac{\|Retrieved_K \cap Relevant\|}{\|Relevant\|}$ | 相关文档中有多少被返回 |
| **MRR** | $MRR = \frac{1}{N}\sum_{i=1}^{N}\frac{1}{rank_i}$ | 第一个相关结果的排名倒数的平均值 |
| **Hit Rate** | $HR = \frac{\|queries\ with\ hit\|}{\|queries\|}$ | 至少返回一个相关结果的查询比例 |

**代码实现**:
```python
# src/evaluation/runner.py

def run_l1_paper_discovery(self, qa_pairs: list[QAPair]) -> L1Result:
    for qa in valid_pairs:
        # 检索
        results = self.rag.search_abstracts(query=qa.question, k=10)
        retrieved_docs = [r["doc_id"] for r in results]
        expected_docs = set(qa.expected_doc_ids)
        
        # 计算指标
        p_at_5 = len(set(retrieved_docs[:5]) & expected_docs) / 5
        p_at_10 = len(set(retrieved_docs[:10]) & expected_docs) / 10
        r_at_k = len(set(retrieved_docs[:k]) & expected_docs) / len(expected_docs)
        
        # MRR: 找到第一个相关文档的位置
        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in expected_docs:
                mrr = 1.0 / rank
                break
```

---

### L2: Section Retrieval（章节检索）

测试目标：给定问题和目标论文，能否找到正确的 chunk？

**使用的 RAG 方法**: `search_by_section(query, doc_id, section_category, k=10)`

| 指标 | 公式 | 说明 |
|------|------|------|
| **Chunk Precision** | $P = \frac{\|Retrieved \cap Expected\|}{\|Retrieved\|}$ | 返回的 chunks 中有多少是期望的 |
| **Chunk Recall** | $R = \frac{\|Retrieved \cap Expected\|}{\|Expected\|}$ | 期望的 chunks 中有多少被返回 |
| **Method Precision** | 同上，限定 `section_category=2` | 方法论章节的检索精度 |
| **Eval Precision** | 同上，限定 `section_category=4` | 实验章节的检索精度 |

**代码实现**:
```python
# src/evaluation/runner.py

def run_l2_chunk_retrieval(self, qa_pairs: list[QAPair]) -> L2Result:
    for qa in valid_pairs:  # 只用有 expected_chunk_ids 的问题
        expected_chunks = set(qa.expected_chunk_ids)
        
        # 根据 answer_source 确定 section_category
        section_cat = self._source_to_category(qa.answer_source)
        
        # 检索
        results = self.rag.search_by_section(
            query=qa.question,
            doc_id=qa.expected_doc_ids[0],
            section_category=section_cat,
            k=10
        )
        
        retrieved_chunks = set(r["chunk_id"] for r in results)
        
        precision = len(expected_chunks & retrieved_chunks) / len(retrieved_chunks)
        recall = len(expected_chunks & retrieved_chunks) / len(expected_chunks)
```

---

### L3: End-to-End QA（端到端问答）

测试目标：完整的 RAG 流程能否生成正确的答案？

**流程**: 检索 → LLM 生成答案 → LLM 评判质量

| 指标 | 范围 | 说明 |
|------|------|------|
| **Correctness** | 0-1 | 答案是否正确（与参考答案对比） |
| **Faithfulness** | 0-1 | 答案是否忠实于检索内容（无幻觉） |
| **Relevance** | 0-1 | 答案是否直接回答问题 |
| **Easy/Medium/Hard Accuracy** | 0-1 | 按难度分类的正确率 |

**LLM-as-Judge 评分**:
```python
# src/evaluation/prompts/l3_evaluation.py

ANSWER_EVALUATION_PROMPT = """
评分标准：
1. Correctness (0-5): 答案是否包含正确信息？
2. Faithfulness (0-5): 答案是否基于检索内容？
3. Relevance (0-5): 答案是否直接回答问题？

输出 JSON: {"correctness": 4, "faithfulness": 5, "relevance": 4}
"""
```

**代码实现**:
```python
# src/evaluation/runner.py

def run_l3_end_to_end(self, qa_pairs: list[QAPair]) -> L3Result:
    for qa in qa_pairs:
        # 1. 检索
        context = self._retrieve_context(qa)
        
        # 2. 生成答案
        generated_answer = self._generate_answer(qa.question, context)
        
        # 3. LLM 评判
        scores = self._evaluate_answer(
            question=qa.question,
            generated_answer=generated_answer,
            reference_answer=qa.reference_answer,
            context=context
        )
        
        # 归一化到 0-1
        correctness = scores["correctness"] / 5.0
        faithfulness = scores["faithfulness"] / 5.0
```

---

## 对比实验配置

### 1. 切换 Index 类型

支持的索引类型：`FLAT`, `HNSW`, `IVF_FLAT`

```python
from evaluation.config import EvaluationConfig, ChunkStrategy
from evaluation.data_preparation.collection_builder import CollectionBuilder

config = EvaluationConfig()
builder = CollectionBuilder(config)

# 创建不同索引类型的 collection
builder.create_collection(
    strategy=ChunkStrategy.PARAGRAPH,
    index_type="HNSW",           # 或 "FLAT", "IVF_FLAT"
    drop_if_exists=True
)
```

**完整对比实验**:
```python
from evaluation.runner import EvaluationRunner
from rag.retriever import get_rag_client_by_provider
from settings import settings

results = {}
for index_type in ["FLAT", "HNSW", "IVF_FLAT"]:
    # 1. 重建 collection
    builder.create_collection(ChunkStrategy.PARAGRAPH, index_type=index_type)
    pipeline.rebuild_from_chunks(ChunkStrategy.PARAGRAPH)
    
    # 2. 运行评估
    with builder.use_chunk_strategy(ChunkStrategy.PARAGRAPH):
        rag_client = get_rag_client_by_provider(settings.rag_provider)
        runner = EvaluationRunner(rag_client=rag_client, config=config)
        report = runner.run_all()
        results[index_type] = report
```

---

### 2. 切换 Chunk 策略

支持的分块策略（定义在 `evaluation/config.py`）:

| 策略 | 说明 |
|------|------|
| `PARAGRAPH` | 按段落分块（默认） |
| `SENTENCE` | 按句子分块（更细粒度） |
| `CONTEXTUAL` | 带上下文前缀的分块 |

```python
from evaluation.config import ChunkStrategy
from rag.retriever import get_rag_client_by_provider
from settings import settings

# 使用不同策略
for strategy in [ChunkStrategy.PARAGRAPH, ChunkStrategy.SENTENCE]:
    with builder.use_chunk_strategy(strategy):
        rag_client = get_rag_client_by_provider(settings.rag_provider)
        runner = EvaluationRunner(rag_client=rag_client)
        report = runner.run_all()
```

**注意**: 不同策略需要分别准备数据：
```bash
# 为每种策略准备 chunks（会创建不同的 collection）
uv run python scripts/run_full_evaluation.py --prepare-only --sample 20
```

---

### 3. 切换 RAG 类型（Agentic vs Non-Agentic）

**Non-Agentic（当前实现）**: 直接向量检索
```python
from rag.retriever import get_rag_client_by_provider
from settings import settings

rag_client = get_rag_client_by_provider(settings.rag_provider)
# 直接调用当前 RAG Provider
results = rag_client.search_abstracts(query, k=10)
```

**Agentic RAG（扩展）**: 需要实现 AgenticSearcher
```python
# 使用 Agent 进行多步推理检索
from agents.searcher import AgenticSearcher

class AgenticRAG:
    def __init__(self, rag_client, llm):
        self.rag_client = rag_client
        self.llm = llm
        self.searcher = AgenticSearcher(rag_client, llm)
    
    def search_abstracts(self, query: str, k: int = 10):
        # Agent 决定如何检索
        return self.searcher.search(query, k)
```

**对比方法**:
```python
# Non-Agentic
runner_basic = EvaluationRunner(rag_client=rag_client)

# Agentic  
agentic_rag = AgenticRAG(rag_client, llm)
runner_agentic = EvaluationRunner(rag_client=agentic_rag)

# 对比
report_basic = runner_basic.run_all()
report_agentic = runner_agentic.run_all()
```

---

### 4. 完整 12 配置对比实验

```python
"""
12 = 2 (chunk) × 3 (index) × 2 (agentic)
"""

from itertools import product
from rag.retriever import get_rag_client_by_provider
from settings import settings

chunk_strategies = [ChunkStrategy.PARAGRAPH, ChunkStrategy.SENTENCE]
index_types = ["FLAT", "HNSW", "IVF_FLAT"]
agentic_modes = [False, True]

all_results = []

for chunk, index, agentic in product(chunk_strategies, index_types, agentic_modes):
    config_name = f"{chunk.value}_{index}_{'agentic' if agentic else 'basic'}"
    
    # 1. 准备 collection
    builder.create_collection(chunk, index_type=index, drop_if_exists=True)
    pipeline.rebuild_from_chunks(chunk)
    
    # 2. 选择 RAG 客户端
    with builder.use_chunk_strategy(chunk):
        base_rag = get_rag_client_by_provider(settings.rag_provider)
        rag_client = AgenticRAG(base_rag, llm) if agentic else base_rag
        
        # 3. 运行评估
        runner = EvaluationRunner(rag_client=rag_client, llm_client=llm)
        report = runner.run_all()
        
        all_results.append({
            "config": config_name,
            "l1_mrr": report.l1_paper_discovery.mrr,
            "l1_hit_rate": report.l1_paper_discovery.hit_rate,
            "l2_precision": report.l2_section_retrieval.overall_precision,
            "l3_accuracy": report.l3_end_to_end.overall_accuracy,
        })

# 输出对比表
import pandas as pd
df = pd.DataFrame(all_results)
print(df.to_markdown())
```

---

## 数据流程

### 目录结构

```
evaluation/
├── data/
│   ├── papers_source.jsonl    # 从业务库导出的论文元数据
│   ├── pdfs/                  # 下载的 PDF 文件（缓存）
│   ├── chunks/
│   │   ├── paragraph/         # PARAGRAPH 策略的 chunks
│   │   │   ├── {doc_id}.json
│   │   │   └── ...
│   │   └── sentence/          # SENTENCE 策略的 chunks
│   ├── ground_truth.json      # 生成的 QA pairs
│   └── reports/
│       └── report_{id}_{timestamp}.json
```

### Chunk 文件格式

```json
{
  "doc_id": "db31bfbb-b56a-4547-8c2f-5ae2b2466c52",
  "title": "Security at the End of the Tunnel...",
  "strategy": "paragraph",
  "chunks": [
    {
      "chunk_index": 0,
      "chunk_text": "We present a qualitative study...",
      "section_title": "Abstract",
      "section_category": 0,
      "contextual_prefix": "This paper discusses VPN security..."
    }
  ]
}
```

### Ground Truth 格式

```json
{
  "version": "1.0",
  "created_at": "2025-11-26T21:29:18",
  "total_papers": 8,
  "difficulty_distribution": {"easy": 2, "medium": 2, "hard": 1},
  "qa_pairs": [
    {
      "id": 1,
      "question": "Which paper discusses corporate VPN mental models?",
      "difficulty": "easy",
      "expected_doc_ids": ["db31bfbb-..."],
      "expected_chunk_ids": null,
      "answer_source": "abstract",
      "reference_answer": "The paper 'Security at the End...' discusses..."
    }
  ]
}
```

### 评估报告格式

```json
{
  "run_id": "7c015e51",
  "run_at": "2025-11-27T11:00:21",
  "total_qa_pairs": 5,
  "rag_provider": "sqlite",
  "embedding_model": "qwen3-embedding:4b",
  "l1_paper_discovery": {
    "precision_at_5": 0.24,
    "precision_at_10": 0.12,
    "recall_at_5": 1.0,
    "recall_at_10": 1.0,
    "mrr": 1.0,
    "hit_rate": 1.0,
    "mean_latency_ms": 695.1
  },
  "l2_section_retrieval": {
    "overall_precision": 0.25,
    "overall_recall": 0.5,
    "method_precision": 0.25
  },
  "l3_end_to_end": {
    "easy_accuracy": 1.0,
    "medium_accuracy": 0.6,
    "hard_accuracy": 1.0,
    "overall_accuracy": 0.84,
    "faithfulness": 1.0
  }
}
```

---

## API 参考

### EvaluationRunner

```python
class EvaluationRunner:
    def __init__(
        self,
        rag_client: RAG,                    # 任意兼容 RAG provider
        llm_client: BaseChatModel = None,   # L3 评估需要
        config: EvaluationConfig = None
    ):
        ...
    
    def run_all(self, ground_truth: GroundTruth = None) -> EvaluationReport:
        """运行完整评估（L1 + L2 + L3）"""
    
    def run_l1_paper_discovery(self, qa_pairs: list[QAPair]) -> L1Result:
        """L1: 论文发现评估"""
    
    def run_l2_chunk_retrieval(self, qa_pairs: list[QAPair]) -> L2Result:
        """L2: Chunk 检索评估"""
    
    def run_l3_end_to_end(self, qa_pairs: list[QAPair]) -> L3Result:
        """L3: 端到端 QA 评估"""
    
    def load_ground_truth(self) -> GroundTruth:
        """加载 Ground Truth"""
    
    def save_report(self, report: EvaluationReport) -> Path:
        """保存评估报告"""
```

### CollectionBuilder

```python
class CollectionBuilder:
    def create_collection(
        self,
        strategy: ChunkStrategy,
        index_type: str = "FLAT",
        drop_if_exists: bool = False
    ):
        """创建评估 collection"""
    
    @contextmanager
    def use_chunk_strategy(self, strategy: ChunkStrategy):
        """Context Manager: 临时切换到指定策略的 collection"""
        # 用法:
        # with builder.use_chunk_strategy(ChunkStrategy.PARAGRAPH):
        #     rag_client = get_rag_client_by_provider(settings.rag_provider)  # 使用切换后的表/collection
```

### QAGenerator

```python
class QAGenerator:
    def generate(
        self,
        strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH,
        num_questions: int = 50,
        difficulty_distribution: dict = {"easy": 0.4, "medium": 0.4, "hard": 0.2}
    ) -> GroundTruth:
        """生成 QA pairs"""
    
    def save_ground_truth(self, ground_truth: GroundTruth) -> Path:
        """保存到文件"""
```

### DataPreparationPipeline

```python
class DataPreparationPipeline:
    def run(
        self,
        strategies: list[ChunkStrategy] = None,
        sample_size: int = None,
        drop_existing: bool = False
    ) -> PipelineResult:
        """运行完整数据准备流程"""
    
    def rebuild_from_chunks(
        self,
        strategy: ChunkStrategy,
        drop_existing: bool = True
    ) -> int:
        """从已有 chunks 文件重建 collection（快速）"""
```

---

## 最佳实践

### 1. 快速迭代测试

```bash
# 使用小样本快速验证
uv run python scripts/run_full_evaluation.py --full --sample 10 --num-questions 15 --no-l3
```

### 2. 只更新 QA 不重新处理 PDF

```bash
# 使用已有 chunks 重新生成问题
uv run python scripts/run_full_evaluation.py --generate-qa --num-questions 50
```

### 3. 对比不同索引效果

```python
# 使用 rebuild_from_chunks 快速切换索引
for index_type in ["FLAT", "HNSW"]:
    builder.create_collection(ChunkStrategy.PARAGRAPH, index_type=index_type)
    pipeline.rebuild_from_chunks(ChunkStrategy.PARAGRAPH)
    # 运行评估...
```

### 4. 查看历史报告

```bash
ls evaluation/data/reports/
cat evaluation/data/reports/report_xxx.json | jq .
```

---

## 常见问题

### Q: L3 评估为什么这么慢？
A: L3 需要对每个问题调用两次 LLM（生成答案 + 评判质量）。可以用 `--no-l3` 跳过。

### Q: 如何增加测试问题数量？
A: 运行 `--generate-qa --num-questions 100`，会覆盖现有的 ground_truth.json。

### Q: 评估 collection 和业务 collection 会冲突吗？
A: 不会。评估使用独立的 collection（如 `papers_eval_paragraph`），通过 `CollectionBuilder.use_chunk_strategy()` 临时切换。

### Q: 如何添加新的评估指标？
A: 在 `src/evaluation/schemas.py` 添加字段，在 `src/evaluation/runner.py` 实现计算逻辑。
