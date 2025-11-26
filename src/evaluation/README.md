# RAG Evaluation Framework

## 概述

本模块用于评估 RAG 系统的检索和生成质量，支持多种配置对比实验。

## 设计原则

1. **复用主逻辑代码**：通过修改 `settings` 切换 collection，复用 `MilvusProvider`、`PDFLoader` 等
2. **Contextual Chunking 与 Annotation 合并**：在生成 context_prefix 时保存到本地，QA 生成时复用
3. **支持 Chunk-level 评估**：Ground Truth 可以精确到 chunk 级别

## 架构

```
evaluation/
├── __init__.py
├── README.md
├── config.py                 # 配置定义
├── schemas.py                # 数据结构定义
├── pipeline.py               # 完整评估流水线
├── runner.py                 # 评估执行器
├── rag_evaluator.py          # 评估指标计算
│
├── data_preparation/         # 数据准备模块
│   ├── __init__.py
│   ├── data_exporter.py      # 从业务库导出数据
│   ├── pdf_downloader.py     # PDF 下载器
│   ├── chunk_processor.py    # 分块处理器 (paragraph/contextual)
│   ├── collection_builder.py # Collection 构建器
│   ├── pipeline.py           # 数据准备流水线
│   └── prompts.py            # Contextual Chunking prompt
│
├── annotation/               # 论文标注模块 (Paper-level)
│   ├── __init__.py
│   ├── paper_annotator.py    # Paper-level 标注
│   └── prompts.py            # 标注 prompt 模板
│
└── qa_generation/            # QA 生成模块
    ├── __init__.py
    ├── qa_generator.py       # QA pairs 生成器 (利用 context_prefix)
    └── prompts.py            # QA 生成 prompt 模板
```

## 实验设计

### 变量

| 变量 | 选项 | 隔离方式 |
|------|------|----------|
| Chunk 策略 | paragraph / contextual | Collection 级别 |
| Index 策略 | FLAT / HNSW / IVF | Index 级别（rebuild） |
| Agentic RAG | 开 / 关 | 代码逻辑 |

### Collection 切换方式

**通过修改 settings 切换 collection，复用 MilvusProvider：**

```python
from settings import settings

# 方式 1: 直接修改
settings.milvus_collection = "papers_eval_paragraph"
rag = MilvusProvider()

# 方式 2: Context Manager (推荐)
with eval_collection("papers_eval_contextual"):
    rag = MilvusProvider()
```

### Collection 设计

```
papers                       # 业务库 (不动)
papers_eval_paragraph        # 评估: 传统段落分块
papers_eval_contextual       # 评估: Contextual Chunking
```

### 评估矩阵 (12 种实验)

| Chunk | Index | Agentic | 实验名 |
|-------|-------|---------|--------|
| paragraph | FLAT | ❌ | para_flat |
| paragraph | FLAT | ✅ | para_flat_agentic |
| paragraph | HNSW | ❌ | para_hnsw |
| paragraph | HNSW | ✅ | para_hnsw_agentic |
| paragraph | IVF | ❌ | para_ivf |
| paragraph | IVF | ✅ | para_ivf_agentic |
| contextual | FLAT | ❌ | ctx_flat |
| contextual | FLAT | ✅ | ctx_flat_agentic |
| contextual | HNSW | ❌ | ctx_hnsw |
| contextual | HNSW | ✅ | ctx_hnsw_agentic |
| contextual | IVF | ❌ | ctx_ivf |
| contextual | IVF | ✅ | ctx_ivf_agentic |

## 数据流

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 数据准备                                            │
│                                                              │
│   1.1 从业务库导出论文元数据                                 │
│       → evaluation/data/papers_source.jsonl                 │
│                                                              │
│   1.2 下载 PDF                                               │
│       → evaluation/data/pdfs/{doc_id}.pdf                   │
│                                                              │
│   1.3 分块处理                                               │
│       Paragraph: 直接分块                                    │
│       Contextual: 每个 chunk → LLM → context_prefix         │
│       → evaluation/data/chunks/{strategy}/{doc_id}.jsonl    │
│                                                              │
│   1.4 入库 (通过修改 settings 切换 collection)               │
│       → papers_eval_{strategy} collections                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Paper-level 标注 (可选)                              │
│                                                              │
│   每篇论文: title + abstract → LLM → summary, keywords       │
│   → evaluation/data/papers_summaries.jsonl                  │
│                                                              │
│   注: 主要用于按 research_area 分组生成 QA                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: QA 生成 (利用 context_prefix)                        │
│                                                              │
│   输入给 LLM:                                                │
│   - Paper title + abstract                                   │
│   - 所有 chunks 的 context_prefix 列表                       │
│                                                              │
│   输出 Ground Truth:                                         │
│   - Paper-level QA: expected_doc_ids                        │
│   - Chunk-level QA: expected_doc_ids + expected_chunk_ids   │
│                                                              │
│   → evaluation/data/ground_truth.json                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 评估                                                │
│                                                              │
│   对 12 种实验配置:                                          │
│   1. 修改 settings 切换 collection                          │
│   2. rebuild_index (FLAT/HNSW/IVF)                          │
│   3. 运行检索 (basic/agentic)                                │
│   4. 计算 L1/L2/L3 指标                                      │
│                                                              │
│   → evaluation/data/reports/report_{timestamp}.json         │
└─────────────────────────────────────────────────────────────┘
```

## Contextual Chunking 与 QA 生成的结合

### 核心思路

Contextual Chunking 生成的 `context_prefix` 本身就是对 chunk 的描述，
在 QA 生成时直接复用，无需额外的 chunk-level annotation。

### 数据流示例

**Step 1.3 保存的 chunk 数据：**
```jsonl
{"chunk_index": 0, "context_prefix": "This is the abstract introducing...", "original_text": "..."}
{"chunk_index": 1, "context_prefix": "This chunk describes background...", "original_text": "..."}
{"chunk_index": 5, "context_prefix": "This chunk presents the core algorithm...", "original_text": "..."}
{"chunk_index": 12, "context_prefix": "This chunk shows results on dataset X...", "original_text": "..."}
```

**Step 3 QA 生成时的输入：**
```
Paper: "Attention Is All You Need"
Abstract: "We propose the Transformer..."

Chunk Contexts:
[0] This is the abstract introducing the Transformer architecture...
[1] This chunk describes the background of sequence modeling...
[5] This chunk presents the multi-head attention mechanism...
[12] This chunk shows BLEU scores on WMT translation...
```

**Step 3 QA 生成的输出：**
```json
[
  {
    "question": "What is the core innovation of the Transformer?",
    "difficulty": "medium",
    "expected_doc_ids": ["doc_123"],
    "expected_chunk_ids": [5],
    "reference_answer": "Multi-head self-attention mechanism"
  },
  {
    "question": "What BLEU score did Transformer achieve?",
    "difficulty": "medium", 
    "expected_doc_ids": ["doc_123"],
    "expected_chunk_ids": [12],
    "reference_answer": "28.4 on WMT En-De"
  }
]
```

## 评估指标

### L1: Paper Discovery
- Precision@K, Recall@K
- MRR (Mean Reciprocal Rank)
- Hit Rate
- Latency

### L2: Section/Chunk Retrieval
- Chunk Precision: 检索到的 chunk 是否是期望的
- Chunk Recall: 期望的 chunk 是否被检索到
- Section Precision/Recall: 按 section_category 统计

### L3: End-to-End QA
- Accuracy (by difficulty: easy/medium/hard)
- Answer Relevance
- Faithfulness

## LLM 调用统计

| 阶段 | 调用次数 | 说明 |
|------|---------|------|
| Contextual Chunking | 20,000 | 500 篇 × 40 chunks |
| Paper Annotation | 500 | 每篇论文一次 (可选) |
| QA Generation | ~20 | 分批生成，每批 5-10 篇 |
| **总计** | ~20,500 | |

## 使用方式

```python
from evaluation.pipeline import EvaluationPipeline
from evaluation.config import EvaluationConfig

# 配置
config = EvaluationConfig(
    sample_size=100,  # 抽样 100 篇论文
    num_qa_pairs=50,  # 生成 50 个问题
)

# 初始化
pipeline = EvaluationPipeline(
    source_rag_client=rag,
    llm_client=llm,
    config=config
)

# 运行完整流程
report = pipeline.run_full_pipeline()

# 或分步运行
papers = pipeline.step1_prepare_data()
annotations = pipeline.step2_annotate_papers(papers)  # 可选
ground_truth = pipeline.step3_generate_qa(annotations)
report = pipeline.step4_run_evaluation(ground_truth)
```

## 数据目录结构

```
evaluation/data/
├── papers_source.jsonl       # 导出的论文元数据
├── pdfs/                     # 下载的 PDF 文件
│   └── {doc_id}.pdf
├── chunks/                   # 分块缓存 (含 context_prefix)
│   ├── paragraph/
│   │   └── {doc_id}.jsonl
│   └── contextual/
│       └── {doc_id}.jsonl
├── papers_summaries.jsonl    # Paper-level 标注 (可选)
├── ground_truth.json         # QA 测试集 (含 chunk-level)
└── reports/                  # 评估报告
    └── report_{timestamp}.json
```
