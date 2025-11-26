# RAG Evaluation Framework

## 概述

本模块用于评估 RAG 系统的检索和生成质量，支持多种配置对比实验。

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
├── annotation/               # 论文标注模块
│   ├── __init__.py
│   ├── paper_annotator.py    # Paper-level 标注
│   ├── section_annotator.py  # Section-level 标注
│   └── prompts.py            # 标注 prompt 模板
│
└── qa_generation/            # QA 生成模块
    ├── __init__.py
    ├── qa_generator.py       # QA pairs 生成器
    └── prompts.py            # QA 生成 prompt 模板
```

## 实验设计

### 变量

| 变量 | 选项 | 隔离方式 |
|------|------|----------|
| Chunk 策略 | paragraph / contextual | Collection 级别 |
| Index 策略 | FLAT / HNSW / IVF | Index 级别（rebuild） |
| Agentic RAG | 开 / 关 | 代码逻辑 |

### Collection 设计

```
papers_eval_paragraph    # 传统段落分块
papers_eval_contextual   # Contextual Chunking
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
│   业务库 → 导出 → 下载 PDF → 分块 → 入库                     │
│   输出: papers_eval_* collections                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 论文标注                                            │
│   LLM 为每篇论文生成 summary, keywords, research_area       │
│   输出: evaluation/data/papers_summaries.jsonl              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: QA 生成                                             │
│   基于标注生成 100 个 QA pairs (Easy/Medium/Hard)           │
│   输出: evaluation/data/ground_truth.json                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 评估                                                │
│   对 12 种实验配置分别运行评估                               │
│   输出: evaluation/data/reports/                            │
└─────────────────────────────────────────────────────────────┘
```

## 评估指标

### L1: Paper Discovery
- Precision@K, Recall@K
- MRR (Mean Reciprocal Rank)
- Hit Rate
- Latency

### L2: Section Retrieval
- Method Precision/Recall
- Evaluation Precision/Recall

### L3: End-to-End QA
- Accuracy (by difficulty)
- Answer Relevance
- Faithfulness

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
annotations = pipeline.step2_annotate_papers(papers)
ground_truth = pipeline.step3_generate_qa(annotations)
report = pipeline.step4_run_evaluation(ground_truth)
```

## 数据目录

```
evaluation/data/
├── papers_source.jsonl       # 导出的论文元数据
├── pdfs/                     # 下载的 PDF 文件
│   └── {doc_id}.pdf
├── chunks/                   # 分块缓存
│   ├── paragraph/
│   │   └── {doc_id}.jsonl
│   └── contextual/
│       └── {doc_id}.jsonl
├── papers_summaries.jsonl    # 论文标注结果
├── ground_truth.json         # QA 测试集
└── reports/                  # 评估报告
    └── report_{timestamp}.json
```
