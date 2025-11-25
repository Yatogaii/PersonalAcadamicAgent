# RAG Evaluation Framework

## 概述

本模块用于评估 RAG 系统的检索和生成质量。

## 架构

```
evaluation/
├── __init__.py
├── README.md
├── rag_evaluator.py          # 评估指标计算 (已实现)
│
├── annotation/               # 阶段 1: LLM 标注
│   ├── __init__.py
│   ├── paper_annotator.py    # Paper-level 标注器
│   ├── section_annotator.py  # Section-level 标注器
│   └── prompts.py            # 标注用的 prompt 模板
│
├── qa_generation/            # 阶段 2: QA 生成
│   ├── __init__.py
│   ├── qa_generator.py       # QA pairs 生成器
│   └── prompts.py            # QA 生成的 prompt 模板
│
├── runner.py                 # 阶段 3: 评估执行器
│
└── schemas.py                # 数据结构定义
```

## 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│ 阶段 1: 标注 (Annotation)                                    │
│                                                              │
│   PaperAnnotator.annotate_all()                             │
│     → 生成 saved_summaries/papers_index.jsonl               │
│                                                              │
│   SectionAnnotator.annotate_loaded_papers()                 │
│     → 更新 papers_index.jsonl 的 section 字段               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段 2: QA 生成 (QA Generation)                              │
│                                                              │
│   QAGenerator.generate()                                     │
│     → 生成 evaluation/ground_truth.json                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段 3: 评估 (Evaluation)                                    │
│                                                              │
│   EvaluationRunner.run_all()                                │
│     → 输出评估报告                                           │
└─────────────────────────────────────────────────────────────┘
```

## 使用方式

```python
# 阶段 1: 标注
from evaluation.annotation import PaperAnnotator, SectionAnnotator

annotator = PaperAnnotator(rag_client, llm_client)
annotator.annotate_all()  # 标注所有论文

section_annotator = SectionAnnotator(rag_client, llm_client)
section_annotator.annotate_loaded_papers()  # 标注已加载 PDF 的论文

# 阶段 2: 生成 QA
from evaluation.qa_generation import QAGenerator

generator = QAGenerator(llm_client)
generator.generate(num_questions=100)

# 阶段 3: 运行评估
from evaluation.runner import EvaluationRunner

runner = EvaluationRunner(rag_client)
report = runner.run_all()
runner.print_report(report)
```

## 数据文件

- `saved_summaries/papers_index.jsonl` - 论文标注结果
- `evaluation/ground_truth.json` - QA 测试集
- `evaluation/reports/` - 评估报告输出目录
