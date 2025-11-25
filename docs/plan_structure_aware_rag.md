# Structure-Aware Agentic RAG Implementation Plan (v2.0)

## 1. 目标 (Objective)
构建一个能够感知论文结构的 **Agentic RAG** 系统。
核心能力：
1.  **Structure Parsing**: 入库时保留章节层级 (Hierarchy) 和类别 (Category)。
2.  **Router & Scope**: 检索时根据问题意图，自动判断是“查单篇”还是“查全库”。
3.  **Context Reconstruction**: 利用结构索引，重组碎片化的 Chunk，提供包含背景(Intro)和前后文(Window)的完整上下文。

## 2. 数据模型设计 (Data Schema)

### 2.1 章节分类枚举 (Section Categories)
保持不变，作为硬过滤条件。

| ID | Category Name | Keywords (Regex) | Note |
| :--- | :--- | :--- | :--- |
| 0 | **Abstract** | `abstract` | 摘要 |
| 1 | **Introduction** | `introduction`, `background`, `motivation` | **高权重背景** |
| 2 | **Method** | `method`, `approach`, `model`, `architecture`, `framework` (or Title Keywords) | 技术细节 |
| 3 | **Evaluation** | `experiment`, `result`, `evaluation`, `ablation`, `comparison` | 数据支撑 |
| 4 | **Conclusion** | `conclusion`, `discussion`, `summary` | 总结 |
| 5 | **Other** | (Default) | 参考文献/附录等 |
| 6 | **Related Work** | `related work` | 相关工作 |

### 2.2 Milvus Collection Schema (Updated for Existing Codebase)
**变更点**：增加 `chunk_index`，复用现有字段名。

| Field Name | Type | Description | Note |
| :--- | :--- | :--- | :--- |
| `id` | INT64 | 主键 (Auto ID) | **Existing**. Milvus 内部唯一ID |
| `doc_id` | VARCHAR | 论文唯一标识 (UUID) | **Existing**. 用于关联同一篇论文 |
| `title` | VARCHAR | 论文标题 | **Existing**. 用于引用和展示 |
| `conference_year` | INT16 | 发表年份 | **Existing**. 用于时间过滤 |
| `chunk_id` | INT64 | Chunk ID | **Existing**. 用于唯一标识 Chunk |
| `chunk_index` | INT32 | 全文段落序号 (0, 1, 2...) | **[Critical New]** 用于检索前后文 (Sliding Window) |
| `text` | VARCHAR | 文本内容 (Chunk) | **Existing** (was text_field) |
| `vector` | FLOAT_VECTOR | 文本向量 | **Existing** |
| `section_title` | VARCHAR | 原始章节标题 | **[New]** e.g., "4.1 Ablation Study" |
| `parent_section` | VARCHAR | 父章节标题 | **[New]** e.g., "4. Experiments" |
| `section_category`| INT8 | 枚举 ID | **[New]** 核心过滤字段 |
| `hierarchy_level`| INT8 | 层级深度 | **[New]** 0=大章, 1=小节 |
| `page_number` | INT16 | 起始页码 | **[New]** |

---

## 3. 模块改造计划 (Module Modifications)

### 3.1 PDF Parser (`src/parser/pdf_parser.py`)
**任务：** 增强解析的鲁棒性和元数据丰富度。

1.  **增强 `SectionClassifier`**:
    *   **Level 1**: 正则匹配 (Regex)。
    *   **Level 2 (Fallback)**: 如果正则失败（返回 Other），但标题长度 < 10 个词，标记为 `Unclassified_Header`，后续处理或丢弃，防止标题混入正文。
2.  **修改 `flatten_pdf_tree`**:
    *   增加计数器 `global_chunk_idx = 0`。
    *   在遍历生成 List 时，为每个 Chunk 赋值 `chunk_index` 并自增。
    *   提取 PDF 第一页的大标题作为 `title` (如果外部未提供)。

### 3.2 Database Layer (`src/rag/milvus.py`)
**任务：** 支持 Metadata 存储和窗口查询。

1.  **Update Schema**: 修改 `_create_schema` 方法，应用 2.2 中的新字段。
2.  **新增 `insert_paper_chunks`**: 批量插入带有结构信息的 Chunks。
3.  **新增 `get_context_window(doc_id, center_chunk_index, window_size=1)`**:
    *   **功能**：给定一个 Chunk 的 ID 和 Index，查出它前后的 Chunk。
    *   **SQL逻辑**：`doc_id == "{doc_id}" && chunk_index in [{idx-1}, {idx}, {idx+1}]`
    *   **用途**：当检索到的片段被截断时，自动补全上下文。

### 3.3 Agentic Retrieval Logic (Core)
**任务：** 实现分流检索策略 (Router -> Retriever)。

#### Step 1: Intent Routing (意图识别)
使用 LLM (Lite model) 分析用户 Query，输出 JSON：
```json
{
  "intent": "SPECIFIC_PAPER" | "TOPIC_SURVEY",
  "target_paper_keywords": ["Attention is all you need", "Transformer"], // if SPECIFIC
  "target_section_category": [2, 3] // e.g. 用户问实验数据 -> [3]
}
```

#### Step 2: Dual-Track Retrieval (双轨检索)

*   **Track A: Specific Paper Mode (查单篇)**
    1.  **Paper Location**: 通过 `target_paper_keywords` 在 `Category=0 (Abstract)` 中进行标量或向量搜索，锁定唯一的 `target_doc_id`。
    2.  **Scoped Search**:
        *   `filter`: `doc_id == target_doc_id` AND `section_category IN target_section_category`。
        *   `search`: 使用 Query 向量搜索 Top-K Chunks。
        *   *优势：在极小的范围内搜索，准确率极高，不会幻觉到其他论文。*

*   **Track B: Topic Survey Mode (查领域/跨篇)**
    1.  **Coarse Search (Abstract)**: 向量搜索 `Category=0`，获取 Top-5 最相关的论文 `doc_ids`。
    2.  **Fine Search (Global with Scope)**:
        *   `filter`: `doc_id IN [doc_ids_from_step_1]` (限制在相关论文范围内，防止噪音)。
        *   `search`: 全文向量搜索 Top-K Chunks。

#### Step 3: Context Assembly & Expansion (上下文组装)
对 Step 2 拿到的 `Candidate Chunks` 进行处理：
1.  **Window Expansion**: 对每个 Candidate，调用 `get_context_window` 补全前后段落，合并成 `Extended Chunk`。
2.  **Background Injection**: 必须查出该 Candidate 所属论文的 **Introduction (Category=1)** 的前 500 tokens。
3.  **Final Prompt Structuring**:
    ```text
    [Paper: Attention Is All You Need]
    > Context (Introduction): ...
    > Retrieved Evidence (Method): ... (expanded text) ...
    ```

---

## 4. 执行步骤 (Action Plan)

1.  **Phase 1: Infrastructure (1-2 Days)**
    *   修改 `pdf_parser.py`：实现 Section 归一化和 `chunk_index` 计数。
    *   修改 `milvus.py`：重建 Collection，实现 `insert_paper_chunks` 和 `get_context_window`。
    *   **Test**: 解析一篇论文，检查数据库里的 `chunk_index` 是否连续，`category` 是否正确。

2.  **Phase 2: Basic Retrieval (1 Day)**
    *   实现不带 Router 的基础逻辑：`Query -> Abstract Search -> Top Doc IDs -> Scoped Chunk Search` (即 Track B 的逻辑)。
    *   这是最通用的 Baseline。

3.  **Phase 3: Agentic Upgrade (2 Days)**
    *   实现 `Query Router` (简单的 Prompt 工程)。
    *   实现 Track A (Specific Mode) 的逻辑。
    *   集成 Context Window 补全功能。

4.  **Phase 4: Evaluation**
    *   构造两个典型测试用例：
        1.  "Llama 3 的训练数据有多少？" (应触发 Track A，精准定位)
        2.  "最近有哪些提升 RAG 准确率的方法？" (应触发 Track B，跨篇总结)

---

## 5. Agentic RAG 详细设计 (Phase 3 Deep Dive)

### 5.1 问题：当前实现的局限性
目前的"Agentic RAG"只是让 LLM 对 Query 做一次 Rewrite，本质上仍是 **Single-Shot Retrieval**：
```
Query -> (LLM Refine) -> Search -> Return
```
这**不是真正的 Agentic**，因为：
- Agent 没有观察检索结果的能力
- Agent 无法根据结果决定下一步行动
- Agent 不能进行多轮迭代直到满意

### 5.2 真正的 Agentic RAG 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agentic Searcher (LLM Agent)                │
│                                                                 │
│  System Prompt: You are a retrieval agent...                    │
│                                                                 │
│  Available Tools:                                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. search_abstracts(query, k=5)                         │   │
│  │    - 在 Abstract 层面搜索，返回最相关的论文列表         │   │
│  │    - 用于：快速定位目标论文、获取领域概览               │   │
│  │                                                         │   │
│  │ 2. search_by_section(query, doc_id?, category?, k=5)    │   │
│  │    - 在特定章节类型中搜索                               │   │
│  │    - category: METHOD/EVALUATION/INTRODUCTION/etc       │   │
│  │    - doc_id: 可选，限定在某篇论文内搜索                 │   │
│  │    - 用于：深入查找技术细节、实验结果                   │   │
│  │                                                         │   │
│  │ 3. get_context_window(doc_id, chunk_id, window=1)       │   │
│  │    - 获取某个 Chunk 的前后文                            │   │
│  │    - 用于：当片段被截断时，补全上下文                   │   │
│  │                                                         │   │
│  │ 4. get_paper_introduction(doc_id)                       │   │
│  │    - 获取某篇论文的 Introduction 段落                   │   │
│  │    - 用于：理解论文背景，避免断章取义                   │   │
│  │                                                         │   │
│  │ 5. finish(answer, sources)                              │   │
│  │    - 结束检索，返回最终答案和引用                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Agent Loop:                                                    │
│  while not finished:                                            │
│      action = LLM.decide(history, tools)                        │
│      observation = execute(action)                              │
│      history.append(observation)                                │
│      if action == "finish": break                               │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Agent Prompt 设计

```markdown
# Role
You are a retrieval agent for an academic paper database. Your goal is to find 
the most relevant and accurate information to answer the user's question.

# Strategy
1. **Understand Intent**: Is the user asking about a specific paper, or a general topic?
2. **Start Broad**: Use `search_abstracts` first to identify relevant papers.
3. **Go Deep**: Once you identify target papers, use `search_by_section` to find 
   specific evidence (e.g., METHOD for "how", EVALUATION for "results").
4. **Expand Context**: If a retrieved chunk seems incomplete, use `get_context_window` 
   to see surrounding paragraphs.
5. **Iterate**: If initial results are insufficient, try different queries or sections.
6. **Cite Sources**: Always note which doc_id and chunk you're referencing.

# IMPORTANT
- Do NOT answer from your own knowledge. Only use retrieved content.
- If you cannot find relevant information after 3 search attempts, say so explicitly.
- Prefer METHOD sections for "how does X work" questions.
- Prefer EVALUATION sections for "what are the results" questions.
- RELATED_WORK sections describe OTHER papers, not the current paper's contribution.
```

### 5.4 典型执行流程示例

**User Query**: "RAG系统中，检索文档的数量对回答质量有什么影响？"

```
Agent Thought: 这是一个关于 RAG 系统实验结果的问题，我需要找实验相关的章节。

Action: search_abstracts("RAG retrieval document number quality impact")
Observation: 
  - [doc_1] "The Power of Noise: Redefining Retrieval..."  
  - [doc_2] "Lost in the Middle: How Language Models..."

Agent Thought: 找到了相关论文，doc_2 看起来最相关。让我查看它的实验部分。

Action: search_by_section(
    query="document number retrieval quality", 
    doc_id="doc_2", 
    category=EVALUATION
)
Observation:
  - [chunk_42] "We find that model performance degrades significantly when 
    relevant information appears in the middle of long contexts..."

Agent Thought: 这个片段直接回答了问题，但我想看看更多上下文。

Action: get_context_window(doc_id="doc_2", chunk_id=42, window=1)
Observation:
  - [chunk_41] "Figure 3 shows the accuracy as a function of..."
  - [chunk_42] "We find that model performance degrades..."
  - [chunk_43] "This suggests that simply increasing the number..."

Agent Thought: 现在我有足够的信息来回答了。

Action: finish(
    answer="根据 Lost in the Middle 这篇论文的研究，检索文档数量增加会导致...",
    sources=["doc_2: chunk_41-43"]
)
```

### 5.5 实现路径

1. **定义 Tools**：在 `searcher.py` 中定义 `@tool` 装饰的函数
2. **创建 Agent**：使用 `langchain.agents.create_agent` 绑定 Tools
3. **设计 Prompt**：创建 `src/prompts/agentic_searcher.md`
4. **迭代限制**：设置 `max_iterations=5` 防止无限循环
5. **降级策略**：如果 `enable_agentic_rag=False`，使用原有的单次检索逻辑
