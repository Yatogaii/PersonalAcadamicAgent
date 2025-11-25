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
