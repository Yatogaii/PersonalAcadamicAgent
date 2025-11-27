"""
L3 End-to-End Evaluation Prompts

用于生成答案和评判答案质量的 prompt
"""

# 基于检索内容生成答案的 prompt
ANSWER_GENERATION_PROMPT = """You are a helpful research assistant. Based on the retrieved content from academic papers, answer the following question.

**Question:**
{question}

**Retrieved Content:**
{context}

**Instructions:**
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information, say "The provided context does not contain sufficient information to answer this question."
3. Be concise but comprehensive
4. Cite specific findings from the papers when relevant

**Answer:**"""


# 评判答案质量的 prompt (使用 LLM-as-Judge)
ANSWER_EVALUATION_PROMPT = """You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system. Your task is to evaluate the quality of a generated answer compared to a reference answer.

**Question:**
{question}

**Generated Answer:**
{generated_answer}

**Reference Answer:**
{reference_answer}

**Retrieved Context:**
{context}

**Evaluation Criteria:**

1. **Correctness (0-5)**: Does the generated answer contain the correct information?
   - 5: Completely correct, covers all key points
   - 4: Mostly correct, minor omissions
   - 3: Partially correct, some important points missing
   - 2: Partially correct but has errors
   - 1: Mostly incorrect
   - 0: Completely incorrect or irrelevant

2. **Faithfulness (0-5)**: Is the answer faithful to the retrieved context?
   - 5: All claims are supported by the context
   - 4: Most claims are supported
   - 3: Some claims are supported
   - 2: Few claims are supported, some hallucinations
   - 1: Major hallucinations
   - 0: Complete hallucination

3. **Relevance (0-5)**: Does the answer directly address the question?
   - 5: Directly and completely addresses the question
   - 4: Addresses the question with minor tangents
   - 3: Partially addresses the question
   - 2: Tangentially related
   - 1: Barely related
   - 0: Completely off-topic

**Output Format (JSON only):**
```json
{{
  "correctness": <score 0-5>,
  "faithfulness": <score 0-5>,
  "relevance": <score 0-5>,
  "reasoning": "<brief explanation of the scores>"
}}
```

Provide your evaluation:"""


# 简化版评判 prompt（如果 LLM 输出不稳定可用这个）
SIMPLE_EVALUATION_PROMPT = """Compare the generated answer with the reference answer for the following question.

Question: {question}
Generated Answer: {generated_answer}
Reference Answer: {reference_answer}

Rate the generated answer:
- Is it correct? (yes/partial/no)
- Is it complete? (yes/partial/no)

Output JSON only:
{{"correct": "yes|partial|no", "complete": "yes|partial|no"}}"""
