"""
QA Generator

负责从 chunks 文件生成测试用的 QA pairs

流程:
1. 读取 data_preparation 保存的 chunks 文件
2. 按论文/section 分组
3. 使用 LLM 生成不同难度的问题
4. 保存 Ground Truth
"""

from typing import TYPE_CHECKING, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json
import random

from logging_config import logger

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

from evaluation.schemas import (
    QAPair, GroundTruth, Difficulty, AnswerSource
)
from evaluation.config import EvaluationConfig, ChunkStrategy
from evaluation.qa_generation.prompts import (
    EASY_QA_PROMPT, MEDIUM_QA_PROMPT, HARD_QA_PROMPT,
    format_chunks_for_easy, format_chunks_for_medium, format_chunks_for_hard
)


@dataclass
class ChunkInfo:
    """Chunk 信息（从文件加载）"""
    doc_id: str
    title: str
    chunk_index: int
    chunk_text: str
    section_title: str = ""
    section_category: int = 0
    contextual_prefix: str = ""


class QAGenerator:
    """QA 生成器"""
    
    def __init__(
        self,
        llm_client: "BaseChatModel",
        config: Optional[EvaluationConfig] = None
    ):
        """
        Args:
            llm_client: LLM 客户端
            config: 评估配置
        """
        self.llm = llm_client
        self.config = config or EvaluationConfig()
        
        # 确保目录存在
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_chunks(
        self, 
        strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH
    ) -> dict[str, list[ChunkInfo]]:
        """
        加载 chunks 文件
        
        Returns:
            dict[doc_id, list[ChunkInfo]]
        """
        chunks_dir = self.config.chunks_dir / strategy.value
        
        if not chunks_dir.exists():
            raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
        
        all_chunks: dict[str, list[ChunkInfo]] = {}
        
        for chunk_file in chunks_dir.glob("*.json"):
            with open(chunk_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            doc_id = data["doc_id"]
            title = data["title"]
            chunks = []
            
            for c in data["chunks"]:
                chunks.append(ChunkInfo(
                    doc_id=doc_id,
                    title=title,
                    chunk_index=c.get("chunk_index", 0),
                    chunk_text=c.get("chunk_text", c.get("text", "")),
                    section_title=c.get("section_title", ""),
                    section_category=c.get("section_category", 0),
                    contextual_prefix=c.get("contextual_prefix", ""),
                ))
            
            all_chunks[doc_id] = chunks
        
        logger.info(f"Loaded chunks for {len(all_chunks)} papers from {chunks_dir}")
        return all_chunks
    
    def generate(
        self,
        strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH,
        num_questions: int = 50,
        difficulty_distribution: Optional[dict] = None
    ) -> GroundTruth:
        """
        生成 QA pairs
        
        Args:
            strategy: 分块策略（决定读取哪个目录的 chunks）
            num_questions: 生成的问题总数
            difficulty_distribution: 难度分布 {"easy": 0.4, "medium": 0.4, "hard": 0.2}
            
        Returns:
            GroundTruth 对象
        """
        difficulty_distribution = difficulty_distribution or {
            "easy": 0.4,
            "medium": 0.4,
            "hard": 0.2
        }
        
        # 加载 chunks
        all_chunks = self.load_chunks(strategy)
        
        if len(all_chunks) < 3:
            raise ValueError(f"Need at least 3 papers, got {len(all_chunks)}")
        
        # 计算各难度的问题数量
        easy_count = int(num_questions * difficulty_distribution.get("easy", 0.4))
        medium_count = int(num_questions * difficulty_distribution.get("medium", 0.4))
        hard_count = num_questions - easy_count - medium_count
        
        logger.info(f"Generating {num_questions} questions: "
                   f"easy={easy_count}, medium={medium_count}, hard={hard_count}")
        
        qa_pairs: list[QAPair] = []
        qa_id = 1
        
        # 最大重试次数
        max_retries = 3
        
        # 生成 Easy 问题（带重试）
        if easy_count > 0:
            easy_pairs = []
            for attempt in range(max_retries):
                easy_pairs = self.generate_easy_questions(all_chunks, easy_count, start_id=qa_id)
                if len(easy_pairs) >= easy_count * 0.5:  # 至少生成 50% 的问题
                    break
                logger.warning(f"Easy questions attempt {attempt + 1}/{max_retries}: got {len(easy_pairs)}/{easy_count}")
            qa_pairs.extend(easy_pairs)
            qa_id += len(easy_pairs)
            logger.info(f"Generated {len(easy_pairs)} easy questions")
        
        # 生成 Medium 问题（带重试）
        if medium_count > 0:
            medium_pairs = []
            for attempt in range(max_retries):
                medium_pairs = self.generate_medium_questions(all_chunks, medium_count, start_id=qa_id)
                if len(medium_pairs) >= medium_count * 0.5:
                    break
                logger.warning(f"Medium questions attempt {attempt + 1}/{max_retries}: got {len(medium_pairs)}/{medium_count}")
            qa_pairs.extend(medium_pairs)
            qa_id += len(medium_pairs)
            logger.info(f"Generated {len(medium_pairs)} medium questions")
        
        # 生成 Hard 问题（带重试）
        if hard_count > 0:
            hard_pairs = []
            for attempt in range(max_retries):
                hard_pairs = self.generate_hard_questions(all_chunks, hard_count, start_id=qa_id)
                if len(hard_pairs) >= hard_count * 0.5:
                    break
                logger.warning(f"Hard questions attempt {attempt + 1}/{max_retries}: got {len(hard_pairs)}/{hard_count}")
            qa_pairs.extend(hard_pairs)
            logger.info(f"Generated {len(hard_pairs)} hard questions")
        
        # 构建 GroundTruth
        ground_truth = GroundTruth(
            version="1.0",
            created_at=datetime.now().isoformat(),
            total_papers=len(all_chunks),
            qa_pairs=qa_pairs,
            difficulty_distribution={
                "easy": len([q for q in qa_pairs if q.difficulty == Difficulty.EASY]),
                "medium": len([q for q in qa_pairs if q.difficulty == Difficulty.MEDIUM]),
                "hard": len([q for q in qa_pairs if q.difficulty == Difficulty.HARD]),
            },
            source_distribution=self._count_source_distribution(qa_pairs),
        )
        
        return ground_truth
    
    def generate_easy_questions(
        self,
        all_chunks: dict[str, list[ChunkInfo]],
        count: int,
        start_id: int = 1
    ) -> list[QAPair]:
        """
        生成 Easy 级别问题
        
        特点: 关键词直接匹配，单论文检索
        """
        # 准备输入
        paper_summaries = format_chunks_for_easy(all_chunks, max_papers=30)
        
        prompt = EASY_QA_PROMPT.format(
            paper_summaries=paper_summaries,
            count=count
        )
        
        # 调用 LLM
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # 解析 JSON
        qa_pairs = self._parse_qa_response(content, Difficulty.EASY, start_id)
        return qa_pairs[:count]
    
    def generate_medium_questions(
        self,
        all_chunks: dict[str, list[ChunkInfo]],
        count: int,
        start_id: int = 1
    ) -> list[QAPair]:
        """
        生成 Medium 级别问题
        
        特点: 需要语义理解，可能需要 section-level 检索
        """
        # 准备输入（包含 method/eval section）
        paper_summaries = format_chunks_for_medium(all_chunks, max_papers=20)
        
        prompt = MEDIUM_QA_PROMPT.format(
            paper_summaries=paper_summaries,
            count=count
        )
        
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        qa_pairs = self._parse_qa_response(content, Difficulty.MEDIUM, start_id)
        return qa_pairs[:count]
    
    def generate_hard_questions(
        self,
        all_chunks: dict[str, list[ChunkInfo]],
        count: int,
        start_id: int = 1
    ) -> list[QAPair]:
        """
        生成 Hard 级别问题
        
        特点: 跨论文综合，需要多步检索
        """
        # 准备输入
        paper_summaries = format_chunks_for_hard(all_chunks, max_papers=30)
        
        prompt = HARD_QA_PROMPT.format(
            paper_summaries=paper_summaries,
            count=count
        )
        
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        qa_pairs = self._parse_qa_response(content, Difficulty.HARD, start_id)
        
        # Hard 问题标记为 multi-paper
        for q in qa_pairs:
            q.is_multi_paper = True
        
        return qa_pairs[:count]
    
    def _parse_qa_response(
        self, 
        content: str, 
        difficulty: Difficulty,
        start_id: int
    ) -> list[QAPair]:
        """解析 LLM 返回的 JSON"""
        import re
        
        # 尝试提取 JSON 数组
        json_match = re.search(r'\[[\s\S]*\]', content)
        
        if not json_match:
            logger.warning(f"No JSON array found in response for {difficulty.value}")
            logger.debug(f"Response content: {content[:500]}...")
            return []
        
        json_str = json_match.group()
        
        # 尝试修复常见的 JSON 错误
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}, attempting to fix...")
            
            # 尝试修复: 移除尾部逗号
            fixed_json = re.sub(r',\s*}', '}', json_str)
            fixed_json = re.sub(r',\s*\]', ']', fixed_json)
            
            # 尝试修复: 替换单引号为双引号
            fixed_json = fixed_json.replace("'", '"')
            
            try:
                data = json.loads(fixed_json)
                logger.info("JSON fixed successfully")
            except json.JSONDecodeError as e2:
                logger.error(f"JSON fix failed: {e2}")
                logger.debug(f"Problematic JSON: {json_str[:1000]}...")
                return []
        
        qa_pairs = []
        for i, item in enumerate(data):
            try:
                # 解析 answer_source
                source_str = item.get("answer_source", "abstract").lower()
                try:
                    source = AnswerSource(source_str)
                except ValueError:
                    source = AnswerSource.ABSTRACT
                
                qa_pair = QAPair(
                    id=start_id + i,
                    question=item.get("question", ""),
                    difficulty=difficulty,
                    expected_doc_ids=item.get("expected_doc_ids", []),
                    expected_chunk_ids=item.get("expected_chunk_ids"),
                    answer_source=source,
                    reference_answer=item.get("reference_answer", ""),
                    is_multi_paper=item.get("is_multi_paper", False),
                )
                qa_pairs.append(qa_pair)
            except Exception as e:
                logger.warning(f"Failed to parse QA item: {e}")
                continue
        
        return qa_pairs
    
    def _count_source_distribution(self, qa_pairs: list[QAPair]) -> dict:
        """统计 answer_source 分布"""
        dist: dict[str, int] = {}
        for q in qa_pairs:
            source = q.answer_source.value
            dist[source] = dist.get(source, 0) + 1
        return dist
    
    def save(self, ground_truth: GroundTruth) -> Path:
        """保存 Ground Truth 到 JSON 文件"""
        output_path = self.config.ground_truth_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为可序列化格式
        data = {
            "version": ground_truth.version,
            "created_at": ground_truth.created_at,
            "total_papers": ground_truth.total_papers,
            "difficulty_distribution": ground_truth.difficulty_distribution,
            "source_distribution": ground_truth.source_distribution,
            "qa_pairs": [
                {
                    "id": q.id,
                    "question": q.question,
                    "difficulty": q.difficulty.value,
                    "expected_doc_ids": q.expected_doc_ids,
                    "expected_chunk_ids": q.expected_chunk_ids,
                    "answer_source": q.answer_source.value,
                    "reference_answer": q.reference_answer,
                    "is_multi_paper": q.is_multi_paper,
                }
                for q in ground_truth.qa_pairs
            ]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(ground_truth.qa_pairs)} QA pairs to {output_path}")
        return output_path
    
    def load(self) -> Optional[GroundTruth]:
        """从文件加载 Ground Truth"""
        input_path = self.config.ground_truth_file
        
        if not input_path.exists():
            return None
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        qa_pairs = []
        for q in data.get("qa_pairs", []):
            qa_pairs.append(QAPair(
                id=q["id"],
                question=q["question"],
                difficulty=Difficulty(q["difficulty"]),
                expected_doc_ids=q["expected_doc_ids"],
                expected_chunk_ids=q.get("expected_chunk_ids"),
                answer_source=AnswerSource(q["answer_source"]),
                reference_answer=q.get("reference_answer", ""),
                is_multi_paper=q.get("is_multi_paper", False),
            ))
        
        ground_truth = GroundTruth(
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", ""),
            total_papers=data.get("total_papers", 0),
            qa_pairs=qa_pairs,
            difficulty_distribution=data.get("difficulty_distribution", {}),
            source_distribution=data.get("source_distribution", {}),
        )
        
        logger.info(f"Loaded {len(qa_pairs)} QA pairs from {input_path}")
        return ground_truth
