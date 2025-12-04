"""
QA Generator (V2)

负责从 chunks 文件生成测试用的 QA pairs

改进版支持 4 个难度级别:
- Level 1 (Easy): 单论文精确题 - 关键词匹配
- Level 2 (Medium): 单论文推理题 - 需要理解方法论
- Level 3 (Hard): 跨论文比较题 - 比较相关论文
- Level 4 (Expert): 领域综述题 - 综合多篇论文

流程:
1. 读取 chunks 文件
2. 对论文进行聚类（用于跨论文问题）
3. 分层次生成不同难度的问题
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
from evaluation.qa_generation.prompts_v2 import (
    LEVEL1_EASY_PROMPT, LEVEL2_MEDIUM_PROMPT, 
    LEVEL3_COMPARISON_PROMPT, LEVEL4_SURVEY_PROMPT,
    format_for_level1, format_for_level2, 
    format_cluster_for_level3, format_for_level4
)
from evaluation.qa_generation.paper_clustering import PaperClusterer, PaperCluster

# 兼容旧版 import
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
        difficulty_distribution: Optional[dict] = None,
        use_clustering: bool = True
    ) -> GroundTruth:
        """
        生成 QA pairs (V2)
        
        Args:
            strategy: 分块策略（决定读取哪个目录的 chunks）
            num_questions: 生成的问题总数
            difficulty_distribution: 难度分布
                - 默认: {"easy": 0.2, "medium": 0.3, "hard": 0.3, "expert": 0.2}
                - 旧版兼容: {"easy": 0.4, "medium": 0.4, "hard": 0.2}
            use_clustering: 是否使用论文聚类生成跨论文问题
            
        Returns:
            GroundTruth 对象
        """
        # 新的默认分布，强调跨论文问题
        default_distribution = {
            "easy": 0.2,      # Level 1: 单论文精确题
            "medium": 0.3,    # Level 2: 单论文推理题  
            "hard": 0.3,      # Level 3: 跨论文比较题
            "expert": 0.2     # Level 4: 领域综述题
        }
        
        difficulty_distribution = difficulty_distribution or default_distribution
        
        # 兼容旧版（没有 expert）
        if "expert" not in difficulty_distribution:
            difficulty_distribution["expert"] = 0
        
        # 加载 chunks
        all_chunks = self.load_chunks(strategy)
        
        if len(all_chunks) < 3:
            raise ValueError(f"Need at least 3 papers, got {len(all_chunks)}")
        
        # 计算各难度的问题数量
        easy_count = int(num_questions * difficulty_distribution.get("easy", 0.2))
        medium_count = int(num_questions * difficulty_distribution.get("medium", 0.3))
        hard_count = int(num_questions * difficulty_distribution.get("hard", 0.3))
        expert_count = num_questions - easy_count - medium_count - hard_count
        
        logger.info(f"Generating {num_questions} questions:")
        logger.info(f"  Level 1 (Easy): {easy_count}")
        logger.info(f"  Level 2 (Medium): {medium_count}")
        logger.info(f"  Level 3 (Hard/Comparison): {hard_count}")
        logger.info(f"  Level 4 (Expert/Survey): {expert_count}")
        
        # 论文聚类（用于 Level 3 和 Level 4）
        clusters = []
        if use_clustering and (hard_count > 0 or expert_count > 0):
            logger.info("Clustering papers using LLM for cross-paper questions...")
            clusterer = PaperClusterer(llm_client=self.llm)
            # 优先使用 LLM 聚类，更准确
            clusters = clusterer.cluster_by_llm(all_chunks, min_cluster_size=2)
            # 如果 LLM 聚类失败，会自动 fallback 到关键词聚类
        
        qa_pairs: list[QAPair] = []
        qa_id = 1
        max_retries = 3
        
        # Level 1: Easy - 单论文精确题
        if easy_count > 0:
            logger.info("Generating Level 1 (Easy) questions...")
            easy_pairs = self._generate_with_retry(
                lambda: self.generate_level1_easy(all_chunks, easy_count, start_id=qa_id),
                easy_count, max_retries, "Level 1"
            )
            qa_pairs.extend(easy_pairs)
            qa_id += len(easy_pairs)
        
        # Level 2: Medium - 单论文推理题
        if medium_count > 0:
            logger.info("Generating Level 2 (Medium) questions...")
            medium_pairs = self._generate_with_retry(
                lambda: self.generate_level2_medium(all_chunks, medium_count, start_id=qa_id),
                medium_count, max_retries, "Level 2"
            )
            qa_pairs.extend(medium_pairs)
            qa_id += len(medium_pairs)
        
        # Level 3: Hard - 跨论文比较题
        if hard_count > 0:
            logger.info("Generating Level 3 (Hard/Comparison) questions...")
            hard_pairs = self._generate_with_retry(
                lambda: self.generate_level3_comparison(all_chunks, clusters, hard_count, start_id=qa_id),
                hard_count, max_retries, "Level 3"
            )
            qa_pairs.extend(hard_pairs)
            qa_id += len(hard_pairs)
        
        # Level 4: Expert - 领域综述题
        if expert_count > 0:
            logger.info("Generating Level 4 (Expert/Survey) questions...")
            expert_pairs = self._generate_with_retry(
                lambda: self.generate_level4_survey(all_chunks, clusters, expert_count, start_id=qa_id),
                expert_count, max_retries, "Level 4"
            )
            qa_pairs.extend(expert_pairs)
        
        # 构建 GroundTruth
        ground_truth = GroundTruth(
            version="2.0",  # 新版本
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
        
        logger.info(f"Total generated: {len(qa_pairs)} questions")
        return ground_truth
    
    def _generate_with_retry(
        self, 
        gen_func, 
        target_count: int, 
        max_retries: int,
        level_name: str
    ) -> list[QAPair]:
        """带重试的问题生成"""
        for attempt in range(max_retries):
            pairs = gen_func()
            if len(pairs) >= target_count * 0.5:
                logger.info(f"  {level_name}: Generated {len(pairs)}/{target_count}")
                return pairs
            logger.warning(f"  {level_name} attempt {attempt + 1}/{max_retries}: got {len(pairs)}/{target_count}")
        return pairs
    
    # ==================== Level 1: Easy ====================
    
    def generate_level1_easy(
        self,
        all_chunks: dict[str, list[ChunkInfo]],
        count: int,
        start_id: int = 1
    ) -> list[QAPair]:
        """
        Level 1: 单论文精确题
        
        特点: 
        - 问题包含论文关键词
        - 答案直接来自 abstract
        - 单论文检索即可回答
        """
        paper_summaries = format_for_level1(all_chunks, max_papers=30)
        
        prompt = LEVEL1_EASY_PROMPT.format(
            paper_summaries=paper_summaries,
            count=count
        )
        
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        return self._parse_qa_response(content, Difficulty.EASY, start_id)[:count]
    
    # ==================== Level 2: Medium ====================
    
    def generate_level2_medium(
        self,
        all_chunks: dict[str, list[ChunkInfo]],
        count: int,
        start_id: int = 1
    ) -> list[QAPair]:
        """
        Level 2: 单论文推理题
        
        特点:
        - 需要理解论文方法论
        - 答案来自 method/evaluation section
        - 问题不包含明显关键词
        """
        paper_summaries = format_for_level2(all_chunks, max_papers=15)
        
        prompt = LEVEL2_MEDIUM_PROMPT.format(
            paper_summaries=paper_summaries,
            count=count
        )
        
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        return self._parse_qa_response(content, Difficulty.MEDIUM, start_id)[:count]
    
    # ==================== Level 3: Hard (Comparison) ====================
    
    def generate_level3_comparison(
        self,
        all_chunks: dict[str, list[ChunkInfo]],
        clusters: list[PaperCluster],
        count: int,
        start_id: int = 1
    ) -> list[QAPair]:
        """
        Level 3: 跨论文比较题
        
        特点:
        - 比较同一聚类内的相关论文
        - 需要理解多篇论文的方法差异
        - 问题不提及具体论文名
        """
        qa_pairs = []
        
        if not clusters:
            # 没有聚类，回退到旧方法
            logger.warning("No clusters available, falling back to old method")
            return self.generate_hard_questions(all_chunks, count, start_id)
        
        # 每个聚类生成一些比较题
        questions_per_cluster = max(1, count // len(clusters))
        
        for cluster in clusters:
            if len(qa_pairs) >= count:
                break
            
            cluster_papers = format_cluster_for_level3(cluster, all_chunks)
            
            prompt = LEVEL3_COMPARISON_PROMPT.format(
                cluster_theme=cluster.theme,
                cluster_papers=cluster_papers,
                count=questions_per_cluster
            )
            
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            pairs = self._parse_qa_response(
                content, Difficulty.HARD, start_id + len(qa_pairs)
            )
            
            # 标记为多论文问题
            for p in pairs:
                p.is_multi_paper = True
            
            qa_pairs.extend(pairs[:questions_per_cluster])
        
        return qa_pairs[:count]
    
    # ==================== Level 4: Expert (Survey) ====================
    
    def generate_level4_survey(
        self,
        all_chunks: dict[str, list[ChunkInfo]],
        clusters: list[PaperCluster],
        count: int,
        start_id: int = 1
    ) -> list[QAPair]:
        """
        Level 4: 领域综述题
        
        特点:
        - 需要综合 3+ 篇论文的信息
        - 问关于研究趋势、方法论模式、共同挑战
        - 答案需要高度综合
        """
        area_overview, paper_list = format_for_level4(all_chunks, clusters)
        
        prompt = LEVEL4_SURVEY_PROMPT.format(
            area_overview=area_overview,
            paper_list=paper_list,
            count=count
        )
        
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        pairs = self._parse_qa_response(content, Difficulty.HARD, start_id)
        
        # 标记为多论文问题
        for p in pairs:
            p.is_multi_paper = True
        
        return pairs[:count]
    
    # ==================== 兼容旧版方法 ====================
    
    def generate_easy_questions(
        self,
        all_chunks: dict[str, list[ChunkInfo]],
        count: int,
        start_id: int = 1
    ) -> list[QAPair]:
        """兼容旧版"""
        return self.generate_level1_easy(all_chunks, count, start_id)
    
    def generate_medium_questions(
        self,
        all_chunks: dict[str, list[ChunkInfo]],
        count: int,
        start_id: int = 1
    ) -> list[QAPair]:
        """兼容旧版"""
        return self.generate_level2_medium(all_chunks, count, start_id)
    
    def generate_hard_questions(
        self,
        all_chunks: dict[str, list[ChunkInfo]],
        count: int,
        start_id: int = 1
    ) -> list[QAPair]:
        """兼容旧版 - 回退到简单的跨论文问题"""
        paper_summaries = format_chunks_for_hard(all_chunks, max_papers=30)
        
        prompt = HARD_QA_PROMPT.format(
            paper_summaries=paper_summaries,
            count=count
        )
        
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        qa_pairs = self._parse_qa_response(content, Difficulty.HARD, start_id)
        
        for q in qa_pairs:
            q.is_multi_paper = True
        
        return qa_pairs[:count]
    
    # ==================== JSON 解析 ====================
    
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
