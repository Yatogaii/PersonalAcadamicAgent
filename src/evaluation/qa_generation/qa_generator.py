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
        batch_size = 5  # 每批生成 5 个问题，提高稳定性
        
        # Level 1: Easy - 单论文精确题
        if easy_count > 0:
            logger.info("Generating Level 1 (Easy) questions...")
            easy_pairs = self._generate_in_batches(
                lambda count, sid: self._generate_level1_batch(all_chunks, count, sid),
                easy_count, batch_size, Difficulty.EASY, "Level 1", qa_id
            )
            qa_pairs.extend(easy_pairs)
            qa_id += len(easy_pairs)
        
        # Level 2: Medium - 单论文推理题
        if medium_count > 0:
            logger.info("Generating Level 2 (Medium) questions...")
            medium_pairs = self._generate_in_batches(
                lambda count, sid: self._generate_level2_batch(all_chunks, count, sid),
                medium_count, batch_size, Difficulty.MEDIUM, "Level 2", qa_id
            )
            qa_pairs.extend(medium_pairs)
            qa_id += len(medium_pairs)
        
        # Level 3: Hard - 跨论文比较题
        if hard_count > 0:
            logger.info("Generating Level 3 (Hard/Comparison) questions...")
            hard_pairs = self._generate_in_batches(
                lambda count, sid: self._generate_level3_batch(all_chunks, clusters, count, sid),
                hard_count, batch_size, Difficulty.HARD, "Level 3", qa_id
            )
            qa_pairs.extend(hard_pairs)
            qa_id += len(hard_pairs)
        
        # Level 4: Expert - 领域综述题
        if expert_count > 0:
            logger.info("Generating Level 4 (Expert/Survey) questions...")
            expert_pairs = self._generate_in_batches(
                lambda count, sid: self._generate_level4_batch(all_chunks, clusters, count, sid),
                expert_count, batch_size, Difficulty.EXPERT, "Level 4", qa_id
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
                "expert": len([q for q in qa_pairs if q.difficulty == Difficulty.EXPERT]),
            },
            source_distribution=self._count_source_distribution(qa_pairs),
        )
        
        logger.info(f"Total generated: {len(qa_pairs)} questions")
        return ground_truth
    
    def _generate_in_batches(
        self,
        gen_func_single_batch,
        total_count: int,
        batch_size: int,
        difficulty: Difficulty,
        level_name: str,
        start_id: int = 1
    ) -> list[QAPair]:
        """
        分批次生成问题，每批生成 batch_size 个
        
        Args:
            gen_func_single_batch: 生成单批问题的函数，接受 (count, start_id) 参数
            total_count: 总共需要的问题数
            batch_size: 每批生成的问题数
            difficulty: 难度级别
            level_name: 级别名称（用于日志）
            start_id: 起始 ID
        """
        all_pairs = []
        current_id = start_id
        remaining = total_count
        max_retries_per_batch = 2
        
        while remaining > 0:
            batch_count = min(batch_size, remaining)
            
            # 尝试生成这一批
            for attempt in range(max_retries_per_batch):
                try:
                    pairs = gen_func_single_batch(batch_count, current_id)
                    if pairs:
                        all_pairs.extend(pairs)
                        current_id += len(pairs)
                        remaining -= len(pairs)
                        logger.debug(f"  {level_name}: Batch done, got {len(pairs)}, total {len(all_pairs)}/{total_count}")
                        break
                except Exception as e:
                    logger.warning(f"  {level_name} batch attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries_per_batch - 1:
                        # 最后一次重试失败，跳过这批
                        remaining -= batch_count
                        logger.warning(f"  {level_name}: Skipping batch after {max_retries_per_batch} retries")
        
        logger.info(f"  {level_name}: Generated {len(all_pairs)}/{total_count}")
        return all_pairs
    
    # ==================== 批次生成内部方法 ====================
    
    def _generate_level1_batch(
        self,
        all_chunks: dict[str, list[ChunkInfo]],
        count: int,
        start_id: int
    ) -> list[QAPair]:
        """生成一批 Level 1 问题"""
        paper_summaries = format_for_level1(all_chunks, max_papers=30)
        
        prompt = LEVEL1_EASY_PROMPT.format(
            paper_summaries=paper_summaries,
            count=count
        )
        
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        return self._parse_qa_response(content, Difficulty.EASY, start_id)[:count]
    
    def _generate_level2_batch(
        self,
        all_chunks: dict[str, list[ChunkInfo]],
        count: int,
        start_id: int
    ) -> list[QAPair]:
        """生成一批 Level 2 问题"""
        paper_summaries = format_for_level2(all_chunks, max_papers=15)
        
        prompt = LEVEL2_MEDIUM_PROMPT.format(
            paper_summaries=paper_summaries,
            count=count
        )
        
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        return self._parse_qa_response(content, Difficulty.MEDIUM, start_id)[:count]
    
    def _generate_level3_batch(
        self,
        all_chunks: dict[str, list[ChunkInfo]],
        clusters: list[PaperCluster],
        count: int,
        start_id: int
    ) -> list[QAPair]:
        """生成一批 Level 3 问题"""
        if not clusters:
            logger.warning("No clusters available for Level 3")
            return []
        
        # 随机选择一个聚类
        import random
        cluster = random.choice(clusters)
        cluster_papers = format_cluster_for_level3(cluster, all_chunks)
        
        prompt = LEVEL3_COMPARISON_PROMPT.format(
            cluster_theme=cluster.theme,
            cluster_papers=cluster_papers,
            count=count
        )
        
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        pairs = self._parse_qa_response(content, Difficulty.HARD, start_id)[:count]
        
        # 标记为多论文问题
        for p in pairs:
            p.is_multi_paper = True
        
        return pairs
    
    def _generate_level4_batch(
        self,
        all_chunks: dict[str, list[ChunkInfo]],
        clusters: list[PaperCluster],
        count: int,
        start_id: int
    ) -> list[QAPair]:
        """生成一批 Level 4 问题"""
        area_overview, paper_list = format_for_level4(all_chunks, clusters)
        
        prompt = LEVEL4_SURVEY_PROMPT.format(
            area_overview=area_overview,
            paper_list=paper_list,
            count=count
        )
        
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        pairs = self._parse_qa_response(content, Difficulty.EXPERT, start_id)[:count]
        
        # 标记为多论文问题
        for p in pairs:
            p.is_multi_paper = True
        
        return pairs
    
    # ==================== Level 1: Easy (保留旧接口) ====================
    
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
        return self._generate_level1_batch(all_chunks, count, start_id)
    
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
        return self._generate_level2_batch(all_chunks, count, start_id)
    
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
    
    def _fix_json_string(self, json_str: str) -> str:
        """尝试修复常见的 JSON 格式错误"""
        import re
        
        fixed = json_str
        
        # 1. 移除尾部逗号
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*\]', ']', fixed)
        
        # 2. 修复未转义的换行符（在字符串内部）
        # 这是一个常见问题：LLM 在字符串值中包含了实际的换行符
        def escape_newlines_in_strings(s: str) -> str:
            result = []
            in_string = False
            escape_next = False
            i = 0
            while i < len(s):
                c = s[i]
                if escape_next:
                    result.append(c)
                    escape_next = False
                elif c == '\\':
                    result.append(c)
                    escape_next = True
                elif c == '"':
                    result.append(c)
                    in_string = not in_string
                elif c == '\n' and in_string:
                    result.append('\\n')
                elif c == '\r' and in_string:
                    result.append('\\r')
                elif c == '\t' and in_string:
                    result.append('\\t')
                else:
                    result.append(c)
                i += 1
            return ''.join(result)
        
        fixed = escape_newlines_in_strings(fixed)
        
        # 3. 修复控制字符
        # 移除或转义 JSON 字符串中不允许的控制字符
        def remove_control_chars(s: str) -> str:
            # 保留换行、回车、制表符（已经在上面转义了）
            # 移除其他控制字符
            return ''.join(c if ord(c) >= 32 or c in '\n\r\t' else '' for c in s)
        
        fixed = remove_control_chars(fixed)
        
        return fixed
    
    def _extract_json_objects(self, content: str) -> list[dict]:
        """
        逐个提取 JSON 对象，使用平衡括号的方法
        即使整体 JSON 数组无效也能工作
        """
        objects = []
        i = 0
        n = len(content)
        
        while i < n:
            # 找到下一个 { 开始
            if content[i] != '{':
                i += 1
                continue
            
            # 尝试找到匹配的 }
            start = i
            brace_count = 0
            in_string = False
            escape_next = False
            j = i
            
            while j < n:
                c = content[j]
                
                if escape_next:
                    escape_next = False
                    j += 1
                    continue
                
                if c == '\\':
                    escape_next = True
                    j += 1
                    continue
                
                if c == '"':
                    in_string = not in_string
                    j += 1
                    continue
                
                if in_string:
                    j += 1
                    continue
                
                if c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # 找到完整的对象
                        obj_str = content[start:j+1]
                        try:
                            obj = json.loads(obj_str)
                            if isinstance(obj, dict) and 'question' in obj:
                                objects.append(obj)
                        except json.JSONDecodeError:
                            # 尝试修复
                            try:
                                fixed_obj = self._fix_json_string(obj_str)
                                obj = json.loads(fixed_obj)
                                if isinstance(obj, dict) and 'question' in obj:
                                    objects.append(obj)
                            except json.JSONDecodeError:
                                pass
                        break
                
                j += 1
            
            i = j + 1
        
        return objects
    
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
        data = None
        
        # 尝试修复常见的 JSON 错误
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}, attempting to fix...")
            
            # 第一轮修复
            fixed_json = self._fix_json_string(json_str)
            
            try:
                data = json.loads(fixed_json)
                logger.info("JSON fixed successfully (method 1)")
            except json.JSONDecodeError as e2:
                logger.warning(f"JSON fix method 1 failed: {e2}, trying fallback...")
                
                # 第二轮修复：逐个提取 JSON 对象
                extracted_objects = self._extract_json_objects(json_str)
                if extracted_objects:
                    data = extracted_objects
                    logger.info(f"JSON fixed via object extraction: got {len(data)} objects")
                else:
                    logger.error(f"All JSON fix methods failed")
                    logger.debug(f"Problematic JSON (first 1000 chars): {json_str[:1000]}...")
                    # 记录错误位置附近的内容以便调试
                    if hasattr(e, 'pos') and e.pos:
                        start = max(0, e.pos - 100)
                        end = min(len(json_str), e.pos + 100)
                        logger.debug(f"Context around error position {e.pos}: ...{json_str[start:end]}...")
                    return []
        
        if data is None:
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
