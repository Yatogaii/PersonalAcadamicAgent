"""
论文聚类模块

基于 LLM 或 embedding 相似度将论文分组，用于生成跨论文问题
"""

from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass
import numpy as np
import json
import re

from logging_config import logger
from models import get_llm_by_usage

if TYPE_CHECKING:
    from evaluation.qa_generation.qa_generator import ChunkInfo


@dataclass
class PaperCluster:
    """论文聚类结果"""
    cluster_id: int
    theme: str  # 聚类主题（由 LLM 生成）
    paper_ids: list[str]
    paper_titles: list[str]
    common_keywords: list[str]


LLM_CLUSTERING_PROMPT = """You are a research paper clustering expert. Given a list of academic papers with their titles and abstracts, group them by research themes/topics.

Papers:
{papers}

Instructions:
1. Analyze the papers and identify common research themes
2. Group papers that address similar problems, use similar techniques, or belong to the same research area
3. Each paper should belong to exactly one cluster
4. Create 2-5 clusters based on the natural grouping of topics
5. A cluster must have at least 2 papers
6. Use the EXACT paper IDs provided (the UUID strings like "55ce8eef-8f65-48df-9688-a842102e3b24")

IMPORTANT: Output ONLY valid JSON, no markdown, no explanation, no other text.

Output format:
{{"clusters": [{{"theme": "Theme description", "paper_ids": ["uuid1", "uuid2"], "keywords": ["kw1", "kw2"]}}]}}"""


class PaperClusterer:
    """论文聚类器"""
    
    def __init__(self, embedding_model=None, llm_client=None):
        """
        Args:
            embedding_model: Embedding 模型（用于计算相似度）
            llm_client: LLM 客户端（用于生成聚类主题和 LLM 聚类）
        """
        self.embedding_model = embedding_model
        self.llm = llm_client
    
    def cluster_by_llm(
        self,
        all_chunks: dict[str, list["ChunkInfo"]],
        min_cluster_size: int = 2,
        max_clusters: int = 5
    ) -> list[PaperCluster]:
        """
        使用 LLM 对论文进行主题聚类
        
        Args:
            all_chunks: 所有论文的 chunk 信息
            min_cluster_size: 最小聚类大小
            max_clusters: 最大聚类数量
            
        Returns:
            论文聚类列表
        """
        # 如果没有提供 LLM 客户端，则使用 ModelScope（init_chat_model_from_modelscope）初始化
        if not self.llm:
            try:
                logger.info("No llm_client provided, initializing default evaluation LLM via get_llm_by_usage('evaluation') as fallback")
                self.llm = get_llm_by_usage('evaluation')
            except Exception as e:
                logger.warning(f"Failed to init fallback model: {e}, falling back to keyword clustering")
                return self.cluster_by_keywords(all_chunks, min_cluster_size, max_clusters)
        
        # 收集论文信息
        papers_info = []
        paper_titles = {}  # doc_id -> title
        
        for doc_id, chunks in all_chunks.items():
            if not chunks:
                continue
            
            title = chunks[0].title if chunks else doc_id
            paper_titles[doc_id] = title
            
            # 获取 abstract
            abstract = ""
            for chunk in chunks:
                if chunk.section_category == 0:  # Abstract
                    abstract = chunk.chunk_text[:500]
                    break
            
            if not abstract and chunks:
                # 如果没有 abstract，使用前几个 chunk
                abstract = chunks[0].chunk_text[:300]
            
            papers_info.append(f"ID: {doc_id}\nTitle: {title}\nAbstract: {abstract}\n")
        
        if len(papers_info) < 2:
            logger.warning(f"Not enough papers for clustering: {len(papers_info)}")
            return []
        
        # 构建 prompt
        papers_text = "\n---\n".join(papers_info)
        prompt = LLM_CLUSTERING_PROMPT.format(papers=papers_text)
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 解析 JSON 响应
            clusters = self._parse_llm_clustering_response(
                content, paper_titles, min_cluster_size, max_clusters
            )
            
            logger.info(f"LLM clustering created {len(clusters)} clusters from {len(papers_info)} papers")
            for c in clusters:
                logger.info(f"  Cluster '{c.theme}': {len(c.paper_ids)} papers - {c.paper_titles}")
            
            return clusters
            
        except Exception as e:
            logger.error(f"LLM clustering failed: {e}, falling back to keyword clustering")
            return self.cluster_by_keywords(all_chunks, min_cluster_size, max_clusters)
    
    def _parse_llm_clustering_response(
        self,
        response: str,
        paper_titles: dict[str, str],
        min_cluster_size: int,
        max_clusters: int
    ) -> list[PaperCluster]:
        """解析 LLM 聚类响应"""
        # 尝试提取 JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试直接解析
            json_str = response.strip()
            # 移除可能的 markdown 标记
            if json_str.startswith('```'):
                json_str = re.sub(r'^```\w*\n?', '', json_str)
                json_str = re.sub(r'\n?```$', '', json_str)
        
        # 尝试找到 JSON 对象
        if not json_str.startswith('{'):
            # 尝试从响应中提取 JSON
            brace_start = response.find('{')
            if brace_start != -1:
                # 找到匹配的闭合括号
                depth = 0
                for i, c in enumerate(response[brace_start:]):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            json_str = response[brace_start:brace_start + i + 1]
                            break
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            logger.warning(f"Response was: {response[:1000]}")
            return []
        
        clusters = []
        valid_doc_ids = set(paper_titles.keys())
        used_papers = set()
        
        raw_clusters = data.get("clusters", [])
        logger.info(f"Parsed {len(raw_clusters)} clusters from LLM response")
        
        for i, cluster_data in enumerate(raw_clusters):
            if i >= max_clusters:
                break
            
            theme = cluster_data.get("theme", f"Cluster {i}")
            paper_ids = cluster_data.get("paper_ids", [])
            keywords = cluster_data.get("keywords", [])
            
            logger.debug(f"Cluster '{theme}': paper_ids={paper_ids}")
            
            # 验证 paper_ids
            valid_paper_ids = [
                pid for pid in paper_ids 
                if pid in valid_doc_ids and pid not in used_papers
            ]
            
            if len(valid_paper_ids) < min_cluster_size:
                logger.warning(f"Cluster '{theme}' has only {len(valid_paper_ids)} valid papers (need {min_cluster_size}), skipping. Original IDs: {paper_ids}")
                continue
            
            # 记录已使用的论文
            used_papers.update(valid_paper_ids)
            
            cluster = PaperCluster(
                cluster_id=len(clusters),
                theme=theme,
                paper_ids=valid_paper_ids,
                paper_titles=[paper_titles[pid] for pid in valid_paper_ids],
                common_keywords=keywords[:5] if keywords else []
            )
            clusters.append(cluster)
        
        return clusters
    
    def cluster_by_keywords(
        self,
        all_chunks: dict[str, list["ChunkInfo"]],
        min_cluster_size: int = 2,
        max_clusters: int = 5
    ) -> list[PaperCluster]:
        """
        基于关键词相似度聚类论文
        
        简化版：使用 abstract 的词频来计算相似度
        """
        from collections import Counter
        import re
        
        # 提取每篇论文的关键词
        paper_keywords: dict[str, set[str]] = {}
        paper_titles: dict[str, str] = {}
        
        # 停用词
        stopwords = {
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'this', 'that', 'these', 'those', 'we', 'our', 'they', 'their',
            'with', 'by', 'from', 'as', 'it', 'its', 'can', 'which', 'how',
            'what', 'when', 'where', 'who', 'why', 'using', 'based', 'paper',
            'propose', 'present', 'show', 'use', 'approach', 'method', 'system',
            'work', 'study', 'research', 'results', 'analysis', 'evaluation',
        }
        
        for doc_id, chunks in all_chunks.items():
            if not chunks:
                continue
            
            paper_titles[doc_id] = chunks[0].title
            
            # 从 abstract 和 title 提取关键词
            abstract_chunks = [c for c in chunks if c.section_category == 0]
            text = chunks[0].title.lower()
            if abstract_chunks:
                text += " " + abstract_chunks[0].chunk_text.lower()
            
            # 简单分词
            words = re.findall(r'\b[a-z]{3,}\b', text)
            # 过滤停用词，保留有意义的词
            keywords = {w for w in words if w not in stopwords}
            paper_keywords[doc_id] = keywords
        
        # 计算论文间的相似度（Jaccard）
        doc_ids = list(paper_keywords.keys())
        n = len(doc_ids)
        
        if n < 2:
            logger.warning("Not enough papers to cluster")
            return []
        
        # 简单的层次聚类：找到最相似的论文对
        clusters: list[PaperCluster] = []
        used_papers: set[str] = set()
        
        # 计算所有论文对的相似度
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                id1, id2 = doc_ids[i], doc_ids[j]
                kw1, kw2 = paper_keywords[id1], paper_keywords[id2]
                
                if len(kw1) == 0 or len(kw2) == 0:
                    continue
                
                # Jaccard 相似度
                intersection = len(kw1 & kw2)
                union = len(kw1 | kw2)
                similarity = intersection / union if union > 0 else 0
                
                if similarity > 0.1:  # 最低相似度阈值
                    similarities.append((similarity, id1, id2, kw1 & kw2))
        
        # 按相似度排序
        similarities.sort(reverse=True)
        
        # 贪心聚类
        cluster_id = 0
        for sim, id1, id2, common_kw in similarities:
            if cluster_id >= max_clusters:
                break
            
            # 检查是否已被使用
            if id1 in used_papers or id2 in used_papers:
                # 尝试扩展现有聚类
                for cluster in clusters:
                    if id1 in cluster.paper_ids and id2 not in used_papers:
                        cluster.paper_ids.append(id2)
                        cluster.paper_titles.append(paper_titles[id2])
                        cluster.common_keywords = list(
                            set(cluster.common_keywords) & common_kw
                        )[:5]
                        used_papers.add(id2)
                        break
                    elif id2 in cluster.paper_ids and id1 not in used_papers:
                        cluster.paper_ids.append(id1)
                        cluster.paper_titles.append(paper_titles[id1])
                        cluster.common_keywords = list(
                            set(cluster.common_keywords) & common_kw
                        )[:5]
                        used_papers.add(id1)
                        break
                continue
            
            # 创建新聚类
            cluster = PaperCluster(
                cluster_id=cluster_id,
                theme="",  # 稍后由 LLM 生成
                paper_ids=[id1, id2],
                paper_titles=[paper_titles[id1], paper_titles[id2]],
                common_keywords=list(common_kw)[:5]
            )
            clusters.append(cluster)
            used_papers.add(id1)
            used_papers.add(id2)
            cluster_id += 1
        
        # 尝试把剩余论文加入现有聚类
        for doc_id in doc_ids:
            if doc_id in used_papers:
                continue
            
            # 找最相似的聚类
            best_cluster = None
            best_sim = 0
            
            for cluster in clusters:
                cluster_keywords = set()
                for pid in cluster.paper_ids:
                    cluster_keywords |= paper_keywords.get(pid, set())
                
                doc_kw = paper_keywords[doc_id]
                intersection = len(doc_kw & cluster_keywords)
                union = len(doc_kw | cluster_keywords)
                sim = intersection / union if union > 0 else 0
                
                if sim > best_sim and sim > 0.08:
                    best_sim = sim
                    best_cluster = cluster
            
            if best_cluster and len(best_cluster.paper_ids) < 5:
                best_cluster.paper_ids.append(doc_id)
                best_cluster.paper_titles.append(paper_titles[doc_id])
                used_papers.add(doc_id)
        
        # 过滤掉太小的聚类
        clusters = [c for c in clusters if len(c.paper_ids) >= min_cluster_size]
        
        logger.info(f"Created {len(clusters)} paper clusters from {n} papers")
        for i, c in enumerate(clusters):
            logger.info(f"  Cluster {i}: {len(c.paper_ids)} papers, keywords: {c.common_keywords[:3]}")
        
        return clusters
    
    def generate_cluster_themes(
        self,
        clusters: list[PaperCluster],
        all_chunks: dict[str, list["ChunkInfo"]]
    ) -> list[PaperCluster]:
        """使用 LLM 为每个聚类生成主题描述"""
        if not self.llm:
            # 没有 LLM，使用关键词作为主题
            for cluster in clusters:
                cluster.theme = ", ".join(cluster.common_keywords[:3]) or "general security"
            return clusters
        
        for cluster in clusters:
            # 收集聚类内论文的 abstracts
            abstracts = []
            for doc_id in cluster.paper_ids:
                chunks = all_chunks.get(doc_id, [])
                abstract_chunks = [c for c in chunks if c.section_category == 0]
                if abstract_chunks:
                    abstracts.append(f"- {chunks[0].title}: {abstract_chunks[0].chunk_text[:200]}")
            
            prompt = f"""Based on these related papers, generate a short theme description (5-10 words) that captures their common research focus:

{chr(10).join(abstracts)}

Theme (output only the theme, no other text):"""
            
            try:
                response = self.llm.invoke(prompt)
                theme = response.content if hasattr(response, 'content') else str(response)
                cluster.theme = theme.strip()[:100]
            except Exception as e:
                logger.warning(f"Failed to generate theme: {e}")
                cluster.theme = ", ".join(cluster.common_keywords[:3])
        
        return clusters
