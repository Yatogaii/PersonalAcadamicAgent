"""
QA Generation Module - 生成测试用的 QA pairs

负责：
1. 读取标注结果
2. 按策略生成多样化的 QA pairs
3. 输出 Ground Truth 文件
"""

from .qa_generator import QAGenerator

__all__ = ["QAGenerator"]
