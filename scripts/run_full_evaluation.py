"""
完整评估流程脚本

从业务 collection 导出数据 → 生成 QA → 运行评估

使用方式:
    # 完整对比实验（12种配置: 2 chunk × 3 index × 2 agentic）
    uv run python scripts/run_full_evaluation.py --compare --sample 20 --num-questions 30
    
    # 快速对比（跳过 L3，只比较检索性能）
    uv run python scripts/run_full_evaluation.py --compare --sample 20 --no-l3
    
    # 单配置评估（使用已有数据）
    uv run python scripts/run_full_evaluation.py --eval-only
    
    # 生成新的 QA（使用已有 chunks）
    uv run python scripts/run_full_evaluation.py --generate-qa --num-questions 20
    
    # 查看当前数据状态
    uv run python scripts/run_full_evaluation.py --status
    
    # 旧的 --full 模式（只跑单配置）
    uv run python scripts/run_full_evaluation.py --full --sample 20
"""

import sys
import argparse
from pathlib import Path

# 添加 src 到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from logging_config import logger


def step1_export_and_prepare(
    sample_size: int = None, 
    drop_existing: bool = False,
    all_strategies: bool = False
):
    """
    Step 1: 从业务库导出数据并准备评估 collection
    
    Args:
        sample_size: 抽样数量
        drop_existing: 是否删除已有 collection
        all_strategies: 是否准备所有 chunk 策略 (paragraph + contextual)
    """
    from rag.milvus import MilvusProvider
    from models import get_llm_by_usage
    from evaluation.config import EvaluationConfig, ChunkStrategy
    from evaluation.data_preparation.pipeline import DataPreparationPipeline
    
    print("=" * 70)
    print("Step 1: 数据准备 (Export & Prepare)")
    print("=" * 70)
    
    # 初始化
    config = EvaluationConfig()
    source_rag = MilvusProvider()
    llm = get_llm_by_usage('evaluation')
    
    print(f"源 Collection: {source_rag.collection}")
    print(f"评估数据目录: {config.data_dir}")
    
    # 创建 pipeline
    pipeline = DataPreparationPipeline(
        source_rag_client=source_rag,
        llm_client=llm,
        config=config
    )
    
    # 选择策略
    if all_strategies:
        strategies = [ChunkStrategy.PARAGRAPH, ChunkStrategy.CONTEXTUAL]
        print(f"策略: paragraph + contextual (两个 collection)")
    else:
        strategies = [ChunkStrategy.PARAGRAPH]
        print(f"策略: paragraph only")
    
    # 运行
    result = pipeline.run(
        strategies=strategies,
        sample_size=sample_size,
        drop_existing=drop_existing
    )
    
    print(f"\n✓ 导出论文: {result.papers_exported}")
    for strategy in strategies:
        s = strategy.value
        print(f"✓ {s}: 处理 {result.papers_success.get(s, 0)} 篇, chunks {result.chunks_saved.get(s, 0)} 个")
    
    return result


def step2_generate_qa(num_questions: int = 50):
    """
    Step 2: 从 chunks 生成 QA pairs
    """
    from models import get_llm_by_usage
    from evaluation.config import EvaluationConfig, ChunkStrategy
    from evaluation.qa_generation.qa_generator import QAGenerator
    
    print("\n" + "=" * 70)
    print("Step 2: QA 生成")
    print("=" * 70)
    
    config = EvaluationConfig()
    llm = get_llm_by_usage('evaluation')
    
    generator = QAGenerator(llm_client=llm, config=config)
    
    # 检查 chunks 是否存在
    chunks_dir = config.chunks_dir / "paragraph"
    if not chunks_dir.exists() or not list(chunks_dir.glob("*.json")):
        print("⚠ 没有找到 chunks 文件，请先运行 step1")
        return None
    
    print(f"Chunks 目录: {chunks_dir}")
    print(f"生成 {num_questions} 个问题...")
    
    # 生成 QA（使用新的 4 级难度分布）
    ground_truth = generator.generate(
        strategy=ChunkStrategy.PARAGRAPH,
        num_questions=num_questions,
        difficulty_distribution={
            "easy": 0.2,      # Level 1: 单论文精确题
            "medium": 0.3,    # Level 2: 单论文推理题
            "hard": 0.3,      # Level 3: 跨论文比较题
            "expert": 0.2     # Level 4: 领域综述题
        }
    )
    
    # 保存
    save_path = generator.save(ground_truth)
    
    print(f"\n✓ 生成 {len(ground_truth.qa_pairs)} 个问题")
    print(f"✓ 保存到: {save_path}")
    print(f"  - Level 1 (Easy): {ground_truth.difficulty_distribution.get('easy', 0)}")
    print(f"  - Level 2 (Medium): {ground_truth.difficulty_distribution.get('medium', 0)}")
    print(f"  - Level 3 (Hard): {ground_truth.difficulty_distribution.get('hard', 0)}")
    
    return ground_truth


def step3_run_evaluation(run_l3: bool = True):
    """
    Step 3: 运行评估
    """
    from models import get_llm_by_usage
    from rag.milvus import MilvusProvider
    from evaluation.config import EvaluationConfig, ChunkStrategy
    from evaluation.runner import EvaluationRunner
    from evaluation.data_preparation.collection_builder import CollectionBuilder
    
    print("\n" + "=" * 70)
    print("Step 3: 运行评估")
    print("=" * 70)
    
    config = EvaluationConfig()
    
    # 检查 ground truth
    if not config.ground_truth_file.exists():
        print("⚠ 没有找到 ground_truth.json，请先运行 step2")
        return None
    
    builder = CollectionBuilder(config)
    llm = get_llm_by_usage('evaluation') if run_l3 else None
    
    print(f"Ground Truth: {config.ground_truth_file}")
    print(f"评估 Collection: papers_eval_paragraph")
    print(f"L3 评估: {'启用' if run_l3 else '禁用'}")
    
    with builder.use_chunk_strategy(ChunkStrategy.PARAGRAPH):
        milvus = MilvusProvider()
        
        runner = EvaluationRunner(
            rag_client=milvus,
            llm_client=llm,
            config=config
        )
        
        # 加载 ground truth
        ground_truth = runner.load_ground_truth()
        print(f"\n加载 {len(ground_truth.qa_pairs)} 个测试问题")
        
        # 运行评估
        print("\n开始评估...")
        report = runner.run_all(ground_truth)
        
        # 打印结果
        runner.print_report(report)
        
        # 保存报告
        save_path = runner.save_report(report)
        print(f"报告已保存到: {save_path}")
    
    return report


def show_status():
    """显示当前数据状态"""
    from evaluation.config import EvaluationConfig
    import json
    
    print("=" * 70)
    print("评估数据状态")
    print("=" * 70)
    
    config = EvaluationConfig()
    
    # 1. 检查 source papers
    if config.source_file.exists():
        with open(config.source_file, "r") as f:
            papers_count = sum(1 for _ in f)
        print(f"\n📄 Source Papers: {config.source_file}")
        print(f"   论文数量: {papers_count}")
    else:
        print(f"\n📄 Source Papers: 不存在")
    
    # 2. 检查 chunks
    chunks_dir = config.chunks_dir / "paragraph"
    if chunks_dir.exists():
        chunk_files = list(chunks_dir.glob("*.json"))
        total_chunks = 0
        for f in chunk_files:
            with open(f, "r") as file:
                data = json.load(file)
                total_chunks += len(data.get("chunks", []))
        print(f"\n📦 Chunks (paragraph): {chunks_dir}")
        print(f"   论文数: {len(chunk_files)}")
        print(f"   总 chunks: {total_chunks}")
    else:
        print(f"\n📦 Chunks: 不存在")
    
    # 3. 检查 ground truth
    if config.ground_truth_file.exists():
        with open(config.ground_truth_file, "r") as f:
            gt = json.load(f)
        qa_pairs = gt.get("qa_pairs", [])
        print(f"\n❓ Ground Truth: {config.ground_truth_file}")
        print(f"   QA 数量: {len(qa_pairs)}")
        print(f"   难度分布: {gt.get('difficulty_distribution', {})}")
    else:
        print(f"\n❓ Ground Truth: 不存在")
    
    # 4. 检查评估 collection
    try:
        from rag.milvus import MilvusProvider
        from evaluation.data_preparation.collection_builder import CollectionBuilder
        from evaluation.config import ChunkStrategy
        
        builder = CollectionBuilder(config)
        stats = builder.get_collection_stats(ChunkStrategy.PARAGRAPH)
        if stats:
            print(f"\n🗄️ Eval Collection: {stats.name}")
            print(f"   总记录: {stats.total_records}")
            print(f"   论文数: {stats.total_papers}")
            print(f"   索引类型: {stats.index_type}")
        else:
            print(f"\n🗄️ Eval Collection: 不存在或未初始化")
    except Exception as e:
        print(f"\n🗄️ Eval Collection: 检查失败 ({e})")
    
    # 5. 检查报告
    reports_dir = config.reports_dir
    if reports_dir.exists():
        reports = list(reports_dir.glob("report_*.json"))
        print(f"\n📊 Reports: {reports_dir}")
        print(f"   报告数量: {len(reports)}")
        if reports:
            # 显示最近的报告
            latest = max(reports, key=lambda x: x.stat().st_mtime)
            print(f"   最新报告: {latest.name}")
    else:
        print(f"\n📊 Reports: 不存在")
    
    print("\n" + "=" * 70)


def run_full_pipeline(sample_size: int = None, num_questions: int = 50, run_l3: bool = True):
    """运行完整流程"""
    print("\n" + "=" * 70)
    print("完整评估流程")
    print("=" * 70)
    print(f"  样本大小: {sample_size if sample_size else '全量'}")
    print(f"  问题数量: {num_questions}")
    print(f"  L3 评估: {'启用' if run_l3 else '禁用'}")
    print("=" * 70)
    
    # Step 1: 数据准备
    step1_export_and_prepare(sample_size=sample_size, drop_existing=True)
    
    # Step 2: 生成 QA
    step2_generate_qa(num_questions=num_questions)
    
    # Step 3: 运行评估
    report = step3_run_evaluation(run_l3=run_l3)
    
    print("\n" + "=" * 70)
    print("✓ 评估流程完成!")
    print("=" * 70)
    
    return report


def run_comparison(
    sample_size: int = None,
    num_questions: int = 50,
    run_l3: bool = True,
    skip_data_preparation: bool = False,
    resume: bool = True,
    clear_cache: bool = False
):
    """
    运行完整对比实验
    
    12 种配置 = 2 (chunk) × 3 (index) × 2 (agentic)
    
    Args:
        resume: 是否从缓存恢复（跳过已完成的实验）
        clear_cache: 清除缓存后重新运行
    """
    from models import get_llm_by_usage
    from evaluation.config import EvaluationConfig
    from evaluation.comparison_runner import ComparisonRunner
    
    print("\n" + "=" * 70)
    print("完整对比实验")
    print("=" * 70)
    
    config = EvaluationConfig()
    all_experiments = config.get_all_experiments()
    
    print(f"  实验配置数: {len(all_experiments)}")
    print(f"  Chunk 策略: {[c.value for c in config.chunk_strategies]}")
    print(f"  Index 类型: {[i.value for i in config.index_types]}")
    print(f"  Agentic 模式: [False, True]")
    print(f"  样本大小: {sample_size if sample_size else '全量'}")
    print(f"  问题数量: {num_questions}")
    print(f"  L3 评估: {'启用' if run_l3 else '禁用'}")
    print(f"  断点续跑: {'启用' if resume else '禁用'}")
    print("=" * 70)
    
    # 初始化 LLM
    llm = get_llm_by_usage('evaluation')
    
    # 创建对比运行器
    runner = ComparisonRunner(llm_client=llm, config=config)
    
    # 清除缓存（如果需要）
    if clear_cache:
        print("\n⚠️  清除实验缓存...")
        runner.clear_cache()
    
    # Step 1: 数据准备（如果需要）
    if not skip_data_preparation:
        print("\n" + "-" * 70)
        print("Step 1: 数据准备")
        print("-" * 70)
        runner.prepare_data(sample_size=sample_size)
    
    # Step 2: 生成 QA（如果需要）
    if not config.ground_truth_file.exists():
        print("\n" + "-" * 70)
        print("Step 2: 生成 QA")
        print("-" * 70)
        step2_generate_qa(num_questions=num_questions)
    
    # Step 3: 运行所有实验
    print("\n" + "-" * 70)
    print("Step 3: 运行对比实验")
    print("-" * 70)
    
    comparison = runner.run_all_experiments(
        run_l3=run_l3,
        skip_data_preparation=True,  # 已经在上面准备好了
        resume=resume
    )
    
    # 打印和保存结果
    runner.print_comparison(comparison)
    save_path = runner.save_comparison(comparison)
    
    print("\n" + "=" * 70)
    print("✓ 对比实验完成!")
    print(f"  报告: {save_path}")
    print(f"  Markdown: {save_path.with_suffix('.md')}")
    print("=" * 70)
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="完整评估流程")
    
    # 运行模式
    parser.add_argument("--compare", action="store_true",
                       help="运行完整对比实验 (12种配置)")
    parser.add_argument("--full", action="store_true", 
                       help="运行单配置完整流程 (export → qa → eval)")
    parser.add_argument("--prepare-only", action="store_true",
                       help="只运行 Step 1: 数据准备")
    parser.add_argument("--generate-qa", action="store_true",
                       help="只运行 Step 2: QA 生成")
    parser.add_argument("--eval-only", action="store_true",
                       help="只运行 Step 3: 评估")
    parser.add_argument("--status", action="store_true",
                       help="显示当前数据状态")
    
    # 参数
    parser.add_argument("--sample", type=int, default=None,
                       help="抽样数量 (默认全量)")
    parser.add_argument("--num-questions", type=int, default=50,
                       help="生成问题数量 (默认 50)")
    parser.add_argument("--no-l3", action="store_true",
                       help="跳过 L3 评估 (更快)")
    parser.add_argument("--drop-existing", action="store_true",
                       help="删除已有的评估 collection")
    parser.add_argument("--skip-data-prep", action="store_true",
                       help="跳过数据准备（使用已有数据）")
    parser.add_argument("--no-resume", action="store_true",
                       help="不使用缓存，从头运行所有实验")
    parser.add_argument("--clear-cache", action="store_true",
                       help="清除实验缓存后运行")
    parser.add_argument("--all-strategies", action="store_true",
                       help="准备所有 chunk 策略 (paragraph + contextual)")
    
    args = parser.parse_args()
    
    run_l3 = not args.no_l3
    
    if args.status:
        show_status()
    elif args.compare:
        run_comparison(
            sample_size=args.sample,
            num_questions=args.num_questions,
            run_l3=run_l3,
            skip_data_preparation=args.skip_data_prep,
            resume=not args.no_resume,
            clear_cache=args.clear_cache
        )
    elif args.full:
        run_full_pipeline(
            sample_size=args.sample,
            num_questions=args.num_questions,
            run_l3=run_l3
        )
    elif args.prepare_only:
        step1_export_and_prepare(
            sample_size=args.sample,
            drop_existing=args.drop_existing,
            all_strategies=args.all_strategies
        )
    elif args.generate_qa:
        step2_generate_qa(num_questions=args.num_questions)
    elif args.eval_only:
        step3_run_evaluation(run_l3=run_l3)
    else:
        # 默认显示帮助
        parser.print_help()


if __name__ == "__main__":
    main()
