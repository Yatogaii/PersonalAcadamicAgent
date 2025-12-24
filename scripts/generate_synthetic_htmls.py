#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path
from typing import List, Tuple, Dict

from faker import Faker

# --- 1. 配置池 ---

LIST_STRUCTURES = [
    ("ul", "li"),
    ("ol", "li"),
    ("div", "div"),
    ("div", "article"),
    ("section", "div"),
]

# 模拟 Tailwind CSS 的原子类
TAILWIND_POOL = [
    "flex", "grid", "p-2", "m-4", "text-sm", "font-bold", "bg-white", 
    "shadow-md", "rounded-lg", "border", "hover:bg-gray-100", "w-full", 
    "relative", "absolute", "z-10", "overflow-hidden", "flex-col"
]

# 常见的数据属性，用于训练属性选择器
DATA_ATTR_POOL = ["data-testid", "data-cy", "data-component", "data-id"]

# --- 2. 辅助函数 ---

def generate_random_class_string(mode: str = "mixed") -> str:
    """
    生成不同风格的 class 字符串。
    modes: 'bem' (语义化), 'tailwind' (原子类), 'hash' (乱码), 'mixed' (混合)
    """
    if mode == "mixed":
        mode = random.choice(["bem", "tailwind", "hash"])

    if mode == "tailwind":
        # 随机取 3-8 个原子类
        return " ".join(random.sample(TAILWIND_POOL, k=random.randint(3, 8)))
    
    elif mode == "hash":
        # 模拟 css-Modules: "css-1a2b3c"
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"css-{suffix}"
    
    else: # bem / semantic
        prefix = random.choice(["item", "card", "box", "prod", "entry"])
        suffix = "".join(random.choices(string.ascii_lowercase, k=3))
        return f"{prefix}-{suffix}"

def wrap_in_useless_divs(content: str, depth: int = 0) -> str:
    """
    模拟 React/Vue 常见的无意义嵌套： <div><div>...</div></div>
    """
    if depth == 0:
        return content
    
    # 有概率带 class，有概率是裸 div
    if random.random() < 0.5:
        wrapper = '<div class="wrapper">'
    else:
        wrapper = '<div>'
        
    return f"{wrapper}\n{wrap_in_useless_divs(content, depth - 1)}\n</div>"

def generate_attributes(fake: Faker) -> str:
    """生成除了 class/id 之外的属性，如 data-testid"""
    attrs = []
    if random.random() < 0.3:
        key = random.choice(DATA_ATTR_POOL)
        val = fake.word()
        attrs.append(f'{key}="{val}"')
    
    if random.random() < 0.2:
        attrs.append(f'aria-label="{fake.word()}"')
        
    return " ".join(attrs)

# --- 3. 核心生成逻辑 ---

def generate_list_section(
    fake: Faker, 
    forced_item_class: str | None = None, 
    forced_structure: Tuple[str, str] | None = None
) -> Dict:
    """
    生成一个列表区块。
    返回字典包含: html, item_class, title_tag, structure
    """
    structure = forced_structure or random.choice(LIST_STRUCTURES)
    wrapper_tag, item_tag = structure

    # 决定这个 List 的风格
    style_mode = random.choice(["bem", "tailwind", "hash"])
    
    container_class = generate_random_class_string(style_mode)
    item_class = forced_item_class or generate_random_class_string(style_mode)
    title_tag = random.choice(["h3", "h4", "span", "div"])
    
    items_html = []
    item_count = random.randint(3, 8)
    
    for _ in range(item_count):
        # 构造 Item 内部
        title_html = f'<{title_tag} class="{generate_random_class_string(style_mode)}">{fake.sentence(nb_words=3).rstrip(".")}</{title_tag}>'
        price_html = f'<span class="{generate_random_class_string(style_mode)}">${random.uniform(10, 100):.2f}</span>'
        
        # 随机属性
        extra_attrs = generate_attributes(fake)
        
        item_inner = f"""
            {title_html}
            {price_html}
            <p>{fake.word()}</p>
        """
        
        items_html.append(
            f'<{item_tag} class="{item_class}" {extra_attrs}>\n{item_inner}\n</{item_tag}>'
        )
    
    full_html = f"""
    <{wrapper_tag} class="{container_class}">
        {'\n'.join(items_html)}
    </{wrapper_tag}>
    """
    
    # 30% 概率增加无意义包裹层 (Wrapper Hell)
    if random.random() < 0.3:
        full_html = wrap_in_useless_divs(full_html, depth=random.randint(1, 2))

    return {
        "html": full_html,
        "item_class": item_class,
        "title_tag": title_tag,
        "structure": structure,
        "count": item_count
    }

def build_complex_page(fake: Faker) -> Tuple[str, str]:
    """
    构建包含对抗样本的复杂页面。
    策略：
    1. 生成一个 Target List (Main)。
    2. (高概率) 生成一个 Trap List (Sidebar/Footer)，它的 Class 和 Target List 完全一样，或者结构完全一样。
    3. 这样模型必须学会用 Parent ID 来区分。
    """
    
    # 1. 生成目标列表
    target_data = generate_list_section(fake)
    target_id = "main-results"
    
    # 2. 决定是否生成陷阱 (Adversarial Trap)
    has_trap = random.random() < 0.7
    trap_html = ""
    trap_id = "sidebar-recommendations"
    
    if has_trap:
        # 陷阱策略：使用和目标列表完全一样的 item class，但放在不同容器里
        trap_data = generate_list_section(
            fake, 
            forced_item_class=target_data["item_class"], # <--- 关键：Class 冲突
            forced_structure=target_data["structure"]
        )
        trap_html = f'<aside id="{trap_id}">\n<h3>Recommendations</h3>\n{trap_data["html"]}\n</aside>'
        
        # 因为有陷阱，Selector 必须限定父级 ID，否则会选多
        # 逻辑：#main-results .item-class > h3
        final_selector = f'#{target_id} .{target_data["item_class"].split()[-1]} > {target_data["title_tag"]}'
        # 注意：如果是 Tailwind 这种多 class，取最后一个或全部稍微复杂点，这里简化处理取最后一个作为特征
        # 在真实场景中，GPT-4 会分析出最具代表性的 class
    else:
        # 没有陷阱，Selector 可以宽泛一点
        final_selector = f'.{target_data["item_class"].split()[-1]} > {target_data["title_tag"]}'

    # 3. 组装页面
    main_html = f'<div id="{target_id}">\n<h1>Search Results</h1>\n{target_data["html"]}\n</div>'
    
    # 随机噪音
    noise_top = generate_list_section(fake)["html"] if random.random() < 0.3 else ""
    
    body_content = f"""
    <header><nav>...</nav></header>
    <div class="container">
        {noise_top}
        <div class="layout-grid">
            {main_html}
            {trap_html}
        </div>
    </div>
    <footer>...</footer>
    """
    
    document = f"""<!DOCTYPE html>
<html>
<head><title>{fake.catch_phrase()}</title></head>
<body>
{body_content}
</body>
</html>"""

    return document, final_selector

# --- 4. 生成入口 ---

def generate_dataset(count: int, output_dir: Path):
    fake = Faker("en_US")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(count):
        html, selector = build_complex_page(fake)
        (output_dir / f"{i:04d}.html").write_text(html, encoding="utf-8")
        
        meta = {
            "file": f"{i:04d}.html",
            "target_selector": selector,
            "task": "Extract titles from the main results list"
        }
        (output_dir / f"{i:04d}.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    generate_dataset(1000, Path("data/synthetic_complex"))