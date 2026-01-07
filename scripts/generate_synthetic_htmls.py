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

UTILITY_CLASSES = set(TAILWIND_POOL)

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
    forced_structure: Tuple[str, str] | None = None,
    variant: str | None = None
) -> Dict:
    """
    生成一个列表区块。
    返回字典包含: html, item_class, title_tag, structure
    """
    structure = forced_structure or random.choice(LIST_STRUCTURES)
    wrapper_tag, item_tag = structure

    # 决定这个 List 的风格
    style_mode = random.choice(["bem", "tailwind", "hash"])
    variant = variant or random.choice(["list", "cards", "table", "dl"])

    container_class = generate_random_class_string(style_mode)
    item_class = forced_item_class or generate_random_class_string(style_mode)
    title_tag = random.choice(["h3", "h4", "span", "div", "strong"])
    
    items_html = []
    item_count = random.randint(8, 30)
    item_data_attr = None

    # 40% 的概率给 item 打 data-* 标签，便于生成更鲁棒的 selector
    if random.random() < 0.4:
        key = random.choice(DATA_ATTR_POOL)
        val = fake.word()
        item_data_attr = f'{key}="{val}"'
    
    for idx in range(item_count):
        title_html = f'<{title_tag} class="{generate_random_class_string(style_mode)}">{fake.sentence(nb_words=3).rstrip(".")}</{title_tag}>'
        price_html = f'<span class="{generate_random_class_string(style_mode)}">${random.uniform(10, 100):.2f}</span>'
        extra_attrs = generate_attributes(fake)

        # 随机隐藏或骨架占位
        visibility_attr = ""
        item_class_aug = item_class
        if random.random() < 0.15:
            visibility_attr = 'style="display:none"'
        if random.random() < 0.2:
            item_class_aug += " skeleton loading"

        if item_data_attr:
            extra_attrs = f"{item_data_attr} {extra_attrs}".strip()

        extra_text = "".join(f"<p>{fake.paragraph(nb_sentences=2)}</p>" for _ in range(random.randint(0, 2)))

        if variant == "table":
            row_html = f"""
            <tr class="{item_class_aug}" {extra_attrs} {visibility_attr}>
                <td>{idx+1}</td>
                <td>{title_html}</td>
                <td>{price_html}</td>
                <td><span>{fake.word()}</span>{extra_text}</td>
            </tr>
            """
            items_html.append(row_html)
        elif variant == "dl":
            row_html = f"""
            <dt class="{item_class_aug}" {extra_attrs} {visibility_attr}>{title_html}</dt>
            <dd><span class="{generate_random_class_string(style_mode)}">{fake.text(max_nb_chars=30)}</span> {price_html} {extra_text}</dd>
            """
            items_html.append(row_html)
        elif variant == "cards":
            card_body = f"""
                <header>{title_html}</header>
                <div class="meta">{price_html}</div>
                <p>{fake.paragraph(nb_sentences=1)}</p>
                {extra_text}
            """
            items_html.append(
                f'<article class="{item_class_aug}" {extra_attrs} {visibility_attr}>\n{card_body}\n</article>'
            )
        else:  # list
            item_inner = f"""
                {title_html}
                {price_html}
                <p>{fake.word()}</p>
                {extra_text}
            """
            items_html.append(
                f'<{item_tag} class="{item_class_aug}" {extra_attrs} {visibility_attr}>\n{item_inner}\n</{item_tag}>'
            )

    if variant == "table":
        full_html = f"""
        <table class="{container_class}">
            <thead><tr><th>#</th><th>Title</th><th>Price</th><th>Note</th></tr></thead>
            <tbody>
                {'\n'.join(items_html)}
            </tbody>
        </table>
        """
    elif variant == "dl":
        full_html = f"""
        <dl class="{container_class}">
            {'\n'.join(items_html)}
        </dl>
        """
    else:
        full_html = f"""
        <{wrapper_tag} class="{container_class}">
            {'\n'.join(items_html)}
        </{wrapper_tag}>
        """
    
    # 30% 概率增加无意义包裹层 (Wrapper Hell)
    if random.random() < 0.3:
        full_html = wrap_in_useless_divs(full_html, depth=random.randint(1, 3))

    return {
        "html": full_html,
        "item_class": item_class,
        "title_tag": title_tag,
        "structure": structure,
        "count": item_count,
        "item_attr": item_data_attr,
        "variant": variant
    }

def _pick_semantic_class(cls_string: str) -> str:
    tokens = cls_string.split()
    for tok in tokens:
        if tok not in UTILITY_CLASSES:
            return tok
    return tokens[-1] if tokens else "item"


def build_complex_page(fake: Faker) -> Tuple[str, str]:
    """
    构建包含对抗样本的复杂页面，加入多布局、隐藏节点、骨架和多重陷阱。
    """
    
    target_data = generate_list_section(fake)
    target_id = "main-results"

    trap_html_blocks = []
    trap_count = random.randint(1, 2) if random.random() < 0.7 else 0
    trap_ids = ["sidebar-recommendations", "footer-recos"]

    for i in range(trap_count):
        trap_data = generate_list_section(
            fake,
            forced_item_class=target_data["item_class"],
            forced_structure=target_data["structure"],
            variant=target_data["variant"]
        )
        trap_html_blocks.append(
            f'<aside id="{trap_ids[i % len(trap_ids)]}">\n<h3>Recommendations</h3>\n{trap_data["html"]}\n</aside>'
        )

    # 优先用 data-* 属性构造 selector，其次用语义 class
    selector_key = None
    if target_data.get("item_attr"):
        selector_key = f'[{target_data["item_attr"]}] {target_data["title_tag"]}'
    else:
        semantic_cls = _pick_semantic_class(target_data["item_class"])
        selector_key = f'.{semantic_cls} {target_data["title_tag"]}'

    parent_prefix = f'#{target_id} '
    final_selector = parent_prefix + selector_key

    main_html = f'<div id="{target_id}">\n<h1>Search Results</h1>\n{target_data["html"]}\n</div>'

    noise_top = generate_list_section(fake)["html"] if random.random() < 0.3 else ""
    long_noise = ""
    if random.random() < 0.5:
        long_noise = "\n".join(
            f"<section class=\"noise\"><h2>{fake.sentence(nb_words=5)}</h2><p>{fake.text(max_nb_chars=600)}</p><p>{fake.text(max_nb_chars=400)}</p></section>"
            for _ in range(random.randint(2, 5))
        )
    script_noise = """
    <script>
    // Simulate lazy append
    document.addEventListener('DOMContentLoaded', () => {
        const n = document.createElement('div');
        n.className = 'lazy-placeholder';
        n.innerHTML = '<p>Loading...</p>';
        document.body.appendChild(n);
    });
    </script>
    """

    body_content = f"""
    <header><nav>...</nav></header>
    <div class="container">
        {noise_top}
        {long_noise}
        <div class="layout-grid">
            {main_html}
            {'\n'.join(trap_html_blocks)}
        </div>
    </div>
    <footer>...</footer>
    {script_noise}
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
    generate_dataset(10, Path("data/synthetic_complex"))