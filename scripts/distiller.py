import argparse
import copy
import json
import re
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 假设你使用 OpenAI 格式的 API (可以是 GPT-4o 或 DeepSeek)
from openai import OpenAI

# 配置你的 API
client = OpenAI(api_key="sk-3c75accf241e412daa7257518e0255fe", base_url="https://api.deepseek.com/v1")

SYSTEM_PROMPT = """You MUST follow these rules strictly:
1) Before calling any tool / function, you MUST output a <thought>...</thought> block FIRST.
     - The <thought> content MUST include:
         (a) analysis of the current Observation / context
         (b) the next search strategy (what you will search / grep / find, and why)
     - Only AFTER the </thought> closing tag may you issue a tool call.

2) Final answer format:
     - Output the JSON result inside a fenced code block: ```json { ... } ```

You are a CSS Selector generator expert.
TASK: Given a webpage, generate robust CSS selectors.

TARGET MODE: list. 
Means you must identify the repeating list structure and generate a selector that targets the **specific element representing the core content** (usually the title or subject) within each list item.
Do NOT just select the container (like 'ul' or 'div.list'). You must go deep to the text node element.

# AVAILABLE TOOLS
1) bash_exec(cmd) -> str
   (Use grep, wc, head to analyze the file structure)
   
When you want an command execution, you should use <bash_exec>cmd</bash_exec> to interact with sandbox.

# OUTPUT FORMAT
{"file": "...", "target_selector": "...", "target": "..."}
"""

JSON_FENCE_RE = re.compile(r"```json", re.IGNORECASE)
THOUGHT_BLOCK_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL | re.IGNORECASE)
BASH_BLOCK_RE = re.compile(r"<bash_exec>(.*?)</bash_exec>", re.DOTALL)

def execute_bash(cmd: str, working_dir: Path) -> str:
    """真实执行 Bash 命令"""
    try:
        # 限制命令只读，防止 rm -rf 等意外
        if any(x in cmd for x in [">", "rm ", "mv ", "cp "]):
            return "Error: Read-only mode. Only grep, wc, head, cat allowed."
        
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=working_dir, 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        output = result.stdout[:2000] # 截断过长输出
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        return output.strip() or "(No output)"
    except Exception as e:
        return f"Execution Error: {str(e)}"

def message_contains_final_json(content: str) -> bool:
    """Check if the assistant content has a ```json block outside <thought>."""
    if not content:
        return False
    stripped = THOUGHT_BLOCK_RE.sub("", content)
    return bool(JSON_FENCE_RE.search(stripped))

def conversation_finished(messages: List[Dict[str, str]]) -> bool:
    """Only the latest assistant turn determines if the convo is finished."""
    if not messages:
        return False
    last_msg = messages[-1]
    if last_msg.get("role") != "assistant":
        return False
    return message_contains_final_json(last_msg.get("content", ""))

def parse_bash_commands(response_content: str) -> List[str]:
    """Extract all bash commands between <bash_exec> tags."""
    if not response_content:
        return []
    open_tags = response_content.count("<bash_exec>")
    close_tags = response_content.count("</bash_exec>")
    parse_text = response_content
    if close_tags < open_tags:
        parse_text = parse_text + "</bash_exec>" * (open_tags - close_tags)
    commands = []
    for match in BASH_BLOCK_RE.findall(parse_text):
        command = match.strip()
        if command:
            commands.append(command)
    return commands

def prepare_sandbox(html_path: Path, sandbox_dir: Path) -> None:
    """Create sandbox_dir and copy html_path to tmp.html inside it."""
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir)
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(html_path, sandbox_dir / "tmp.html")

def cleanup_sandbox(sandbox_dir: Path) -> None:
    """Remove sandbox directory if it exists."""
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir)

def run_distillation(html_path: Path, meta_path: Path, output_file: Path):
    """核心循环：Teacher Model 在沙箱中解题"""
    
    # 1. 准备沙箱环境
    sandbox_dir = Path("tmp_sandbox")
    sandbox_dir.mkdir(exist_ok=True)
    shutil.copy(html_path, sandbox_dir / "tmp.html") # 假装已经下载好了
    
    # 加载正确答案用于验证 (可选)
    meta = json.loads(meta_path.read_text())
    ground_truth_selector = meta["target_selector"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Please generate HTML selectors for the local file: tmp.html. Target mode: list. The file is already prepared in the current directory."}
    ]

    # 记录完整的 log 用于训练
    full_log_entry = {
        "prompt": messages[:2], # 初始 prompt
        "messages": list(messages), # 这是一个引用，会随着循环更新
        "ground_truth": meta,
        "metadata": {"source": "synthetic_distillation"}
    }

    print(f"Start distilling: {html_path.name}...")
    
    # 2. 交互循环 (最多 10 轮防止死循环)
    for turn in range(30):
        # 调用 Teacher Model
        completion = client.chat.completions.create(
            model="deepseek-chat", # 或 deepseek-chat
            messages=messages,
            temperature=0.2, # 让老师稳一点
            stop=["</bash_exec>"] # 辅助停止
        )
        
        response_content = completion.choices[0].message.content
        
        # 补全 stop token (因为 API 可能会截断)
        if "<bash_exec>" in response_content and "</bash_exec>" not in response_content:
            response_content += "</bash_exec>"

        # 将老师的回复加入历史
        new_msg = {"role": "assistant", "content": response_content}
        messages.append(new_msg)
        
        # 3. 解析回复
        # Case A: 结束了 (输出了 json)
        if "```json" in response_content:
            print(f"  ✅ Finished in {turn+1} turns.")
            break
            
        # Case B: 调用工具
        if "<bash_exec>" in response_content:
            try:
                # 简单的 XML 解析
                xml_part = response_content.split("<bash_exec>")[1].split("</bash_exec>")[0]
                bash_command = xml_part
                
                tool_output = ""
                
                # --- MOCK TOOLS ---
                if bash_command:
                    print(f"    Executed: {bash_command}")
                    tool_output = execute_bash(bash_command, sandbox_dir)

                
                # --- 将工具输出喂回给模型 ---
                # 注意：你的 log 格式好像没有专门的 tool role，通常放在 user 里或者专门的 tool 块
                # 这里为了适配你的 SFT 格式，我们模拟 user 返回 observation
                obs_msg = {
                    "role": "user", 
                    "content": f"Observation:\n{tool_output}"
                }
                messages.append(obs_msg)
                
            except Exception as e:
                print(f"  ❌ Parse/Exec Error: {e}")
                break
    
    # 4. 保存数据
    # 这里把 messages 保存进 full_log_entry
    full_log_entry["messages"] = messages
    
    # 追加写入 jsonl
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(full_log_entry) + "\n")
    
    # 清理沙箱
    shutil.rmtree(sandbox_dir)

def run_repair_loop(
    original_messages: List[Dict[str, str]],
    sandbox_dir: Path,
    max_turns: int = 30,
) -> Tuple[List[Dict[str, str]], int]:
    """Continue a conversation until a valid final json block appears."""
    messages = copy.deepcopy(original_messages)
    repair_turns = 0

    existing_turns = sum(1 for msg in messages if msg.get("role") == "assistant")
    remaining_turns = max(0, max_turns - existing_turns)
    if remaining_turns <= 0:
        raise RuntimeError("No remaining turns available for repair.")

    for _ in range(remaining_turns):
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.2,
            stop=["</bash_exec>"],
        )
        response_content = completion.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": response_content})
        repair_turns += 1

        commands = parse_bash_commands(response_content)
        for bash_command in commands:
            tool_output = execute_bash(bash_command, sandbox_dir)
            obs_msg = {"role": "user", "content": f"Observation:\n{tool_output}"}
            messages.append(obs_msg)

        if message_contains_final_json(response_content):
            return messages, repair_turns

    raise RuntimeError("Repair loop exceeded max_turns without completion.")

def needs_repair(messages: List[Dict[str, str]]) -> bool:
    """Determine whether a sample still lacks the required final json output."""
    if not messages:
        return True
    return not conversation_finished(messages)

def resolve_html_path(entry: Dict, html_root: Path) -> Optional[Path]:
    """Find the path to the html file referenced by a dataset entry."""
    if "html_path" in entry:
        candidate = Path(entry["html_path"])
        return candidate if candidate.exists() else None
    metadata = entry.get("metadata") or {}
    if "html_path" in metadata:
        candidate = Path(metadata["html_path"])
        return candidate if candidate.exists() else None
    ground_truth = entry.get("ground_truth") or {}
    file_name = ground_truth.get("file")
    if file_name:
        candidate = html_root / file_name
        return candidate if candidate.exists() else None
    return None

def repair_dataset(
    input_path: Path,
    output_path: Path,
    html_root: Path,
    max_turns: int,
    limit: Optional[int] = None,
) -> None:
    """Repair unfinished samples line-by-line and write them to a new jsonl."""
    sandbox_dir = Path("tmp_sandbox")
    repaired = 0
    total = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            if limit is not None and total >= limit:
                break
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            messages = entry.get("messages", [])
            if needs_repair(messages):
                html_path = resolve_html_path(entry, html_root)
                if not html_path:
                    entry["repaired"] = False
                    entry["repair_turns"] = 0
                    entry["repair_error"] = "Missing html_path for repair."
                else:
                    prepare_sandbox(html_path, sandbox_dir)
                    try:
                        updated_messages, repair_turns = run_repair_loop(
                            messages, sandbox_dir, max_turns=max_turns
                        )
                        entry["messages"] = updated_messages
                        entry["repaired"] = True
                        entry["repair_turns"] = repair_turns
                        repaired += 1
                    finally:
                        cleanup_sandbox(sandbox_dir)
            else:
                entry.setdefault("repaired", False)
                entry.setdefault("repair_turns", 0)

            dst.write(json.dumps(entry) + "\n")
            total += 1

    print(f"Repair completed: {repaired} / {total} samples updated.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distillation and repair utility.")
    parser.add_argument("--repair", action="store_true", help="Repair existing dataset entries.")
    parser.add_argument("--input", type=Path, default=Path("data/sft_train_data.jsonl"), help="Input jsonl for repair.")
    parser.add_argument("--output", type=Path, default=None, help="Output jsonl path.")
    parser.add_argument("--html-root", type=Path, default=Path("data/synthetic_complex"), help="Directory with html files.")
    parser.add_argument("--max-repair-turns", type=int, default=30, help="Max assistant turns during repair.")
    parser.add_argument("--repair-limit", type=int, default=None, help="Optional limit on repaired samples.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.output is None:
        if args.repair:
            args.output = Path("data/sft_train_data.repaired.jsonl")
        else:
            args.output = Path("data/sft_train_data.jsonl")

    if args.repair:
        repair_dataset(
            input_path=args.input,
            output_path=args.output,
            html_root=args.html_root,
            max_turns=args.max_repair_turns,
            limit=args.repair_limit,
        )
    else:
        data_dir = args.html_root
        output_dataset = args.output
        files = sorted(list(data_dir.glob("*.html")))
        for html_file in files[:1000]:
            json_file = html_file.with_suffix(".json")
            if json_file.exists():
                run_distillation(html_file, json_file, output_dataset)
