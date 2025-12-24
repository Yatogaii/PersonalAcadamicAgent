import json
import logging
import os
import pprint
import threading
from typing import Any

from agents.coordinator import invoke_coordinator
from prompts.template import apply_prompt_template

from dotenv import load_dotenv
from settings import settings
from logging_config import PLAIN_LOG_FORMAT, logger, setup_logging
from utils import extract_text_from_message_content

try:
    load_dotenv()
except Exception as e:
    exit(0)
    
enable_clarification = False
MAX_CLARIFICATION_ROUNDS = 3

def workflow(user_input:str):
    # Parse user-input
    user_input = user_input.strip()

    # Invoke Coordinator Agent with clarification
    is_clarification_complete = False
    while not is_clarification_complete and enable_clarification:
        final_state = invoke_coordinator(user_input, enable_clarification)

        # Summary current clarification Messages
        if final_state["need_clarification"] and final_state["clarification_times"]> MAX_CLARIFICATION_ROUNDS:
            pass

    # Get an summary for all clarification rounds.
    # clarification_complete will be True iff clarification is enabled and is completed.
    if is_clarification_complete:
        pass

    res = invoke_coordinator(user_input, False)
    return res


def _format_answer(result: Any) -> str:
    if isinstance(result, tuple):
        if len(result) == 1:
            return _format_answer(result[0])
        return "\n".join(_format_answer(item) for item in result)
    if isinstance(result, list):
        if all(isinstance(item, str) for item in result):
            return "\n".join(item.strip() for item in result if item)
        if all(isinstance(item, dict) and "text" in item for item in result):
            return extract_text_from_message_content(result)
        return pprint.pformat(result, width=120, compact=True)
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False, indent=2)
    if isinstance(result, str):
        stripped = result.strip()
        try:
            parsed = json.loads(stripped)
        except Exception:
            return stripped
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    return extract_text_from_message_content(result)


def run_tui() -> None:
    try:
        from textual.app import App, ComposeResult
        from textual.widgets import Input, Markdown, RichLog
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: textual. Install with `pip install textual`."
        ) from exc

    class CoordinatorApp(App):
        CSS = """
        Screen { layout: vertical; }
        #answer { height: 1fr; padding: 1 2; }
        #log { height: 12; display: none; }
        #input { height: auto; }
        Screen.show-logs #log { display: block; }
        """

        BINDINGS = [
            ("ctrl+q", "quit", "Quit"),
            ("ctrl+l", "toggle_logs", "Logs"),
        ]

        def compose(self) -> ComposeResult:
            yield Markdown(id="answer")
            yield RichLog(id="log")
            yield Input(id="input", placeholder="输入你的问题，回车提交...")

        def on_mount(self) -> None:
            self.answer_history = []
            self.log_widget = self.query_one("#log", RichLog)
            self.answer_widget = self.query_one("#answer", Markdown)
            os.makedirs("logs", exist_ok=True)
            setup_logging(
                level="INFO",
                sink=self._emit_log,
                log_format=PLAIN_LOG_FORMAT,
                enqueue=False,
            )
            if hasattr(logger, "add"):
                logger.add(
                    os.path.join("logs", "tui.log"),
                    level="INFO",
                    format=PLAIN_LOG_FORMAT,
                    enqueue=False,
                )
            else:
                file_handler = logging.FileHandler(os.path.join("logs", "tui.log"))
                file_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
                    )
                )
                logger.addHandler(file_handler)
            self.answer_widget.update("Ready. Ctrl+Q to quit. Ctrl+L to toggle logs.")

        def _emit_log(self, message) -> None:
            text = str(message).rstrip("\n")
            self.call_from_thread(self.log_widget.write, text)

        def _emit_answer(self, question: str, text: str) -> None:
            entry = f"### Q\n{question}\n\n### A\n{text}"
            self.answer_history.append(entry)
            content = "\n\n---\n\n".join(self.answer_history)
            self.call_from_thread(self.answer_widget.update, content)

        def on_input_submitted(self, event: Input.Submitted) -> None:
            user_text = event.value.strip()
            event.input.value = ""
            if not user_text:
                return
            threading.Thread(
                target=self._run_workflow,
                args=(user_text,),
                daemon=True,
            ).start()

        def _run_workflow(self, user_text: str) -> None:
            try:
                result = workflow(user_text)
                if result is None:
                    return
                formatted = _format_answer(result)
                self._emit_answer(user_text, formatted)
            except Exception:
                logger.exception("Workflow failed.")

        def action_toggle_logs(self) -> None:
            self.screen.toggle_class("show-logs")

    CoordinatorApp().run()


if __name__ == "__main__":
    run_tui()
