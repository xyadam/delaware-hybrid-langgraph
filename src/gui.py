###########################################################################
##                            IMPORTS
###########################################################################

import json
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from dotenv import load_dotenv
from rich.columns import Columns
from rich.markdown import Markdown
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from textual import on
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, RadioButton, RadioSet, RichLog, Static

load_dotenv()

###########################################################################
##                        CUSTOM IMPORTS
###########################################################################

from agent import APP_DB, workflow


###########################################################################
##                           FUNCTIONS
###########################################################################

def _format_checkpoint_datetime(checkpoint_id: str) -> str:
    parts = checkpoint_id.split("-")
    if len(parts) < 3:
        return "Unknown Date"
    try:
        time_high = int(parts[0], 16)
        time_mid = int(parts[1], 16)
        time_low = int(parts[2], 16) & 0x0FFF
        timestamp_100ns = (time_high << 28) | (time_mid << 12) | time_low
        dt_utc = datetime(1582, 10, 15, tzinfo=timezone.utc) + timedelta(microseconds=timestamp_100ns / 10)
        return dt_utc.astimezone().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "Unknown Date"


def get_existing_threads() -> list[dict[str, str]]:
    if not APP_DB.exists():
        return []
    conn = sqlite3.connect(str(APP_DB))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'")
    if not cursor.fetchone():
        conn.close()
        return []
    cursor.execute(
        """
        SELECT thread_id, MAX(checkpoint_id) AS last_checkpoint
        FROM checkpoints
        GROUP BY thread_id
        ORDER BY MAX(rowid) DESC
        """
    )
    threads = []
    for thread_id, last_checkpoint in cursor.fetchall():
        if not thread_id:
            continue
        threads.append(
            {
                "thread_id": thread_id,
                "label": _format_checkpoint_datetime(last_checkpoint) if last_checkpoint else "Unknown Date",
            }
        )
    conn.close()
    return threads


def new_thread_id() -> str:
    return str(uuid.uuid4())[:8]


def content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(block.get("text", "") for block in content if isinstance(block, dict)).strip()
    return str(content)


EXAMPLE_QUERIES = [
    "What are you capable of doing?",
    "What was the profit and number of items sold in Beijing, each month in 2024?",
    "What are the main countries where we are selling our products, and how many employees do we have in those countries?",
    "Which of our stores performed the best in 2024?",
    "In Germany what was the most sold product in 2024 and what does it's style card say?",
    "In USA if you check the detailed style infos of the 3 most sold products, can you realize something that is common between these items that could induce the large sales?",
    "Whats the material of the 5 most sold products in Germany in 2024?",
    "Which of our employee generate the most revenue in 2024?",
]

WELCOME_MESSAGE = (
    "Welcome! I am your hybrid retail analyst for 2024 sales + product style knowledge. "
    "Pick an example query or ask your own question."
)

WELCOME_PANELS = [
    (
        "Sales Performance",
        "- Best stores by revenue\n- Country and city comparisons\n- Category winners by market\n- Monthly trend breakdowns",
    ),
    (
        "Product Intelligence",
        "- Most sold products by region\n- Product-level revenue analysis\n- Top products by category\n- Return rate checks",
    ),
    (
        "Style Card RAG",
        "- Material and fabric details\n- Care and washing guidance\n- Sustainability and origin notes\n- Size and styling recommendations",
    ),
]


class ExampleQueriesScreen(ModalScreen[str | None]):
    DEFAULT_CSS = """
    ExampleQueriesScreen {
        align: center middle;
        background: #000000 45%;
    }
    #example_popup {
        width: 90%;
        max-width: 110;
        height: auto;
        max-height: 80%;
        border: solid #6a4530;
        background: #2a1a13;
        padding: 1;
    }
    #example_popup_title {
        text-style: bold;
        color: white;
        margin: 0 0 1 0;
    }
    .popup_query_btn {
        width: 100%;
        margin: 0 0 1 0;
        background: #1b3f73;
        color: white;
        border: solid #2b5d9a;
    }
    #close_popup_btn {
        width: 100%;
        background: #4b2e1f;
        color: white;
        border: solid #6a4530;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="example_popup"):
            yield Static("Select an Example Query", id="example_popup_title")
            for index, query in enumerate(EXAMPLE_QUERIES, start=1):
                label = f"{index}. {query}"
                yield Button(label, name=query, classes="popup_query_btn")
            yield Button("Close", id="close_popup_btn")

    @on(Button.Pressed, ".popup_query_btn")
    def on_query_selected(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.name)

    @on(Button.Pressed, "#close_popup_btn")
    def on_close(self) -> None:
        self.dismiss(None)


###########################################################################
##                               GUI APP
###########################################################################

class RetailAgentGui(App):
    TITLE = "Fashion Retail Assistant"
    CSS = """
    #body { height: 1fr; }
    #chat_column { width: 3fr; padding: 0 1; }
    #sessions_column { width: 1fr; border-left: solid $panel; padding: 0 1; }
    #chat_log { height: 1fr; border: solid $panel; }
    #chat_input { margin-top: 1; }
    #depth_title { text-style: bold; margin: 1 0 0 0; }
    #depth_selector { height: auto; margin: 0 0 1 0; }
    #sessions_title { text-style: bold; margin: 1 0 1 0; }
    .full_btn { width: 100%; margin: 0 0 1 0; }
    #example_queries_toggle {
        background: #4b2e1f;
        color: white;
        border: solid #6a4530;
    }
    #sessions_separator {
        height: 1;
        margin: 0 0 1 0;
        border-top: solid $panel;
    }
    #sessions_list { height: 1fr; }
    .thread_btn { width: 100%; margin: 0 0 1 0; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.agent: Any = None
        self.conn = None
        self.initial_state: dict = {"depth": 3, "max_iterations": 6}
        self.current_thread_id = ""
        self.current_session_label = ""
        self.startup_intro_visible = False
        self.depth = 3

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="chat_column"):
                yield RichLog(id="chat_log", wrap=True, highlight=True, markup=True)
                yield Input(placeholder="Ask a question and press Enter", id="chat_input")
            with Vertical(id="sessions_column"):
                yield Static("Depth", id="depth_title")
                with RadioSet(id="depth_selector"):
                    yield RadioButton("1 - Quick", value=False)
                    yield RadioButton("2 - Standard", value=False)
                    yield RadioButton("3 - Deep", value=True)
                yield Static("Previous Sessions", id="sessions_title")
                yield Button("New Session", id="new_session", variant="primary", classes="full_btn")
                yield Button("Example Queries", id="example_queries_toggle", variant="default", classes="full_btn")
                yield Static("", id="sessions_separator")
                yield VerticalScroll(id="sessions_list")
        yield Footer()

    def on_mount(self) -> None:
        APP_DB.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(APP_DB), check_same_thread=False)
        self.agent = workflow.compile(checkpointer=SqliteSaver(self.conn))
        self.current_thread_id = new_thread_id()
        self._load_thread(self.current_thread_id)

    def on_unmount(self) -> None:
        if self.conn:
            self.conn.close()

    def _chat_log(self) -> RichLog:
        return self.query_one("#chat_log", RichLog)

    def _chat_input(self) -> Input:
        return self.query_one("#chat_input", Input)

    def _thread_container(self) -> VerticalScroll:
        return self.query_one("#sessions_list", VerticalScroll)

    def _write_system_box(self, text: str) -> None:
        self._chat_log().write(Panel(text, border_style="cyan", title="Session"))

    def _write_user_box(self, text: str) -> None:
        self._chat_log().write(
            Panel(text, border_style="bright_black", title="You", style="on #3a3a3a")
        )

    def _write_assistant_box(self, text: str) -> None:
        parts = re.split(r"<tabledata>(.*?)</tabledata>", text, flags=re.DOTALL)
        renderables = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                try:
                    data = json.loads(part)
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        table = Table(show_header=True, header_style="bold magenta", expand=True)
                        for col in data[0].keys():
                            table.add_column(col)
                        for row in data:
                            table.add_row(*[str(row.get(col, "")) for col in data[0].keys()])
                        renderables.append(table)
                        continue
                except (json.JSONDecodeError, IndexError, TypeError):
                    pass
                renderables.append(Markdown(part.strip()))
            else:
                stripped = part.strip()
                if stripped:
                    renderables.append(Markdown(stripped))
        if not renderables:
            return
        self._chat_log().write(Panel(Group(*renderables), border_style="magenta", title="Assistant"))

    def _write_welcome_box(self, text: str) -> None:
        self._chat_log().write(
            Panel(Markdown(text), border_style="bright_black", title="Welcome", style="on #3a3a3a", expand=True)
        )

    def _write_welcome_panels(self) -> None:
        feature_panels = [
            Panel(body, title=title, border_style="bright_black", style="on #3a3a3a")
            for title, body in WELCOME_PANELS
        ]
        self._chat_log().write(Columns(feature_panels, equal=True, expand=True))
        self._write_spacer()
        self._chat_log().write(
            Panel(
                "Helper: if you want more ideas, ask me: what are you capable of doing",
                title="Quick Tip",
                border_style="cyan",
                style="on #2f2f2f",
            )
        )

    def _write_spacer(self) -> None:
        self._chat_log().write("")

    def _write_tool_box(self, step_number: int, tool_name: str, args_json: str) -> None:
        self._chat_log().write(
            Panel(
                Syntax(args_json, "json", theme="monokai", word_wrap=True),
                border_style="yellow",
                title=f"Tool Step {step_number}: {tool_name}",
            )
        )

    def _render_thread_buttons(self, threads: list[dict[str, str]]) -> None:
        container = self._thread_container()
        container.remove_children()
        label_map = {item["thread_id"]: item["label"] for item in threads}
        ordered_threads = [item["thread_id"] for item in threads if item["thread_id"] != self.current_thread_id]
        if self.current_thread_id:
            ordered_threads.insert(0, self.current_thread_id)

        for thread_id in ordered_threads:
            label = label_map.get(thread_id, datetime.now().strftime("%Y-%m-%d %H:%M"))
            if thread_id == self.current_thread_id:
                label = f"{label} (active)"
            container.mount(Button(label, name=thread_id, classes="thread_btn"))

    def _load_thread(self, thread_id: str) -> None:
        self.current_thread_id = thread_id
        self._chat_log().clear()
        label_map = {item["thread_id"]: item["label"] for item in get_existing_threads()}
        session_label = label_map.get(thread_id, datetime.now().strftime("%Y-%m-%d %H:%M"))
        self.current_session_label = session_label
        self._write_system_box(session_label)
        config = {"configurable": {"thread_id": thread_id}}
        state = self.agent.get_state(config)
        messages = state.values.get("messages", [])

        if not messages:
            self.startup_intro_visible = True
            self._write_welcome_box(WELCOME_MESSAGE)
            self._write_spacer()
            self._write_welcome_panels()
        else:
            self.startup_intro_visible = False

        for message in messages:
            msg_type = getattr(message, "type", "")
            text = content_to_text(getattr(message, "content", "")).strip()
            if not text:
                continue
            if msg_type == "human":
                self._write_user_box(text)
            elif msg_type == "ai":
                self._write_assistant_box(text)

        self._render_thread_buttons(get_existing_threads())

    @on(RadioSet.Changed, "#depth_selector")
    def on_depth_changed(self, event: RadioSet.Changed) -> None:
        self.depth = event.radio_set.pressed_index + 1
        self.initial_state["depth"] = self.depth
        self.initial_state["max_iterations"] = self.depth * 2

    @on(Button.Pressed, "#new_session")
    def on_new_session(self) -> None:
        self._load_thread(new_thread_id())

    @on(Button.Pressed, "#example_queries_toggle")
    def on_example_queries_toggle(self) -> None:
        self.push_screen(ExampleQueriesScreen(), self._on_example_query_selected)

    def _on_example_query_selected(self, selected_query: str | None) -> None:
        if selected_query:
            self._chat_input().value = selected_query
            self._chat_input().focus()

    @on(Button.Pressed, ".thread_btn")
    def on_thread_selected(self, event: Button.Pressed) -> None:
        thread_id = event.button.name or ""
        if not thread_id:
            return
        self._load_thread(thread_id)

    @on(Input.Submitted, "#chat_input")
    def on_chat_submit(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return

        if self.startup_intro_visible:
            self._chat_log().clear()
            self._write_system_box(self.current_session_label)
            self.startup_intro_visible = False

        self._chat_input().value = ""
        self._chat_input().disabled = True
        self._write_user_box(user_text)

        config = {"configurable": {"thread_id": self.current_thread_id}}
        state = self.agent.get_state(config)
        prev_count = len(state.values.get("messages", []))
        self.run_agent_turn(user_text, prev_count)

    @work(thread=True)
    def run_agent_turn(self, user_text: str, prev_count: int) -> None:
        config = {"configurable": {"thread_id": self.current_thread_id}}
        invoke_state = {
            **self.initial_state,
            "messages": [{"role": "user", "content": user_text}],
            "iteration": 0,
            "collected_results": [],
            "todo": [],
            "reflection": "",
            "rag_sources": [],
        }
        tool_step = 1
        try:
            for chunk in self.agent.stream(invoke_state, config, stream_mode="updates"):
              for node_name, update in chunk.items():
                messages = update.get("messages", [])

                # Stream tool calls live
                for msg in messages:
                    tool_calls = getattr(msg, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            name = tc.get("name", "unknown_tool")
                            args_json = json.dumps(tc.get("args", {}), indent=2, ensure_ascii=True)
                            self.call_from_thread(self._write_tool_box, tool_step, name, args_json)
                            tool_step += 1

                # Stream reflection status
                if node_name == "reflect":
                    satisfied = update.get("reflection_satisfied", False)
                    feedback = update.get("reflection", "")
                    status = "Satisfied" if satisfied else "Needs more data"
                    label = f"Reflect: {status}"
                    if feedback and not satisfied:
                        label += f"\n{feedback}"
                    self.call_from_thread(self._write_reflect_box, label)

                # Stream iteration marker
                if node_name == "plan" and "iteration" in update:
                    self.call_from_thread(self._write_iteration_marker, update["iteration"])

            self.call_from_thread(self._write_final_ai_messages, prev_count)
            self.call_from_thread(self._finish_turn)
        except Exception as error:
            self.call_from_thread(self._render_error, str(error))
            self.call_from_thread(self._finish_turn)

    def _write_final_ai_messages(self, prev_count: int) -> None:
        config = {"configurable": {"thread_id": self.current_thread_id}}
        state = self.agent.get_state(config)
        all_messages = state.values.get("messages", [])

        found_response = False
        for msg in all_messages[prev_count:]:
            if getattr(msg, "type", "") != "ai":
                continue
            if getattr(msg, "tool_calls", None):
                continue
            content = content_to_text(getattr(msg, "content", "")).strip()
            if content:
                self._write_assistant_box(content)
                found_response = True

        if not found_response:
            self._render_error("The AI model did not return a response. This may be due to an API issue or content filter.")

    def _write_reflect_box(self, text: str) -> None:
        self._chat_log().write(Panel(text, border_style="green", title="Reflect"))

    def _write_iteration_marker(self, iteration: int) -> None:
        self._chat_log().write(Panel(f"Iteration {iteration}", border_style="dim", title="Plan"))

    def _finish_turn(self) -> None:
        self._chat_input().disabled = False
        self._chat_input().focus()
        self._render_thread_buttons(get_existing_threads())

    def _render_error(self, error_message: str) -> None:
        self._chat_log().write(Panel(error_message, border_style="red", title="Error"))
        self._chat_input().disabled = False
        self._chat_input().focus()


###########################################################################
##                                MAIN
###########################################################################

def main() -> None:
    RetailAgentGui().run()


if __name__ == "__main__":
    main()
