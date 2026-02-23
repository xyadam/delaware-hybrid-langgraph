###########################################################################
##                            IMPORTS
###########################################################################

import json
import re
import uuid
import sqlite3

from dotenv import load_dotenv
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt

load_dotenv()

###########################################################################
##                        CUSTOM IMPORTS
###########################################################################

from agent import workflow, APP_DB
from langgraph.checkpoint.sqlite import SqliteSaver


###########################################################################
##                           CONSTANTS
###########################################################################

console = Console()

DEPTH_LABELS = {
    1: "Quick    (max 2 tool rounds)",
    2: "Standard (max 4 tool rounds)",
    3: "Deep     (max 6 tool rounds)",
}


###########################################################################
##                       RENDERING FUNCTIONS
###########################################################################


def render_ai_content(text: str) -> Group:
    """Parse AI response, converting <tabledata> JSON blocks into Rich tables."""
    parts = re.split(r"<tabledata>(.*?)</tabledata>", text, flags=re.DOTALL)
    renderables = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            try:
                data = json.loads(part)
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    table = Table(show_header=True, header_style="bold cyan", expand=True)
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
    return Group(*renderables) if renderables else Markdown(text)


def display_node_update(node_name: str, update: dict) -> None:
    """Display a single streamed node update as it arrives."""
    messages = update.get("messages", [])

    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                args = tc.get("args", {})
                if tool_name == "query_sql":
                    console.print(f"\n  [dim]Tool: query_sql[/]")
                    console.print(Syntax(args.get("sql", ""), "sql", theme="monokai", padding=1))
                elif tool_name == "query_rag":
                    console.print(f"\n  [dim]Tool: query_rag[/]")
                    console.print(f"  [yellow]{args.get('question', '')}[/]")

    if node_name == "reflect":
        satisfied = update.get("reflection_satisfied", False)
        feedback = update.get("reflection", "")
        todo = update.get("todo", [])
        status = "[green]satisfied[/]" if satisfied else "[yellow]needs more data[/]"
        console.print(f"\n  [dim]Reflect:[/] {status}")
        if feedback and not satisfied:
            console.print(f"  [dim]{feedback}[/]")
        if todo and not satisfied:
            for t in todo:
                console.print(f"  [dim]  - {t}[/]")

    if node_name == "plan":
        iteration = update.get("iteration", 0)
        console.print(f"\n  [dim]--- Iteration {iteration} ---[/]")


###########################################################################
##                     SESSION SETUP FUNCTIONS
###########################################################################


def get_existing_threads() -> list[dict]:
    """Fetch all existing threads from the checkpoint DB."""
    if not APP_DB.exists():
        return []

    conn = sqlite3.connect(str(APP_DB))
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'")
    if not cursor.fetchone():
        conn.close()
        return []

    cursor.execute("""
        SELECT DISTINCT thread_id
        FROM checkpoints
        ORDER BY thread_id
    """)
    threads = [{"thread_id": row[0]} for row in cursor.fetchall()]
    conn.close()
    return threads


def select_thread() -> str:
    """Let the user pick an existing thread or start a new one."""
    threads = get_existing_threads()

    if not threads:
        thread_id = str(uuid.uuid4())[:8]
        console.print(f"\n  No existing threads. Starting new thread: [bold cyan]{thread_id}[/]")
        return thread_id

    console.print("\n[bold]Existing threads:[/]")
    table = Table(show_header=True, header_style="bold cyan", padding=(0, 2))
    table.add_column("#", style="dim", width=4)
    table.add_column("Thread ID", style="cyan")

    for i, t in enumerate(threads, 1):
        table.add_row(str(i), t["thread_id"])

    console.print(table)
    console.print(f"  [dim]Enter a number to resume, or press Enter for a new thread[/]\n")

    choice = Prompt.ask("  Select", default="")

    if choice.strip() == "" or not choice.strip().isdigit():
        thread_id = str(uuid.uuid4())[:8]
        console.print(f"  New thread: [bold cyan]{thread_id}[/]")
        return thread_id

    idx = int(choice.strip()) - 1
    if 0 <= idx < len(threads):
        thread_id = threads[idx]["thread_id"]
        console.print(f"  Resuming thread: [bold cyan]{thread_id}[/]")
        return thread_id

    thread_id = str(uuid.uuid4())[:8]
    console.print(f"  Invalid selection. New thread: [bold cyan]{thread_id}[/]")
    return thread_id


def select_depth() -> int:
    """Let the user pick analysis depth 1-3."""
    console.print("\n[bold]Analysis depth:[/]")
    for level, label in DEPTH_LABELS.items():
        console.print(f"  [cyan]{level}[/] - {label}")
    console.print()

    choice = Prompt.ask("  Depth", default="3")
    depth = int(choice.strip()) if choice.strip().isdigit() else 3
    depth = max(1, min(3, depth))
    console.print(f"  Using depth: [bold cyan]{depth}[/] - {DEPTH_LABELS[depth]}")
    return depth


###########################################################################
##                         CHAT LOOP
###########################################################################


def chat_loop(graph, initial_state: dict, thread_id: str) -> None:
    """Main chat loop -- read user input, invoke graph, display results."""
    config = {"configurable": {"thread_id": thread_id}}

    console.print(Panel(
        "[bold]Fashion Retail Data Assistant[/]\n"
        "Ask questions about sales, products, stores, customers, and more.\n"
        f"Depth: [cyan]{initial_state['depth']}[/] (max {initial_state['max_iterations']} tool rounds)\n"
        "Type [bold cyan]quit[/] or [bold cyan]exit[/] to leave.",
        border_style="cyan",
    ))

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        # Reset iteration state for each new question
        invoke_state = {
            **initial_state,
            "messages": [{"role": "user", "content": user_input}],
            "iteration": 0,
            "collected_results": [],
            "todo": [],
            "reflection": "",
            "rag_sources": [],
        }

        console.print("[dim]  Thinking...[/]")

        # Stream node-by-node updates for live visibility
        final_content = ""
        iterations_used = 0
        try:
            for chunk in graph.stream(invoke_state, config, stream_mode="updates"):
                for node_name, update in chunk.items():
                    display_node_update(node_name, update)

                    if node_name in ("synthesize", "respond"):
                        for msg in update.get("messages", []):
                            content = getattr(msg, "content", "")
                            if isinstance(content, list):
                                content = " ".join(b.get("text", "") for b in content if isinstance(b, dict)).strip()
                            if content:
                                final_content = content

                    if "iteration" in update:
                        iterations_used = update["iteration"]
        except Exception as e:
            console.print(f"\n[bold red]Error:[/] {e}")
            continue

        console.print(f"\n[dim]  (used {iterations_used}/{initial_state['max_iterations']} iterations)[/]")
        if final_content:
            console.print(f"\n[bold cyan]Assistant:[/]")
            console.print(render_ai_content(final_content))
        else:
            console.print(f"\n[bold red]Error:[/] The AI model did not return a response. This may be due to an API issue or content filter.")


###########################################################################
##                              MAIN
###########################################################################

def main():
    console.print("\n[bold cyan]Fashion Retail[/] [dim]Data Assistant[/]")
    console.print("[dim]Powered by gemini-2.5-flash + LangGraph StateGraph[/]\n")

    thread_id = select_thread()
    depth = select_depth()

    APP_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(APP_DB), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = workflow.compile(checkpointer=checkpointer)

    initial_state = {
        "depth": depth,
        "max_iterations": depth * 2,
    }

    try:
        chat_loop(graph, initial_state, thread_id)
    finally:
        conn.close()

    console.print("\n[dim]Session saved. Goodbye![/]\n")


if __name__ == "__main__":
    main()
