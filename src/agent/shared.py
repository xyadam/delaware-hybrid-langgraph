###########################################################################
##                            IMPORTS
###########################################################################

from pathlib import Path

from langchain.chat_models import init_chat_model

###########################################################################
##                        CUSTOM IMPORTS
###########################################################################

from tools_sql import query_sql
from tools_rag import query_rag
from agent.prompts import build_system_prompt


###########################################################################
##                           CONSTANTS
###########################################################################

DB_DIR = Path(__file__).resolve().parent.parent.parent / "db"
APP_DB = DB_DIR / "application.db"
TOOLS = [query_sql, query_rag]
SYSTEM_PROMPT = build_system_prompt()


###########################################################################
##                           LLM SETUP
###########################################################################


def _get_llm():
    return init_chat_model("google_genai:gemini-2.5-flash", temperature=0.6)
