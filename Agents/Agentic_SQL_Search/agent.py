"""
LangGraph ReAct agent for Agentic SQL Search, with three tools:
    load_skills: loads SQL syntax rules and schema reference
    get_schema: insoects the live database schema
    execute_sql: runs a SELECT query and returns results
"""

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

DB_PATH = Path(__file__).parent / "data" / "ecommerse.db"
SKILLS_DIR = Path(__file__).parent / "skills"

# Mutable active path - updated at runtime when user uploads their own database
_active_db_path: Path = DB_PATH

def set_db_path(path) -> None:
    """Switch the database the agent tools query against."""
    global _active_db_path
    _active_db_path = Path(path)

llm = ChatGoogleGenerativeAI(
    model = "gemma-4-26b-a4b-it",
    google_api_key=os.getenv("GEMINI_API_KEY")
    temperature=0,
)

#______Tools_______________________________________________________________________

@tool
def load_skill(skill_name: str)-> str:
    """
    Load a skill document with syntax rules and query patterns.
    Always call this first with skill_name= 'sql' before writing any query.
    """

    path = SKILLS_DIR / f"{skill_name}.md"
    if not path.exists():
        available = [p.stem for p in SKILLS_DIR.glob("*.md")]
        return f"Skill '{skill_name}' not found. Available: {', '.join(available)}"
    return path.read_text()


@tool
def get_schema() -> str:
    """
    Return the full database schema - table names, column names, and steps.

    call this to understand what tables and columns are available before writing SQL.
    """

    conn = sqlite3.connect(_active_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    schema_parts = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()
        col_defs = "\n".join([f"   {c[1]:30s} {c[2]}" for c in cols])
        schema_parts.append(f"TABLE: {table}\n{col_defs}")

    conn.close()
    return "\n\n".join(schema_parts)

@tool
def execute_sql(query: str)-> str:
    """
    Execute a SQL SELECT query against the e-commerse SQLite database.
    Returns results as a formatted table, Only SELECT queries are allowed.
    """

    query = query.strip()
    if not query.upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed."

    try:
        conn = sqlite3.connect(_active_db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()

        if not rows:
            return "Query executed successfully but returned no results."

        col_widths = [
            max(len(col), max((len(str(row[i])) for row in rows), default=0))
            for i, col in enumerate(columns)
        ]

        header = " | ".join(col.ljust(w) for col, w in zip(columns, col_widths))
        seperator = "-+-".join("-"* w  in col_widths)
        data_rows = [
            " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(columns)))
            for row in rows[:100]
        ]
        result = "\n".join([header, seperator] + data_rows)
        if len(rows) > 100:
            result += f"\n... ({len(rows)} total rows, showing first 100)"
        return result

    except sqlite3.Error as e:
        return f"SQL Error: {e}"

# _____________________Agent_____________________________________

tools = [load_skill, get_schema, execute_sql]

SYSTEM_PROMPT = """
You are a data analyst agent with access to e-commerse sqlite database\
containing customers, products, orders, and order items.

Your job is to answer the users question by quering the database.

Always follow this exact workflow:
1. Call load_skill with skill_name='sql' to review SQL Syntax rules and query patterns 
2. Call get_schema to confirm available tables and column names
3. Write and run the SQL query with execute_sql
4. Return a clear, consice summary of the results

Never guess column names - always check the schema first.

Only Write SELECT queries. Be precise with JOINs and aggregations.
"""

graph = create_react_agent(llm, tools=tools, prompt=SYSTEM_PROMPT)


def run_agent(question: str)-> dict:
    """
    Run the agent on question and return:
       {
       "answer": str,
       "steps": [{"tool": str, "input": dict, "output": str}]
       } 
    """
    result = graph.invoke({"messages": [("human", question)]})
    messages = result["messages"]

    # ----Extract final text answer(last non-tool AI message)-----
    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            content = msg.content
            if isinstance(content, str):
                answer = content.strip()
            elif isinstance(content, list):
                parts = [
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                ]
                answer = " ".join(parts).strip()

            if answer:
                break
        # Extract tool call steps
        steps = []
        tool_ouputs[msg.tool_call_id] = (
            
        )

