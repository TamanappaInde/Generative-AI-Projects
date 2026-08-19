"""
Data Analysis Agent     - Core Agent definition

This module defines the PydanticAI agent for data analysis with pandas dataframes.

Built with PydanticAI framework

Author: Tamanappa Inde

"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
from pydantic_ai import Agent, RunContext

@dataclass
class Deps:
    """Agent dependencies (injected into tools/prompts)."""

    df: pd.DataFrame

SYSTEM_PROMPT = """ You are an AI assistant that helps extract information from a pandas dataframe.
If asked about columns, check the column name first.
Be concise.

You have access to a tool that can evaluate limited pandas expressions over a Dataframe names 'df'.
"""

async def df_query(ctx: RunContext[Deps], query: str) -> str:
    """Run a pandas expression against the injected dataframe.

    The expression is evaluated via 'pandas.eval' with 'df' available in scope.
    
    """
    print(f"Running Query: '{query}'")

    # Tiny Guardrails: keep the model inside "pandas-on-df" land.
    lowered = query.lower()
    blocked = ["__", "import", "exec", "open(","os.", "sys.", "subprocess", "socket", "pickle"]
    if any(b in lowered for b in blocked):
        raise ModelRetry("Unsafe query detected. Use pandas operations on 'df' only.") 