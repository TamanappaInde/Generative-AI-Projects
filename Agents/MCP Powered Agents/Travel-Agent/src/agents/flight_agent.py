"""
Flight Agent - Searches for flight schedules matching travel requirements

"""

import json
import os
from typing import Dict, Any, List, Optional
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from ..prompts.agent_prompts import FLIGHT_AGENT_PROMPT
""" from ..tools.flight_tools import {
    future_flights_schedule_tool,
    flights_with_airline_tool,
    flights_with_airline_wrapper,
} """


