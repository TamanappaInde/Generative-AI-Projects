"""
Streamlit Web Application for data analysis agent.

Interactive Web Interface for the simple data analysis agent.

Author: Tamanappa Inde
Built With: streamlit + pydanticAI

"""

from __future__ import annotations
import os
from io import StringIO
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables before imoorting agents
load_dotenv
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

# Now import agent components
