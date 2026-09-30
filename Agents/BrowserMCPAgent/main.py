import asyncio
import os
import streamlit as st
from textwrap import dedent

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

#Page config
st.set_page_config(page_title="Browser MCP Agent", page_icon="🌐", layout="wide")

# Title and description
st.markdown("<h1 class='main-header'> 🌐 Browser MCP Agent</h1>", unsafe_allow_html=True)
st.markdown("Interact with a powerfull web browsing agent that can navigate and interact with websites")

# Setup sidebar with example commands
with st.sidebar:
    st.markdown("### Example Commands")

    st.markdown("**Navigation**")
    #st.markdown