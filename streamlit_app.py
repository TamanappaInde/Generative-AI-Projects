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
from src.agent import Deps, build_agent
from src.data import make_car_sales_df

# Page Config
st.set_page_config(
    page_title="Simple Data Analysis Agent (PydanticAI).",
    layout="wide",
)

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_source" not in st.session_state:
    st.session_state.data_source = None
if "agent" not in st.session_state:
    st.session_state.agent = None


def get_agent():
    """Get or Create the agent instance."""
    if st.session_state.agent is None:
        # Check for API Key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        st.session_state.agent = build_agent()
    return st.session_state.agent()


def ask_agent(question: str, df: pd.DataFrame) -> str:
    """Ask the agent a question about dataframe."""
    agent = get_agent()
    if agent is None:
        return "Error: OpenAI API key is not set. Please set it in sidebar."
    deps = Deps(df=df)
    result = agent.run_sync(question, deps=deps)
    # Access the response data - try different methods
    try:
        # Try .data first (common in PydanticAI)
        return str(result.data)
    except AttributeError:
        try:
            # Try new messages approach
            return result.new_messages()[-1].content
        except (AttributeError, IndexError):
            # Fallback to string representation
            return str(result)

def main():
    st.title(" Simple Data Analysis Agent")
    st.markdown("**Built with PydanticAI** | Ask questions on your natural language!")

    # Sidebar for data upload
    with st.sidebar:
        # API key configuration
        st.header(" Configuration")
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API Key. You can set also via OPENAI_API_KEY environment variable.",
            key="api_key_input"
        )

        if api_key_input:
            os.getenv["OPENAI_API_KEY"] = api_key_input
            # Resey agent if API key changed
            if st.session_state.agent is not None:
                st.session_state.agent = None
            st.session_state.agent = build_agent
            st.success(" API Key Set !")
        elif not os.getenv("OPENAI_API_KEY"):
            st.warning(" Please enter OpenAI API key to use agent.")

        st.divider()
        st.header(" Data Source")

        data_option = st.radio(
            "Choose Data Source:",
            ["Use Sample Data", "Upload CSV"],
            key= "data_option"
        )

        if data_option == "Use Sample Data":
            if st.button("Generate Sample Data", type="primary"):
                with st.spinner("Generating sample car sales data...."):
                    st.session_state.df = make_car_sales_df()
                    st.session_state.data_source = "Sample Data"
                    st.session_state.messages = [] # clear chat history
                    st.success("Sample data loaded!")
                    st.rerun()
        else: # Upload CSV
            uploaded_file = st.file_uploader(
                "Upload a CSV File",
                type= ['csv'],
                help=" Upload your csv file for analyse"
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.session_state.data_source = uploaded_file.name
                    st.session_state.messages = [] # clear chat history
                    st.success(" Loaded {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        # Display data info
        if st.session_state.df is not None:
            st.divider()
            st.subheader("Data Set Info")
            st.write(f"**Rows:** {len(st.session_state.df):, }")
            st.write(f"**Columns:** {len(st.session_state.df.columns)}")
            st.write(f"**Source:** {st.session_state.data_source}")

    # Main Content Area
    if st.session_state.df is None:
        st.info("Please load data from sidebar to get started.")
        st.markdown("""
        ### Example Questions:
        - "What are the column names?"
        - "How many rows are in this data set?"
        - "What is the average price?"
        - "Which salesperson sold the most cars?"
        - "Show me the most common car color?"
        """)
    else:
        # Display dataframe preview
        with st.expander("View Dataset", expanded=False):
            st.dataframe(st.session_state.df, use_container_width=True)
            st.download_button(
                label= "Download CSV",
                data= st.session_state.df.to_csv(index=False),
                file_name="dataset.csv",
                mime="text/csv"
            )

        # Chat Interface
        st.subheader("Ask Questions")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat inout
        if prompt := st.chat_input("Ask question on your data...."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get agent response
            with st.chat_message("assistant"):
                with st.spinner("Analysing data..."):
                    try:
                        response = ask_agent(prompt, st.session_state.df)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error : {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

        # Clear chat button
        if st.session_state.messages:
            if st.button(" Clear chat history"):
                st.session_state.messages = []
                st.rerun()

if __name__ == "__main__":
    main()




