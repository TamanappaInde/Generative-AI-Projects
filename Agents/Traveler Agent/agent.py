from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from openai import OpenAI

from IPython.display import Image, display
import graphviz
from typing import TypedDict

llm = ChatOpenAI(
    api_key='sk-4K9tT07E4hJgwidLd72kve06uDYUqPT27BlO5gN2rIcP4y88',
    base_url='https://tokenra.io/v1',
    model="gemini-3.8-flash"
)

class ChatState(TypedDict):
    message: str
    response: str


def agent_node(state: ChatState)-> ChatState:
    """Agent rephrases the user query before passing to tool."""
    prompt = ChatPromptTemplate.from_template(
        "The User Asked: {question}"
        "Rephrase this as just a city name for weather lookup."
    )
    chain = prompt | llm
    output = chain.invoke({"question": state["message"]}).content
    return {"message": output, "response": ""}

def weather_tool(state: ChatState) -> ChatState:
    """Simple fake weather tool with static responses."""
    city = state["message"].lower()
    weather_data = {
        "delhi": "Sunny, 32c",
        "mumbai": "rainy, 28c"
    }
    forecast = weather_data.get(city, "Weather data not available")
    return {"message": state["message"], "response": forecast}

graph = StateGraph(ChatState)
graph.add_node("agent", agent_node)
graph.add_node("weather_tool", weather_tool)

graph.set_entry_point("agent")
graph.add_edge("agent", "weather_tool")
graph.add_edge("weather_tool", END)

app = graph.compile()

app.invoke({"message": "What is the weather in delhi today?", "response": ""})

