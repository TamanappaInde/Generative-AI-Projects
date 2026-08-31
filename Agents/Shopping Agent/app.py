"""
Shopping Agent
Redefining the concept of online shopping experience with agentic AI.

This is a lightweight, runnable translation of the provided notebook code:
- LangGraph workflow orchestration
- Tavily web search + content loading
- Groq Llama 3.1 (via langchain-groq) for extraction/comparison
- YouTube Data API for review link
- Optional SMTP email sending (Gmail SMTP)
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import time
import smtplib 

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from googleapiclient.discovery import build
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing_extensions import TypedDict

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")

def require_env(name: str)-> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is not set. Add it to your enviornment or .env file")
    return value


def build_llm() -> ChatGroq:
    api_key = require_env("GROQ_API_KEY")
    return ChatGroq(
        model = os.getenv("GROQ_MODEL","llma-3.1-70b-versatile"),
        api_key=api_key,
        temperature=float(os.getenv("GROQ_TEMPERATURE", "0.5")),
    )

def build_tavily() -> TavilyClient:
    api_key = require_env("TAVILY_API_KEY")
    return TavilyClient(api_key=api_key)


def build_youtube_client():
    if not YOUTUBE_API_KEY:
        return None
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# -----------------------------
# Pydantic schemas (structured outputs)
# -----------------------------

class ProductHighlights(BaseModel):
    Camera: Optional[str] = None
    Performance: Optional[str] = None
    Display: Optional[str] = None
    Fast_Charging: Optional[str] = None

class ProductReview(BaseModel):
    title: str = Field(..., description="The prodyct name/title")
    url: Optional[str] = Field(None, description="Source URL")
    content: Optional[str] = Field(None, description="Consize summory of the product")
    pros: Optional[List[str]] = Field(None, description="Pros List")
    cons: Optional[List[str]] = Field(None, description="Cons List")
    highlights: Optional[dict] = Field(None, description="Notable specs/features")
    score: Optional[float] = Field(0.0, description="Numeric Score if available, else 0.0")

class ListofProductReviews(BaseModel):
    products: List[ProductReview] = Field(..., description="List of extracted products")

class SpecComparison(BaseModel):
    processor: str = Field(..., description="Processor Type and Model")
    battery: str = Field(..., description="Battery capacity and type")
    camera: str = Field(..., description="Camera specs")
    display: str = Field(..., description="Display specs")
    storage: str = Field(..., description="Storage specs")


class RatingComparison(BaseModel):
    overall_rating: float = Field(..., description="Overall rating out of 5")
    performance: float = Field(..., description="Performance rating out of 5")
    battery_life: float = Field(..., description="Battery life of rating out of 5")
    camera_quality: float = Field(..., description="Camera quality rating out of 5")
    display_quality: float = Field(..., description="Display quality rating out of 5")

class Comparison(BaseModel):
    product_name: str = Field(..., description="Name of the Product")
    specs_comparison: SpecComparison
    ratings_comparison: RatingComparison
    reviews_summory: str = Field(..., description="Summory of review points from user reviews")

class BestProduct(BaseModel):
    product_name: str = Field(..., description="Name of the best product")
    justification: str = Field(..., description="why this is the best product")

class ProductComparison(BaseModel):
    comparisons: List[Comparison]
    best_product: BestProduct

class EmailRecommendation(BaseModel):
    subject: str
    heading: str
    justification_line: str

# -----------------------------
# LangGraph state
# -----------------------------

class State(TypedDict, total=False):
    query: str
    email: str
    blogs_content: List[dict]
    product_schema: List[dict]
    comparison: List[dict]
    best_product: dict
    youtube_link: Optional[str]

# -----------------------------
# Email utilities
# -----------------------------

def send_email(recipient_mail: str, subject: str, body_html: str) -> None:
    """Send an email using Gmail SMTP (requires Gmail User and Pass)."""