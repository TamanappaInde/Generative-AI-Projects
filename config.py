"""
Configuration management for customer support agent.

"""

import os
from dotenv import load_dotenv

load_dotenv

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def validate_config():
    """Validate that required configuration is present."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set."
            "Please create a .env file with your OPENAI_API_KEY or set it as an environment variable."
        )
    return True

