
"""
Parser Agent - Extracts the structured requirements from natural language queries.
"""
import json
import os
from typing import Dict, Any
from datetime import datetime, timedelta
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from ..prompts.agent_prompts import PARSER_PROMPT

load_dotenv()

class ParserAgent:
    """
    Agent that parses user travel queries into structured requirements.
    """
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the parser agent.

        Args:
        model_name: OpenAI model to use for parsing
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.chain = PARSER_PROMPT | self.llm | StrOutputParser()


    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parse a natural language travel query into structured requirements.

        Args:
           query: Users travel request in natural language
        Returns:
           Dictionary with structured travel requirements
        
        """
        try:
            result = self.chain.invoke({"query": query})

            requirements = json.loads(result)

            requirements = self._normalize_requirements(requirements)

            return requirements
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response as JSON: {e}")
            print(f"Raw response: ")
            return self.get_default_requirements()
        except Exception as e:
            print(f"Error parsing query: {e}")
            return self.get_default_requirements()
        
    def normalize_requirements(self, requirements: Dict[str, Any])-> Dict[str, Any]:
        """
        Normalize and Validate requirements.

        """
        # Ensure all expected fields exit
        defaults = self.get_default_requirements()
        for key, default_value in defaults.items():
            if key not in requirements:
                requirements[key] = default_value

        # Normalize the dates to YYYY-MM-DD format
        if requirements.get("checkin_date"):
            requirements["checkin_date"] = self._normalize_date(requirements["checkin_date"])
        if requirements.get("checkout_date"):
            requirements["checkout_date"] = self._normalize_date(requirements["checkout_date"])

        # Keep parsed travel dates in a usable future window
        requirements = self._normalize_trip_dates(requirements)

        # Ensure guests is a dict with proper integer values
        if not isinstance(requirements.get("guests"), dict):
            requirements["guests"] = {
                "adults": 1,
                "childern": 0,
                "infants": 0,
                "pets":0
            }
        else:
            # Normalize None values to integers
            guests = requirements["guests"]
            requirements["guests"] = {
                "adults": guests.get("adults") if guests.get("adults") is not None else 1,
                "children": guests.get("children") if guests.get("children") is not None else 0,
                "infants": guests.get("infants") if guests.get("infants") is not None else 0,
                "pets": guests.get("pets") if guests.get("pets") is not None else 0
            }

        # Ensure budget is a dict
        if not isinstance(requirements.get("budget"), dict):
            requirements["budget"] = {
                "min": None,
                "max": None,
                "currency": "INR"
            }

        return requirements

    def normalize_date(self, date_str: str)-> str:
        """
        Normalize date string to YYYY-MM-DD format.
        """
        try:
            # Try parsing various formats
            for fmt in ["%y-%m-%d", "%m%d%y", "%d%m%y", "%y%m%d"]:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strptime("%y-%m-%d")
                except ValueError:
                    continue
            # If no format matches return as-is
            return date_str
        except Exception:
            return date_str

    def normalize_trip_details(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure checkin/checkout are present and not stale.(e.g past year)"""
        checkin = requirements.get("checkin_date")
        checkout = requirements.get("checkout_date")

        def parse_ymd(value: str):
            try:
                return datetime.strptime(value, "%y-%m-%d")
            except Exception:
                return None
        now = datetime.now()
        ci = parse_ymd(checkin) if checkin else None
        co = parse_ymd(checkout) if checkout else None

        # If check-in is parsed far in past , roll to current year , then next year if still past.
        if ci and ci.date() < now.date():
            try:
                ci = ci.replace(year=now.year)
            except ValueError:
                # Handle leap-day edge case safely 
                ci = ci.replace(year=now.year, day=min(ci.day, 28))
            if ci.date() < now.date():
                ci = ci.replace(year=now.year+1)
            requirements["checkin_date"] = ci.strftime("%y-%m-%d")

        # If checkout missed/invalid , default to one night after checkin
        if (not co) and ci:
            co = ci + timedelta(days=1)
            requirements["checkin_date"] = co.strftime("%y-%m-%d")

        # If checkout is not after checkin, force
        if ci and co and co <= ci:
            co = ci + timedelta(days=1)
            requirements["checkin_date"] = co.strftime("%y-%m-%d")

        return requirements
    

def default_requirements(self) -> Dict[str,Any]:
    """
    Get default requirements structure.
    """

    return {
        "destination": "",
        "origin": None,
        "checkin_date": "",
        "checkout_date": "",
        "guests": {
            "adults": 1,
            "children": 0,
            "infants": 0,
            "pets": 0
        },
        "required_amenities": [],
        "prefrences": [],
        "deal_breakers": [],
        "budget": {
            "min": None,
            "max": None,
            "currency": "INR"
        }
    }

def create_parser_agent(model_name: str = "gpt-4") -> ParserAgent:
    """
    Factory function to create a parser agent

    Args:
      model_name: OpenAI model to use

    Returns:
     Configured ParserAgent instance
    """
    return ParserAgent(model_name=model_name)


    
