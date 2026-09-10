"""
Property Agent - Searches for Airbnb properties matching requirements.
"""

import json
import os
from typing import Dict, Any, List
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from ..prompts.agent_prompts import PROPERTY_AGENT_PROMPT
from ..tools.airbnb_tools import airbnb_search_tool

# load enviornment variables
load_dotenv()

class PropertyAgent:
    """
    Agent that searches for properties matching user requirements.
    """
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the property agent.

        Args:
           model_name: OpenAI model to use

        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # create agent with airbnb search tool
        self.tools = [airbnb_search_tool]
        self.agent = create_tool_calling_agent(
            self.llm,
            self.tools,
            PROPERTY_AGENT_PROMPT
        )
        self.executor = AgentExecutor(
            agent = self.agent,
            tools = self.tools,
            verbose=True,
            max_iterations= 2,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )

    def search(self, requirements: Dict[str, Any]) -> List[Dict[str,Any]]:
        """
        Search for properties matching the requirements.

        Args:
           requirements: Parse travel requirements.

        Returns:
           List of property dictionaries with details and URLs.
        """
        # Keep the codepath deterministic and robust by using direct MCP tool parsing.
        # The llm+tool loop can exceed context window with large Aribnb payloads.
        properties = self._direct_search(requirements)
        if properties:
            return properties

        # Fallback to agent path only if direct search unexpectedly fails.
        try:
            requirements_str = json.dumps(requirements, indent=2)
            result = self.executor.invoke({"requirements": requirements_str})
            return self.extract_properties(result, requirements)
        except Exception as e:
            print(f"Error searching properties: {e}")
            return []


    def direct_search(self, requirements: Dict[str, Any])-> List[Dict[str, Any]]:
        """
        Direct search bypassing agent (fallback for context issues).

        """
        try:
            from ..tools.airbnb_tools import airbnb_search_wrapper

            location = requirements.get("destination", "")
            checkin = requirements.get("checkin_date", "")
            checkout = requirements.get("checkout_date", "")
            guests = requirements.get("guests", {})
            budget = requirements.get("budget", {})

            result_str = airbnb_search_wrapper (
                location=location,
                checkin=checkin,
                checkout=checkout,
                adults=guests.get("adults",1),
                children=guests.get("children",0),
                infants=guests.get("infants",0),
                pets=guests.get("pets",0),
                min_price=budget.get("min"),
                max_price=budget.get("max")
        
            )
            # parse and extract - handle mcp response format
            search_results = None
            try:
                # Try parsing as json
                parsed = json.loads(result_str)
                # Handle MCP content array format
                if isinstance(parsed, dict):
                    if "content" in parsed:
                        content = parsed["content"]
                        if isinstance(content, list) and len(content) > 0:
                            text_content = content[0].get("text","")
                            if text_content:
                                try:
                                    search_results = json.loads(text_content)
                                except json.JSONDecodeError:
                                    # Extract json from text 
                                    strat_idx = text_content.find('{')
                                    end_idx = text_content.rfind('}')
                                    if strat_idx != -1 and end_idx > strat_idx:
                                        json_str = text_content[strat_idx:end_idx+1]
                                        search_results=json.loads(json_str)

                    elif "searchResults" in parsed:
                        search_results = parsed
                elif isinstance(parsed, dict) and "searchResults" in parsed:
                    search_results = parsed
            except json.JSONDecodeError:
                # Try to extract json from string
                if "searchReulsts" in result_str:
                    start_idx = result_str.find('{')
                    end_idx = result_str.find('}')
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = result_str[start_idx:end_idx]
                        try:
                            search_results = json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
            properties = []

            if search_results and "searchResults" in search_results:
                for prop in search_results["searchResults"][:10]:
                    formatted = self.__format_property(prop, requirements)
                    if formatted:
                        properties.append(formatted)
            return properties
        except Exception as e:
            print(f"Direct search failed: {e}")
            import traceback
            traceback.print_exc()
            return []

        
    def extract_from_failed_agent(self, requirements: Dict[str, Any])-> List[Dict[str, Any]]:
        """
        Extract properties from agents intermediate steps even if agent failed.
        """
        # This would require access to the executors 
        # For now fall back to direct search
        return self._direct_search(requirements)

    def extract_properties(
            self, 
            agent_result: Dict[str, Any],
            requirements: Dict[str, Any]
    )-> List[Dict[str, Any]]:
        """
        Extract and format property data from agent result.
        """
        properties = []

        try:
            # Try to extract JOSN from intermediate steps.
            intermediate_steps = agent_result.get("intermediate_steps", [])

            for action, observation in intermediate_steps:
                if action.tool == "airbnb_search":
                    # parse the observation (tool output)
                    try:
                        # Handle different response formats
                        search_resulsts = None
                        # Try parsing as direct JSON
                        if isinstance(observation, str):
                            try:
                                search_resulsts = json.loads(observation)
                            except json.JSONDecodeError:
                                # Try to extract JSON from text content
                                # some MCP tools wrap responses in content arrays
                                if '"content"' in observation or '"searchResults"' in observation:
                                    # Try to find JSON object in the String
                                    start_idx = observation.find('{')
                                    if start_idx != -1:
                                        # Find matching closing brace
                                        brace_count = 0
                                        end_idx = start_idx
                                        for i in range(start_idx, len(observation)):
                                            if observation[i] == '{':
                                                brace_count+=1
                                            elif observation[i] == '}':
                                                brace_count -= 1
                                                if brace_count == 0:
                                                    end_idx = i + 1
                                                    break
                                        if end_idx > start_idx:
                                            json_str = observation[start_idx:end_idx]
                                            search_resulsts = json.loads(json_str)

                        # Handle structured content format (MCP response format) 
                        if not search_resulsts:
                            # Check if observation is already a dict(from MCP)
                            if isinstance(observation, dict):
                                # Handle MCP content array format
                                if "content" in observation:
                                    content = observation["content"]
                                    if isinstance(content, list) and len(content) > 0:
                                        # Get text from first content item
                                        first_item = content[0]
                                        if isinstance(first_item, dict):
                                            text_content = first_item.get("text", "")
                                            if text_content:
                                                try:
                                                    search_resulsts = json.loads(text_content)
                                                except json.JSONDecodeError:
                                                    # Try to extract JSON from text (find first { to last})
                                                    start_idx = text_content.find('{')
                                                    if start_idx != -1:
                                                        # Find the last } to get the complete JSON
                                                        end_idx = text_content.rfind('}')
                                                        if end_idx > start_idx:
                                                            json_str = text_content[start_idx:end_idx+1]
                                                            try:
                                                                search_resulsts = json.loads(json_str)
                                                            except json.JSONDecodeError:
                                                                pass
                                # handle structuredcontent format
                                elif "structuredContent" in observation:
                                    result_text = observation.get("structuredContent", {}).get("result", "")
                                    if result_text:
                                        try:
                                            search_resulsts = json.loads(result_text)
                                        except json.JSONDecodeError:
                                            pass
                            # If observation is a String try to parse it
                            elif isinstance(observation, str):
                                


                    except Exception as e:

        except Exception as e:
            


