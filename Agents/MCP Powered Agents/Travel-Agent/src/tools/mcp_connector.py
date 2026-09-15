"""
MCP Connector - Bridge between Python and Cursors MCP tools
"""


import json
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path

class MCPConnector:
    """
    Connects to MCP servers configured in cursor.

    Since Cursors MCP tools run via npx/uvx commands, we will use subprocess
    to invoke them directly with the same configuration from mcp.json
    
    """

    def __init__(self, mcp_config_path: Optional[str] = None):
        """
        Initialize MCP Connector.

        Args:
           mcp_config_path: Path to Cursors mcp.json config file.
        """

        if mcp_config_path is None:
            # Default to cursors MCP config location
            mcp_config_path = str(Path.home() / ".cursor" / "mcp.json")

        self.mcp_config_path = mcp_config_path
        self.config = self._load_config()

    def load_config(self)-> Dict:
        """ Load MCP server configuration from Cursors mcp.json."""
        try:
            with open(self.mcp_config_path, 'r') as f:
                config = json.load(f)
                return config.get('mcpServers', {})
        except FileNotFoundError:
            print(f"Warning: MCP config not found at {self.mcp_config_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing MCP config: {e}")
            return {}

    def call_mcp_tool(self,
            server_name: str,
            tool_name: str,
            arguments: Dict[str, Any],
            timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Call an MCP tool by comminunicating with MCP server

        Args:
          server_name: Name of the MCP server (e.g , 'airbnb', 'Aviationstack MCP')
          tool_name: Name of the tool to Call
          arguments: Tool arguments as a dictionary 
          timeout: Timeout in seconds

        Returns:
          Tool response as a dictionary

        Raises:
           RunTimeError: If the tool call fails
        """

        # Map user-facing names to config names
        server_key = server_name.lower().replace('user-', '').replace(' ', '-')

        if server_key not in self.config:
            # Try extact match
            server_key = server_name

    