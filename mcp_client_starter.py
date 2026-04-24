from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self, command: str, args: list[str], env: Optional[dict] = None):
        # Store the command and args needed to launch the MCP server subprocess
        # e.g. command="python", args=["my_server.py"]
        # or   command="npx",    args=["-y", "@some/mcp-server"]
        # env is optional — use it to pass environment variables to the server process
        self._command = command
        self._args = args
        self._env = env
        self._session: Optional[ClientSession] = None
        # AsyncExitStack manages the lifecycle of multiple async context managers
        # (stdio transport + client session) so they are cleaned up in the right order
        self._exit_stack: AsyncExitStack = AsyncExitStack()

    async def connect(self):
        # Build the parameters needed to launch the server as a subprocess
        server_params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
        )
        # Start the server subprocess and get the stdio read/write streams
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        _stdio, _write = stdio_transport

        # Open an MCP client session over the stdio streams
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(_stdio, _write)
        )
        # Perform the MCP handshake — exchanges capabilities between client and server
        await self._session.initialize()

    def session(self) -> ClientSession:
        # Returns the active session, or raises if connect() was never called
        if self._session is None:
            raise ConnectionError("Client session not initialized. Call connect() first.")
        return self._session

    async def list_tools(self) -> list[types.Tool]:
        # Asks the server for all tools it exposes and returns them as a list
        result = await self.session().list_tools()
        return result.tools

    async def call_tool(self, tool_name: str, tool_input: dict) -> types.CallToolResult | None:
        # Invokes a specific tool on the server with the given arguments
        # tool_input is a dict matching the tool's input schema
        return await self.session().call_tool(tool_name, tool_input)

    async def cleanup(self):
        # Closes the session and the stdio transport in reverse order
        # AsyncExitStack handles this automatically when aclose() is called
        await self._exit_stack.aclose()
        self._session = None

    async def __aenter__(self):
        # Allows using MCPClient as: async with MCPClient(...) as client:
        await self.connect()
        return self

    async def __aexit__(self, *_):
        # Called automatically at the end of the async with block — cleans up
        await self.cleanup()
