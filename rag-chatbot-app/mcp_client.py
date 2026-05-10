"""
mcp_client.py

McpClient – connects to a Java MCP server at /mcp/tools.

Spec-required McpClient methods:
  health() -> bool
  list_tools() -> list[dict]
  execute_tool(tool_name, arguments) -> dict
  get_tool_descriptions_for_prompt() -> str

Spec-required McpAugmentedGenerator methods:
  initialize() -> bool
  ask_with_tools(question, session_id) -> dict
  _try_tool_only(question) -> dict | None
  _parse_tool_call(text) -> dict | None
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class McpConfig:
    server_url: str = "http://localhost:8080"
    api_key: str = field(default_factory=lambda: os.getenv("MCP_API_KEY", ""))
    timeout_seconds: int = 10
    retry_attempts: int = 3
    enabled: bool = True


# ── McpClient ─────────────────────────────────────────────────────────────────

class McpClient:
    """
    HTTP client for a Java MCP (Model Context Protocol) server.

    The server exposes tools at /mcp/tools that augment generation
    with external data sources (databases, APIs, file systems, etc.).
    """

    def __init__(self, config: Optional[McpConfig] = None) -> None:
        self._cfg = config or McpConfig()
        self._session = requests.Session()
        if self._cfg.api_key:
            self._session.headers["Authorization"] = f"Bearer {self._cfg.api_key}"
        self._session.headers["Content-Type"] = "application/json"
        self._tools_cache: Optional[list[dict]] = None

    # ── Spec-required methods ─────────────────────────────────────────────────

    def health(self) -> bool:
        """Return True if the MCP server is reachable and healthy."""
        try:
            resp = self._get("/health")
            return resp.get("status") in ("ok", "healthy", "UP")
        except Exception:
            return False

    def list_tools(self) -> list[dict]:
        """
        Return available tools from the MCP server.
        Each tool dict has: name, description, parameters (JSON schema).
        """
        if self._tools_cache is not None:
            return self._tools_cache
        result = self._get("/mcp/tools")
        # Server may return {"tools": [...]} or a bare list
        if isinstance(result, list):
            self._tools_cache = result
        else:
            self._tools_cache = result.get("tools", [])
        return self._tools_cache

    def execute_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Execute a named MCP tool with the given arguments.

        Returns the tool result dict.
        Raises RuntimeError on failure.
        """
        payload = {
            "tool": tool_name,
            "arguments": arguments,
            "timestamp": time.time(),
        }
        return self._post("/mcp/execute", payload)

    def get_tool_descriptions_for_prompt(self) -> str:
        """
        Return a formatted string describing all available tools,
        suitable for injection into an LLM prompt.

        Format:
            AVAILABLE TOOLS:
            - search_knowledge_base(query, top_k): Search the knowledge base...
            - get_employee_info(employee_id): Retrieve employee information...
        """
        tools = self.list_tools()
        if not tools:
            return "No MCP tools available."

        lines = ["AVAILABLE TOOLS:"]
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "No description.")
            params = tool.get("parameters", {})
            param_names = list(params.get("properties", {}).keys()) if params else []
            param_str = ", ".join(param_names) if param_names else ""
            lines.append(f"- {name}({param_str}): {desc}")

        return "\n".join(lines)

    # ── Convenience wrappers ──────────────────────────────────────────────────

    def search_knowledge_base(self, query: str, top_k: int = 5) -> list[dict]:
        result = self.execute_tool("search_knowledge_base", {"query": query, "top_k": top_k})
        return result.get("results", [])

    def get_employee_info(self, employee_id: str) -> dict:
        return self.execute_tool("get_employee_info", {"employee_id": employee_id})

    def get_policy_document(self, policy_name: str) -> dict:
        return self.execute_tool("get_policy_document", {"policy_name": policy_name})

    # Backward-compat alias
    def health_check(self) -> bool:
        return self.health()

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        return self.execute_tool(tool_name, arguments)

    def get_tool_schema(self, tool_name: str) -> dict:
        return self._get(f"/mcp/tools/{tool_name}/schema")

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _get(self, path: str) -> Any:
        url = self._cfg.server_url.rstrip("/") + path
        for attempt in range(self._cfg.retry_attempts):
            try:
                resp = self._session.get(url, timeout=self._cfg.timeout_seconds)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt == self._cfg.retry_attempts - 1:
                    raise RuntimeError(
                        f"MCP GET {path} failed after {self._cfg.retry_attempts} attempts: {exc}"
                    )
                time.sleep(0.5 * (attempt + 1))
        return {}

    def _post(self, path: str, payload: dict) -> dict:
        url = self._cfg.server_url.rstrip("/") + path
        for attempt in range(self._cfg.retry_attempts):
            try:
                resp = self._session.post(
                    url,
                    data=json.dumps(payload),
                    timeout=self._cfg.timeout_seconds,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt == self._cfg.retry_attempts - 1:
                    raise RuntimeError(f"MCP POST {path} failed: {exc}")
                time.sleep(0.5 * (attempt + 1))
        return {}


# ── McpAugmentedGenerator ─────────────────────────────────────────────────────

class McpAugmentedGenerator:
    """
    Wraps the RAG chatbot with MCP tool augmentation.

    Spec-required methods:
      initialize() -> bool
      ask_with_tools(question, session_id) -> dict
      _try_tool_only(question) -> dict | None
      _parse_tool_call(text) -> dict | None
    """

    # Regex to detect tool call syntax in LLM output:
    # TOOL_CALL: tool_name({"arg": "value"})
    _TOOL_CALL_RE = re.compile(
        r"TOOL_CALL:\s*(\w+)\s*\((\{.*?\})\)",
        re.DOTALL,
    )

    def __init__(self, chatbot: Any, mcp_client: McpClient) -> None:
        self._chatbot = chatbot
        self._mcp = mcp_client
        self._initialized = False
        self._tool_descriptions = ""

    # ── Spec-required methods ─────────────────────────────────────────────────

    def initialize(self) -> bool:
        """
        Connect to the MCP server and cache tool descriptions.
        Returns True if successful, False otherwise.
        """
        try:
            if not self._mcp.health():
                print("[mcp] Server not reachable. MCP augmentation disabled.")
                self._initialized = False
                return False
            self._tool_descriptions = self._mcp.get_tool_descriptions_for_prompt()
            self._initialized = True
            tools = self._mcp.list_tools()
            print(f"[mcp] Initialized with {len(tools)} tool(s).")
            return True
        except Exception as exc:
            print(f"[mcp] Initialization failed: {exc}")
            self._initialized = False
            return False

    def ask_with_tools(
        self,
        question: str,
        session_id: str = "default",
        persona: str = "default",
    ) -> dict:
        """
        Ask a question with MCP tool augmentation.

        Strategy:
        1. Try to answer using tools only (_try_tool_only)
        2. If tools don't have the answer, fall back to RAG with tool context
        """
        if not self._initialized:
            return self._chatbot.ask(question, session_id=session_id, persona=persona)

        # 1. Try tool-only answer
        tool_result = self._try_tool_only(question)
        if tool_result and tool_result.get("answer"):
            return {
                **tool_result,
                "session_id": session_id,
                "source": "mcp_tool",
            }

        # 2. Augment RAG with MCP context
        mcp_context = ""
        try:
            results = self._mcp.search_knowledge_base(question, top_k=3)
            if results:
                mcp_context = "\n\n[External Knowledge Base]\n" + "\n".join(
                    f"- {r.get('content', r.get('text', ''))}" for r in results
                )
        except Exception as exc:
            print(f"[mcp] Context augmentation failed (non-fatal): {exc}")

        return self._chatbot.ask(
            question,
            session_id=session_id,
            persona=persona,
        )

    def _try_tool_only(self, question: str) -> Optional[dict]:
        """
        Ask the LLM whether a tool can answer the question directly.
        Returns a result dict if a tool was called, None otherwise.
        """
        if not self._tool_descriptions:
            return None

        prompt = (
            f"{self._tool_descriptions}\n\n"
            "If one of the above tools can directly answer the question, "
            "respond with:\n"
            "TOOL_CALL: tool_name({\"arg\": \"value\"})\n\n"
            "Otherwise respond with: NO_TOOL\n\n"
            f"Question: {question}"
        )

        try:
            response = self._chatbot._llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)

            if "NO_TOOL" in text:
                return None

            parsed = self._parse_tool_call(text)
            if parsed is None:
                return None

            tool_result = self._mcp.execute_tool(parsed["tool"], parsed["arguments"])
            answer = tool_result.get("answer") or tool_result.get("content") or str(tool_result)
            return {"answer": answer, "tool_used": parsed["tool"], "tool_result": tool_result}

        except Exception as exc:
            print(f"[mcp] Tool-only attempt failed: {exc}")
            return None

    def _parse_tool_call(self, text: str) -> Optional[dict]:
        """
        Parse a TOOL_CALL directive from LLM output.

        Expected format: TOOL_CALL: tool_name({"arg": "value"})

        Returns:
            {"tool": str, "arguments": dict} or None if not found.
        """
        match = self._TOOL_CALL_RE.search(text)
        if not match:
            return None
        tool_name = match.group(1)
        try:
            arguments = json.loads(match.group(2))
        except json.JSONDecodeError:
            arguments = {}
        return {"tool": tool_name, "arguments": arguments}

    # Backward-compat alias
    def ask(self, question: str, session_id: str = "default", **kwargs) -> dict:
        return self.ask_with_tools(question, session_id=session_id)
