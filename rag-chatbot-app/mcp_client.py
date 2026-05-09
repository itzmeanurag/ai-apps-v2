"""
mcp_client.py
McpClient – connects to a Java MCP server for augmented generation.
McpAugmentedGenerator – wraps the RAG pipeline with MCP tool calls.
"""
from __future__ import annotations

import json
import os
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


# ── MCP Client ────────────────────────────────────────────────────────────────

class McpClient:
    """
    HTTP client for a Java MCP (Model Context Protocol) server.

    The MCP server exposes tools that can be called to augment generation
    with external data sources (databases, APIs, file systems, etc.).
    """

    def __init__(self, config: Optional[McpConfig] = None) -> None:
        self._cfg = config or McpConfig()
        self._session = requests.Session()
        if self._cfg.api_key:
            self._session.headers.update({"Authorization": f"Bearer {self._cfg.api_key}"})
        self._session.headers.update({"Content-Type": "application/json"})

    # ── Tool discovery ────────────────────────────────────────────────────────

    def list_tools(self) -> list[dict]:
        """Return available tools from the MCP server."""
        return self._get("/mcp/tools")

    def get_tool_schema(self, tool_name: str) -> dict:
        """Return the JSON schema for a specific tool."""
        return self._get(f"/mcp/tools/{tool_name}/schema")

    # ── Tool execution ────────────────────────────────────────────────────────

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Call an MCP tool and return the result.

        Args:
            tool_name: Name of the tool to invoke.
            arguments: Tool input arguments.

        Returns:
            Tool result as a dict.
        """
        payload = {
            "tool": tool_name,
            "arguments": arguments,
            "timestamp": time.time(),
        }
        return self._post("/mcp/execute", payload)

    def search_knowledge_base(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the MCP server's knowledge base."""
        result = self.call_tool("search_knowledge_base", {"query": query, "top_k": top_k})
        return result.get("results", [])

    def get_employee_info(self, employee_id: str) -> dict:
        """Retrieve employee information via MCP."""
        return self.call_tool("get_employee_info", {"employee_id": employee_id})

    def get_policy_document(self, policy_name: str) -> dict:
        """Retrieve a specific policy document via MCP."""
        return self.call_tool("get_policy_document", {"policy_name": policy_name})

    def health_check(self) -> bool:
        """Return True if the MCP server is reachable."""
        try:
            resp = self._get("/health")
            return resp.get("status") == "ok"
        except Exception:
            return False

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _get(self, path: str) -> dict:
        url = self._cfg.server_url.rstrip("/") + path
        for attempt in range(self._cfg.retry_attempts):
            try:
                resp = self._session.get(url, timeout=self._cfg.timeout_seconds)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt == self._cfg.retry_attempts - 1:
                    raise RuntimeError(f"MCP GET {path} failed after {self._cfg.retry_attempts} attempts: {exc}")
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


# ── MCP-augmented generator ───────────────────────────────────────────────────

class McpAugmentedGenerator:
    """
    Wraps the RAG chatbot with MCP tool augmentation.
    Before generating, it queries the MCP server for additional context.
    """

    def __init__(self, chatbot: Any, mcp_client: McpClient) -> None:
        self._chatbot = chatbot
        self._mcp = mcp_client

    def ask(self, question: str, session_id: str = "default", **kwargs) -> dict:
        """
        Ask a question with MCP-augmented context.
        1. Query MCP for additional context
        2. Inject into RAG pipeline
        3. Generate answer
        """
        mcp_context = ""
        try:
            mcp_results = self._mcp.search_knowledge_base(question, top_k=3)
            if mcp_results:
                mcp_context = "\n\n[External Knowledge Base]\n" + "\n".join(
                    f"- {r.get('content', '')}" for r in mcp_results
                )
        except Exception as exc:
            print(f"[mcp] Context augmentation failed (non-fatal): {exc}")

        return self._chatbot.ask(
            question,
            session_id=session_id,
            extra_context=mcp_context,
            **kwargs,
        )
