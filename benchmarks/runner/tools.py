"""Shared tools for benchmark agents.

Real file I/O and code execution, scoped to a task's workspace directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


class TaskTools:
    """Tools scoped to a specific task workspace."""

    def __init__(self, workspace_dir: str) -> None:
        self.workspace = Path(workspace_dir).resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"Workspace not found: {workspace_dir}")

    def _safe_path(self, relative_path: str) -> Path:
        """Resolve a path and ensure it's within workspace."""
        target = (self.workspace / relative_path).resolve()
        if not str(target).startswith(str(self.workspace)):
            raise ValueError(f"Path escape detected: {relative_path}")
        return target

    def read_file(self, path: str) -> str:
        """Read a text file from workspace. Returns content or error string."""
        try:
            target = self._safe_path(path)
            if not target.is_file():
                return f"ERROR: File not found: {path}"
            return target.read_text(encoding="utf-8")
        except Exception as e:
            return f"ERROR: {e}"

    def list_directory(self, path: str = ".") -> str:
        """List directory contents. Returns a newline-separated list."""
        try:
            target = self._safe_path(path)
            if not target.is_dir():
                return f"ERROR: Not a directory: {path}"
            entries = []
            for item in sorted(target.iterdir()):
                suffix = "/" if item.is_dir() else ""
                entries.append(f"{item.name}{suffix}")
            return "\n".join(entries) if entries else "(empty)"
        except Exception as e:
            return f"ERROR: {e}"

    def execute_code(self, code: str, timeout: int = 10) -> str:
        """Execute Python code in workspace directory with timeout."""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False,
                dir=str(self.workspace), encoding="utf-8"
            ) as f:
                f.write(code)
                temp_path = f.name

            try:
                result = subprocess.run(
                    [sys.executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(self.workspace),
                )
                output = result.stdout
                if result.stderr:
                    output += f"\n[STDERR]\n{result.stderr}"
                if result.returncode != 0:
                    output += f"\n[exit code: {result.returncode}]"
                return output.strip() or "(no output)"
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        except subprocess.TimeoutExpired:
            return f"ERROR: Execution timed out after {timeout}s"
        except Exception as e:
            return f"ERROR: {e}"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to file in workspace."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List contents of a directory in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path (default: '.')."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code in the workspace and return output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute."
                    }
                },
                "required": ["code"]
            }
        }
    }
]


def execute_tool_call(tools: TaskTools, name: str, args: dict) -> str:
    """Route a tool call to the correct method."""
    if name == "read_file":
        return tools.read_file(args.get("path", ""))
    elif name == "list_directory":
        return tools.list_directory(args.get("path", "."))
    elif name == "execute_code":
        return tools.execute_code(args.get("code", ""))
    else:
        return f"ERROR: Unknown tool: {name}"