"""
tests/test_actions.py — Tool ve ActionExecutor testleri

Tüm I/O mock'lanmış, gerçek network/dosya çağrısı yok.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sca.actions import (
    ActionExecutor,
    CodeExecutorTool,
    FileReaderTool,
    SearchTool,
    UserAskerTool,
    WebFetcherTool,
    create_default_executor,
)
from sca.prediction import ActionProposal, ActionType


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# FileReaderTool testleri
# ---------------------------------------------------------------------------

class TestFileReaderTool:
    def test_read_allowed_file(self, tmp_path):
        """İzin verilen dizindeki dosyayı okuyabilmeli."""
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        tool = FileReaderTool(allowed_directories=[str(tmp_path)])
        result = run(tool.execute({"path": str(f)}))
        assert result == "hello world"

    def test_path_traversal_blocked(self, tmp_path):
        """../.. path traversal engellenmiş olmalı."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        # allowed sadece subdir ama üst dizine çıkmaya çalış
        tool = FileReaderTool(allowed_directories=[str(subdir)])
        result = run(tool.execute({"path": str(tmp_path / "secret.txt")}))
        assert result.startswith("Error:")

    def test_file_not_found(self, tmp_path):
        """Olmayan dosya için hata döndürmeli."""
        tool = FileReaderTool(allowed_directories=[str(tmp_path)])
        result = run(tool.execute({"path": str(tmp_path / "nonexistent.txt")}))
        assert result.startswith("Error:")

    def test_binary_file_rejected(self, tmp_path):
        """Binary dosya reddedilmeli."""
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00\x01\x02\x03binary content")
        tool = FileReaderTool(allowed_directories=[str(tmp_path)])
        result = run(tool.execute({"path": str(f)}))
        assert result.startswith("Error:")

    def test_file_too_large(self, tmp_path):
        """Çok büyük dosya reddedilmeli."""
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * (11 * 1024 * 1024))  # 11 MB > limit 10 MB
        tool = FileReaderTool(allowed_directories=[str(tmp_path)], max_file_size=10 * 1024 * 1024)
        result = run(tool.execute({"path": str(f)}))
        assert result.startswith("Error:")

    def test_missing_path_param(self, tmp_path):
        """path parametresi yoksa hata döndürmeli."""
        tool = FileReaderTool(allowed_directories=[str(tmp_path)])
        result = run(tool.execute({}))
        assert result.startswith("Error:")

    def test_estimate_cost(self, tmp_path):
        """estimate_cost float döndürmeli."""
        tool = FileReaderTool()
        cost = tool.estimate_cost({"path": "/tmp/test.txt"})
        assert isinstance(cost, float)
        assert cost >= 0


# ---------------------------------------------------------------------------
# WebFetcherTool testleri
# ---------------------------------------------------------------------------

class TestWebFetcherTool:
    def test_ssrf_localhost_blocked(self):
        """localhost erişimi SSRF koruması ile engellenmeli."""
        tool = WebFetcherTool()
        result = run(tool.execute({"url": "http://localhost:8080/secret"}))
        assert result.startswith("Error:")
        assert "engellendi" in result.lower() or "ssrf" in result.lower() or "güvenlik" in result.lower()

    def test_ssrf_private_ip_blocked(self):
        """192.168.x.x IP'si engellenmeli."""
        tool = WebFetcherTool()
        result = run(tool.execute({"url": "http://192.168.1.1/admin"}))
        assert result.startswith("Error:")

    def test_ssrf_10x_blocked(self):
        """10.x.x.x IP'si engellenmeli."""
        tool = WebFetcherTool()
        result = run(tool.execute({"url": "http://10.0.0.1/"}))
        assert result.startswith("Error:")

    def test_invalid_scheme(self):
        """ftp:// gibi desteklenmeyen şema hata döndürmeli."""
        tool = WebFetcherTool()
        result = run(tool.execute({"url": "ftp://example.com/file"}))
        assert result.startswith("Error:")

    def test_missing_url_param(self):
        """url parametresi yoksa hata döndürmeli."""
        tool = WebFetcherTool()
        result = run(tool.execute({}))
        assert result.startswith("Error:")

    @patch("sca.actions.WebFetcherTool")
    def test_mock_successful_fetch(self, mock_cls, tmp_path):
        """Mock ile başarılı fetch simülasyonu."""
        tool = WebFetcherTool()
        # Direkt execute mock'la
        async def mock_execute(params):
            return "Mock web content"
        tool.execute = mock_execute
        result = run(tool.execute({"url": "https://example.com"}))
        assert result == "Mock web content"

    def test_estimate_cost(self):
        """estimate_cost float döndürmeli."""
        tool = WebFetcherTool()
        cost = tool.estimate_cost({"url": "https://example.com"})
        assert isinstance(cost, float)
        assert cost > 0


# ---------------------------------------------------------------------------
# CodeExecutorTool testleri
# ---------------------------------------------------------------------------

class TestCodeExecutorTool:
    def test_simple_code_execution(self):
        """Basit Python kodu çalıştırabilmeli."""
        tool = CodeExecutorTool(sandbox=True)
        result = run(tool.execute({"code": "print(2 + 2)"}))
        assert "4" in result

    def test_sandbox_disabled_blocks_execution(self):
        """sandbox=False iken kod çalıştırma engellenmeli."""
        tool = CodeExecutorTool(sandbox=False)
        result = run(tool.execute({"code": "print('test')"}))
        assert result.startswith("Error:")

    def test_syntax_error_handled(self):
        """Sözdizimi hataları gracefully döndürülmeli (raise değil)."""
        tool = CodeExecutorTool(sandbox=True)
        result = run(tool.execute({"code": "def f(\n  invalid syntax here!!!"}))
        # Hata mesajı döner ama exception raise etmez
        assert isinstance(result, str)
        assert len(result) > 0

    def test_timeout_handling(self):
        """Zaman aşımı hata mesajı döndürmeli."""
        tool = CodeExecutorTool(sandbox=True, timeout=1.0)
        result = run(tool.execute({"code": "import time; time.sleep(10)"}))
        assert "Error:" in result or "timeout" in result.lower() or "zaman" in result.lower()

    def test_empty_code(self):
        """Boş kod hata döndürmeli."""
        tool = CodeExecutorTool(sandbox=True)
        result = run(tool.execute({"code": ""}))
        assert result.startswith("Error:")

    def test_missing_code_param(self):
        """code parametresi yoksa hata döndürmeli."""
        tool = CodeExecutorTool(sandbox=True)
        result = run(tool.execute({}))
        assert result.startswith("Error:")

    def test_estimate_cost(self):
        """estimate_cost float döndürmeli."""
        tool = CodeExecutorTool()
        cost = tool.estimate_cost({"code": "print('hello')"})
        assert isinstance(cost, float)
        assert cost > 0


# ---------------------------------------------------------------------------
# UserAskerTool testleri
# ---------------------------------------------------------------------------

class TestUserAskerTool:
    def test_callback_injection(self):
        """Callback injection ile kullanıcı yanıtı simüle edilmeli."""
        tool = UserAskerTool(callback=lambda prompt: "Evet, doğru")
        result = run(tool.execute({"question": "Dosya var mı?"}))
        assert result == "Evet, doğru"

    def test_default_lambda_callback(self):
        """Lambda callback çalışmalı."""
        tool = UserAskerTool(callback=lambda q: "mock_answer")
        result = run(tool.execute({"question": "Herhangi bir soru?"}))
        assert result == "mock_answer"

    def test_missing_question_param(self):
        """question parametresi yoksa hata döndürmeli."""
        tool = UserAskerTool(callback=lambda q: "test")
        result = run(tool.execute({}))
        assert result.startswith("Error:")

    def test_empty_answer_handled(self):
        """Boş kullanıcı yanıtı gracefully işlenmeli."""
        tool = UserAskerTool(callback=lambda q: "")
        result = run(tool.execute({"question": "Soru?"}))
        assert isinstance(result, str)

    def test_estimate_cost(self):
        """estimate_cost float döndürmeli."""
        tool = UserAskerTool()
        cost = tool.estimate_cost({"question": "?"})
        assert isinstance(cost, float)


# ---------------------------------------------------------------------------
# ActionExecutor testleri
# ---------------------------------------------------------------------------

class TestActionExecutor:
    def _make_executor_with_mock(self):
        """Mock tool'larla executor oluşturur."""
        executor = ActionExecutor()

        mock_tool = MagicMock()
        mock_tool.estimate_cost.return_value = 1.0

        async def mock_execute(params):
            return "mock result"

        mock_tool.execute = mock_execute
        executor.register_tool(ActionType.READ_FILE, mock_tool)
        return executor, mock_tool

    def test_no_action_returns_immediately(self):
        """NO_ACTION proposal yürütme atlar."""
        executor = ActionExecutor()
        proposal = ActionProposal(
            action_type=ActionType.NO_ACTION,
            parameters={},
        )
        result, t, cost, error = run(executor.execute(proposal))
        assert "NO_ACTION" in result
        assert error is None

    def test_unknown_action_type_returns_error(self):
        """Kayıtlı olmayan tool için hata string döndürmeli."""
        executor = ActionExecutor()
        proposal = ActionProposal(
            action_type=ActionType.READ_FILE,
            parameters={"path": "/tmp/test.txt"},
        )
        result, t, cost, error = run(executor.execute(proposal))
        assert error is not None
        assert "bulunamadı" in error or "tool" in error.lower()

    def test_registered_tool_called(self):
        """Register edilmiş tool çağrılmalı."""
        executor, mock_tool = self._make_executor_with_mock()
        proposal = ActionProposal(
            action_type=ActionType.READ_FILE,
            parameters={"path": "/tmp/test.txt"},
        )
        result, exec_time, cost, error = run(executor.execute(proposal))
        assert result == "mock result"
        assert error is None
        assert exec_time >= 0

    def test_tool_error_string_not_exception(self):
        """Tool exception fırlatırsa string olarak yakalanmalı."""
        executor = ActionExecutor()

        class BrokenTool(FileReaderTool):
            async def execute(self, params):
                raise RuntimeError("Beklenmedik hata")

        executor.register_tool(ActionType.READ_FILE, BrokenTool())
        proposal = ActionProposal(
            action_type=ActionType.READ_FILE,
            parameters={"path": "/tmp/test.txt"},
        )
        result, t, cost, error = run(executor.execute(proposal))
        assert error is not None
        assert isinstance(error, str)

    def test_create_default_executor(self, tmp_path):
        """create_default_executor tüm tool'ları register etmeli."""
        executor = create_default_executor(
            allowed_dirs=[str(tmp_path)], sandbox=True
        )
        assert ActionType.READ_FILE in executor._tools
        assert ActionType.WEB_FETCH in executor._tools
        assert ActionType.EXECUTE_CODE in executor._tools
        assert ActionType.USER_QUESTION in executor._tools
        assert ActionType.SEARCH in executor._tools
