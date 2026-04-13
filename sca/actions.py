"""
sca/actions.py — Tool Execution

BaseTool soyut sınıfı ve yerleşik araçlar:
- FileReaderTool
- WebFetcherTool
- CodeExecutorTool
- UserAskerTool
- SearchTool

ActionExecutor, proposal tipine göre doğru tool'u seçer ve çalıştırır.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from sca.prediction import ActionProposal, ActionType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
DEFAULT_FILE_READ_TIMEOUT_SECONDS: float = 10.0
DEFAULT_WEB_TIMEOUT_SECONDS: float = 15.0
DEFAULT_CODE_TIMEOUT_SECONDS: float = 30.0
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
MAX_WEB_RESPONSE_BYTES: int = 1 * 1024 * 1024  # 1 MB
MAX_SEARCH_RESULTS: int = 5
WEB_USER_AGENT: str = (
    "Mozilla/5.0 (compatible; SCA-Agent/1.0; +https://github.com/sca-agent)"
)
PRIVATE_IP_RANGES: list[str] = [
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "127.0.0.0/8",
    "::1/128",
    "fc00::/7",
]


# ---------------------------------------------------------------------------
# BaseTool
# ---------------------------------------------------------------------------
class BaseTool(ABC):
    """Tüm araçların soyut temel sınıfı.

    Her araç, bir action_type ile eşleşen async execute() ve
    maliyet tahmini veren estimate_cost() metodunu implemente etmelidir.
    """

    name: str
    description: str
    schema: dict[str, Any]

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> str:
        """Eylemi gerçekleştirir ve sonucu string olarak döndürür.

        Args:
            params: Eyleme özgü parametreler.

        Returns:
            Eylemin sonucu (her zaman str, exception raise etmez).
        """

    @abstractmethod
    def estimate_cost(self, params: dict[str, Any]) -> float:
        """Eylemin tahmini maliyetini döndürür (saniye veya token).

        Args:
            params: Eyleme özgü parametreler.

        Returns:
            Tahmini maliyet (float).
        """


# ---------------------------------------------------------------------------
# FileReaderTool
# ---------------------------------------------------------------------------
class FileReaderTool(BaseTool):
    """Güvenli yerel dosya okuyucu.

    Sadece izin verilen dizinlerde okur, path traversal engellenmiş,
    binary dosyalar reddedilir.

    Args:
        allowed_directories: İzin verilen dizinler. Boşsa cwd kullanılır.
        max_file_size: Maksimum dosya boyutu (byte).

    Example:
        >>> tool = FileReaderTool(allowed_directories=["/tmp"])
        >>> result = asyncio.run(tool.execute({"path": "/tmp/test.txt"}))
    """

    name = "file_reader"
    description = "Yerel bir dosyayı okur ve içeriğini döndürür."
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Okunacak dosyanın tam yolu."}
        },
        "required": ["path"],
    }

    def __init__(
        self,
        allowed_directories: Optional[list[str | Path]] = None,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
    ) -> None:
        if allowed_directories:
            self._allowed = [Path(d).resolve() for d in allowed_directories]
        else:
            self._allowed = [Path.cwd()]
        self._max_size = max_file_size
        logger.debug("FileReaderTool başlatıldı: allowed=%s", self._allowed)

    def _is_allowed_path(self, target: Path) -> bool:
        """Hedef path izin verilen dizinlerden birinin altında mı?"""
        resolved = target.resolve()
        return any(
            str(resolved).startswith(str(allowed))
            for allowed in self._allowed
        )

    async def execute(self, params: dict[str, Any]) -> str:
        """Dosyayı okur.

        Args:
            params: {"path": "<dosya yolu>"}

        Returns:
            Dosya içeriği veya "Error: <mesaj>" formatında hata.
        """
        raw_path = params.get("path", "")
        if not raw_path:
            return "Error: 'path' parametresi gerekli."

        target = Path(raw_path)

        # Path traversal kontrolü
        try:
            resolved = target.resolve()
        except (OSError, ValueError) as exc:
            return f"Error: Geçersiz path: {exc}"

        if not self._is_allowed_path(resolved):
            logger.warning("İzinsiz dosya erişimi engellendi: %s", resolved)
            return f"Error: '{resolved}' path'ine erişim izni yok."

        if not resolved.exists():
            return f"Error: Dosya bulunamadı: {resolved}"

        if not resolved.is_file():
            return f"Error: '{resolved}' bir dosya değil."

        file_size = resolved.stat().st_size
        if file_size > self._max_size:
            return (
                f"Error: Dosya çok büyük ({file_size} byte > {self._max_size} byte limiti)."
            )

        # Binary dosya testi
        try:
            with resolved.open("rb") as fh:
                raw = fh.read(512)
            if b"\x00" in raw:
                return "Error: Binary dosya okunmuyor, sadece metin dosyaları desteklenir."
        except OSError as exc:
            return f"Error: Dosya okunamadı: {exc}"

        try:
            content = resolved.read_text(encoding="utf-8")
            logger.debug("Dosya okundu: %s (%d karakter)", resolved, len(content))
            return content
        except UnicodeDecodeError:
            return "Error: Dosya UTF-8 ile decode edilemiyor."
        except OSError as exc:
            return f"Error: Dosya okunamadı: {exc}"

    def estimate_cost(self, params: dict[str, Any]) -> float:
        """Dosya okuma için ~0.1 saniye maliyet tahmini."""
        return 0.1


# ---------------------------------------------------------------------------
# WebFetcherTool
# ---------------------------------------------------------------------------
class WebFetcherTool(BaseTool):
    """Güvenli web içerik çekici.

    SSRF korumalı, özel IP aralıklarını engeller, HTML'den plain text çıkarır.

    Args:
        timeout: İstek zaman aşımı (saniye).
        max_response_size: Maksimum yanıt boyutu (byte).

    Example:
        >>> tool = WebFetcherTool(timeout=10)
        >>> result = asyncio.run(tool.execute({"url": "https://example.com"}))
    """

    name = "web_fetcher"
    description = "Bir URL'den içerik çeker ve metin olarak döndürür."
    schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Çekilecek URL."}
        },
        "required": ["url"],
    }

    def __init__(
        self,
        timeout: float = DEFAULT_WEB_TIMEOUT_SECONDS,
        max_response_size: int = MAX_WEB_RESPONSE_BYTES,
    ) -> None:
        self._timeout = timeout
        self._max_size = max_response_size
        self._private_networks = [
            ipaddress.ip_network(r) for r in PRIVATE_IP_RANGES
        ]
        logger.debug("WebFetcherTool başlatıldı: timeout=%s", timeout)

    def _is_private_ip(self, hostname: str) -> bool:
        """Hostname özel/loopback bir IP'ye çözümleniyor mu?"""
        lower = hostname.lower()
        if lower in ("localhost", "127.0.0.1", "::1"):
            return True
        try:
            addr = ipaddress.ip_address(hostname)
            return any(addr in net for net in self._private_networks)
        except ValueError:
            return False

    async def execute(self, params: dict[str, Any]) -> str:
        """URL içeriğini çeker.

        Args:
            params: {"url": "<URL>"}

        Returns:
            Sayfanın metin içeriği veya "Error: <mesaj>" formatında hata.
        """
        url = params.get("url", "")
        if not url:
            return "Error: 'url' parametresi gerekli."

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"Error: Desteklenmeyen URL şeması: {parsed.scheme!r}. Sadece http/https."

        hostname = parsed.hostname or ""
        if self._is_private_ip(hostname):
            logger.warning("SSRF girişimi engellendi: %s", hostname)
            return f"Error: '{hostname}' özel/loopback IP'ye erişim güvenlik nedeniyle engellendi."

        try:
            import requests
        except ImportError:
            return "Error: 'requests' kütüphanesi kurulu değil."

        loop = asyncio.get_event_loop()
        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: requests.get(
                        url,
                        timeout=self._timeout,
                        headers={"User-Agent": WEB_USER_AGENT},
                        stream=True,
                    ),
                ),
                timeout=self._timeout + 5,
            )
        except asyncio.TimeoutError:
            return f"Error: İstek zaman aşımına uğradı ({self._timeout}s)."
        except Exception as exc:
            return f"Error: HTTP isteği başarısız: {exc}"

        try:
            content_bytes = b""
            for chunk in response.iter_content(chunk_size=8192):
                content_bytes += chunk
                if len(content_bytes) > self._max_size:
                    content_bytes = content_bytes[: self._max_size]
                    logger.warning("Web yanıtı boyut limitinde kesildi: %s", url)
                    break

            html_text = content_bytes.decode("utf-8", errors="replace")

            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html_text, "html.parser")
                # Script ve style etiketlerini kaldır
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                plain = soup.get_text(separator="\n", strip=True)
                # Boş satırları sıkıştır
                lines = [ln for ln in plain.splitlines() if ln.strip()]
                result = "\n".join(lines[:500])  # Max 500 satır
            except ImportError:
                result = html_text[:5000]

            logger.debug("Web içerik çekildi: %s (%d karakter)", url, len(result))
            return result

        except Exception as exc:
            return f"Error: Yanıt işlenemedi: {exc}"

    def estimate_cost(self, params: dict[str, Any]) -> float:
        """Web fetch için ~2 saniye maliyet tahmini."""
        return 2.0


# ---------------------------------------------------------------------------
# CodeExecutorTool
# ---------------------------------------------------------------------------
class CodeExecutorTool(BaseTool):
    """Izole edilmiş Python kodu çalıştırıcı.

    Geçici dizinde subprocess ile çalışır, timeout uygulanır.

    Args:
        timeout: Yürütme zaman aşımı (saniye).
        sandbox: Güvenli sandbox modu (False ise çalıştırmayı reddeder).

    Example:
        >>> tool = CodeExecutorTool(sandbox=True)
        >>> result = asyncio.run(tool.execute({"code": "print(2 + 2)"}))
        >>> # "4"
    """

    name = "code_executor"
    description = "Python kodunu izole edilmiş ortamda çalıştırır ve çıktısını döndürür."
    schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Çalıştırılacak Python kodu."}
        },
        "required": ["code"],
    }

    def __init__(
        self,
        timeout: float = DEFAULT_CODE_TIMEOUT_SECONDS,
        sandbox: bool = True,
    ) -> None:
        self._timeout = timeout
        self._sandbox = sandbox
        logger.debug("CodeExecutorTool başlatıldı: sandbox=%s timeout=%s", sandbox, timeout)

    async def execute(self, params: dict[str, Any]) -> str:
        """Python kodunu çalıştırır.

        Args:
            params: {"code": "<Python kodu>"}

        Returns:
            stdout+stderr çıktısı veya "Error: <mesaj>" formatında hata.
        """
        if not self._sandbox:
            return "Error: Sandbox modu devre dışı, kod yürütme izni yok."

        code = params.get("code", "")
        if not code or not code.strip():
            return "Error: 'code' parametresi gerekli."

        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._run_code(code)),
                timeout=self._timeout + 5,
            )
            return result
        except asyncio.TimeoutError:
            return f"Error: Kod yürütme zaman aşımına uğradı ({self._timeout}s)."
        except Exception as exc:
            return f"Error: Kod yürütülemedi: {exc}"

    def _run_code(self, code: str) -> str:
        """Kodu geçici dizinde subprocess ile çalıştırır (sync helper)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = Path(tmpdir) / "script.py"
            code_file.write_text(code, encoding="utf-8")

            cmd = [sys.executable, str(code_file)]
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"

            try:
                # Linux'ta resource limit uygula
                preexec = None
                if platform.system() == "Linux":
                    import resource

                    def limit_resources() -> None:
                        # 256 MB memory limit
                        resource.setrlimit(
                            resource.RLIMIT_AS,
                            (256 * 1024 * 1024, 256 * 1024 * 1024),
                        )
                        # 25s CPU time limit
                        resource.setrlimit(
                            resource.RLIMIT_CPU,
                            (25, 25),
                        )

                    preexec = limit_resources

                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=tmpdir,
                    env=env,
                    preexec_fn=preexec,
                )
                output = proc.stdout
                if proc.stderr:
                    output += ("\n" if output else "") + proc.stderr
                return output.strip() if output.strip() else "(boş çıktı)"

            except subprocess.TimeoutExpired:
                return f"Error: Kod {self._timeout}s içinde tamamlanamadı."
            except Exception as exc:
                return f"Error: Subprocess hatası: {exc}"

    def estimate_cost(self, params: dict[str, Any]) -> float:
        """Kod çalıştırma için ~5 saniye maliyet tahmini."""
        return 5.0


# ---------------------------------------------------------------------------
# UserAskerTool
# ---------------------------------------------------------------------------
class UserAskerTool(BaseTool):
    """Kullanıcıya soru soran araç.

    Callback pattern kullanır — test için farklı callback enjekte edilebilir.

    Args:
        callback: Soru soran callable. Default: input().

    Example:
        >>> tool = UserAskerTool(callback=lambda q: "Evet")
        >>> result = asyncio.run(tool.execute({"question": "Dosya var mı?"}))
        >>> # "Evet"
    """

    name = "user_asker"
    description = "Kullanıcıya bir soru sorar ve yanıtı döndürür."
    schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Kullanıcıya sorulacak soru."}
        },
        "required": ["question"],
    }

    def __init__(self, callback: Optional[Callable[[str], str]] = None) -> None:
        self._callback = callback or input
        logger.debug("UserAskerTool başlatıldı.")

    async def execute(self, params: dict[str, Any]) -> str:
        """Kullanıcıya soru sorar.

        Args:
            params: {"question": "<soru metni>"}

        Returns:
            Kullanıcının yanıtı veya "Error: <mesaj>" formatında hata.
        """
        question = params.get("question", "")
        if not question:
            return "Error: 'question' parametresi gerekli."

        loop = asyncio.get_event_loop()
        try:
            answer = await loop.run_in_executor(
                None, lambda: self._callback(f"\n[AOGL Soru] {question}\nCevap: ")
            )
            logger.debug("Kullanıcı yanıtı alındı: %d karakter.", len(answer))
            return answer.strip() if answer else "(boş yanıt)"
        except Exception as exc:
            return f"Error: Kullanıcı yanıtı alınamadı: {exc}"

    def estimate_cost(self, params: dict[str, Any]) -> float:
        """Kullanıcı etkileşimi için ~30 saniye maliyet tahmini."""
        return 30.0


# ---------------------------------------------------------------------------
# SearchTool
# ---------------------------------------------------------------------------
class SearchTool(BaseTool):
    """DuckDuckGo HTML tabanlı web arama aracı.

    API anahtarı gerektirmez, HTML scraping ile çalışır.

    Args:
        max_results: Döndürülecek maksimum sonuç sayısı.
        timeout: İstek zaman aşımı (saniye).

    Example:
        >>> tool = SearchTool()
        >>> result = asyncio.run(tool.execute({"query": "Python asyncio"}))
    """

    name = "search"
    description = "DuckDuckGo üzerinden web araması yapar ve ilk sonuçları döndürür."
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Arama sorgusu."}
        },
        "required": ["query"],
    }

    def __init__(
        self,
        max_results: int = MAX_SEARCH_RESULTS,
        timeout: float = DEFAULT_WEB_TIMEOUT_SECONDS,
    ) -> None:
        self._max_results = max_results
        self._timeout = timeout
        logger.debug("SearchTool başlatıldı.")

    async def execute(self, params: dict[str, Any]) -> str:
        """DuckDuckGo araması yapar.

        Args:
            params: {"query": "<arama sorgusu>"}

        Returns:
            Bulunan sonuçların metin özeti veya "Error: <mesaj>".
        """
        query = params.get("query", "")
        if not query:
            return "Error: 'query' parametresi gerekli."

        try:
            import requests
        except ImportError:
            return "Error: 'requests' kütüphanesi kurulu değil."

        url = f"https://html.duckduckgo.com/html/?q={query}"
        loop = asyncio.get_event_loop()

        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: requests.get(
                        url,
                        timeout=self._timeout,
                        headers={"User-Agent": WEB_USER_AGENT},
                    ),
                ),
                timeout=self._timeout + 5,
            )
        except asyncio.TimeoutError:
            return f"Error: Arama zaman aşımına uğradı ({self._timeout}s)."
        except Exception as exc:
            return f"Error: Arama isteği başarısız: {exc}"

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            for result in soup.select(".result__body")[: self._max_results]:
                title_tag = result.select_one(".result__title")
                snippet_tag = result.select_one(".result__snippet")
                title = title_tag.get_text(strip=True) if title_tag else "Başlık yok"
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else "Özet yok"
                results.append(f"• {title}\n  {snippet}")

            if not results:
                return f"Arama sonucu bulunamadı: '{query}'"

            return f"Arama: '{query}'\n\n" + "\n\n".join(results)

        except ImportError:
            return response.text[:2000]
        except Exception as exc:
            return f"Error: Arama sonuçları parse edilemedi: {exc}"

    def estimate_cost(self, params: dict[str, Any]) -> float:
        """Arama için ~3 saniye maliyet tahmini."""
        return 3.0


# ---------------------------------------------------------------------------
# ActionExecutor
# ---------------------------------------------------------------------------
class ActionExecutor:
    """ActionProposal'ları yürüten merkezi koordinatör.

    Tool'ları register eder ve proposal action_type'ına göre doğru
    tool'u seçerek çalıştırır.

    Example:
        >>> executor = create_default_executor(allowed_dirs=["/tmp"])
        >>> result, t, cost, err = asyncio.run(executor.execute(proposal))
    """

    def __init__(self) -> None:
        self._tools: dict[ActionType, BaseTool] = {}
        logger.debug("ActionExecutor başlatıldı.")

    def register_tool(self, action_type: ActionType, tool: BaseTool) -> None:
        """Bir tool'u belirli bir ActionType ile ilişkilendirir.

        Args:
            action_type: Tool'un karşılayacağı eylem tipi.
            tool: Kayıt edilecek BaseTool nesnesi.
        """
        self._tools[action_type] = tool
        logger.debug("Tool register edildi: %s → %s", action_type, tool.name)

    async def execute(
        self, proposal: ActionProposal
    ) -> tuple[str, float, float, Optional[str]]:
        """ActionProposal'ı yürütür.

        Args:
            proposal: Yürütülecek ActionProposal.

        Returns:
            (result_str, exec_time_seconds, cost_actual, error_or_None) tuple'ı.
            Hata durumunda bile exception raise etmez.
        """
        action_type = proposal.action_type

        if action_type == ActionType.NO_ACTION:
            return ("NO_ACTION: Bu tahmin otomatik doğrulanamaz.", 0.0, 0.0, None)

        tool = self._tools.get(action_type)
        if tool is None:
            msg = f"'{action_type}' için kayıtlı tool bulunamadı."
            logger.warning(msg)
            return ("", 0.0, 0.0, msg)

        cost_estimate = tool.estimate_cost(proposal.parameters)
        start_time = time.monotonic()

        try:
            result = await tool.execute(proposal.parameters)
            exec_time = time.monotonic() - start_time
            error = None
            if result.startswith("Error:"):
                error = result
            logger.debug(
                "Tool %s tamamlandı: %.2fs, %d karakter çıktı.",
                tool.name, exec_time, len(result),
            )
            return (result, exec_time, cost_estimate, error)

        except Exception as exc:
            exec_time = time.monotonic() - start_time
            error_msg = f"Tool {tool.name} beklenmedik hata: {exc}"
            logger.error(error_msg, exc_info=True)
            return ("", exec_time, cost_estimate, error_msg)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_default_executor(
    allowed_dirs: Optional[list[str | Path]] = None,
    sandbox: bool = True,
    user_callback: Optional[Callable[[str], str]] = None,
) -> ActionExecutor:
    """Yerleşik tool'larla yapılandırılmış bir ActionExecutor oluşturur.

    Args:
        allowed_dirs: FileReaderTool için izin verilen dizinler.
        sandbox: CodeExecutorTool sandbox modu.
        user_callback: UserAskerTool için callback (default: input()).

    Returns:
        Yapılandırılmış ActionExecutor nesnesi.

    Example:
        >>> executor = create_default_executor(allowed_dirs=["/tmp"], sandbox=True)
    """
    executor = ActionExecutor()

    executor.register_tool(
        ActionType.READ_FILE,
        FileReaderTool(allowed_directories=allowed_dirs),
    )
    executor.register_tool(
        ActionType.WEB_FETCH,
        WebFetcherTool(),
    )
    executor.register_tool(
        ActionType.EXECUTE_CODE,
        CodeExecutorTool(sandbox=sandbox),
    )
    executor.register_tool(
        ActionType.USER_QUESTION,
        UserAskerTool(callback=user_callback),
    )
    executor.register_tool(
        ActionType.SEARCH,
        SearchTool(),
    )

    logger.info("Varsayılan ActionExecutor oluşturuldu: sandbox=%s", sandbox)
    return executor
