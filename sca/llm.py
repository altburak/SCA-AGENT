"""
PSM — LLM İstemcisi
====================
Groq üzerinde Llama 3.3 70B için LiteLLM sarmalayıcısı.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = "groq/llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_MAX_TOKENS: int = 1024
ENV_GROQ_KEY: str = "GROQ_API_KEY"


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------
class LLMClient:
    """LiteLLM üzerinden Groq/Llama çağrıları yapan istemci.

    Args:
        model: LiteLLM model adı (default: groq/llama-3.3-70b-versatile).
        temperature: Örnekleme sıcaklığı (default: 0.2).
        max_tokens: Maksimum üretilecek token (default: 1024).
        api_key: Groq API anahtarı. Verilmezse GROQ_API_KEY env'den okunur.

    Raises:
        EnvironmentError: API anahtarı bulunamazsa.
        ImportError: litellm kurulu değilse.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            import litellm  # noqa: F401 — varlığını kontrol et
        except ImportError as exc:
            raise ImportError(
                "litellm kurulu değil. `pip install litellm` komutunu çalıştırın."
            ) from exc

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get(ENV_GROQ_KEY)
        if not self._api_key:
            raise EnvironmentError(
                f"Groq API anahtarı bulunamadı. "
                f"{ENV_GROQ_KEY} ortam değişkenini tanımlayın veya "
                f"api_key parametresi olarak geçirin."
            )
        logger.info("LLMClient başlatıldı: model=%s", self.model)

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """LiteLLM üzerinden sohbet tamamlama isteği gönderir.

        Args:
            messages: [{"role": "...", "content": "..."}] formatında liste.
            **kwargs: litellm.completion'a iletilecek ek parametreler.
                    temperature, max_tokens gibi parametreler burada override edilebilir.

        Returns:
            Modelin ürettiği metin (content).

        Raises:
            RuntimeError: API çağrısı başarısız olursa.
        """
        import litellm

        # kwargs'ta açıkça belirtilmemiş parametreler için default değerleri kullan
        # Bu sayede çağıran taraf temperature/max_tokens override edebilir
        call_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key": self._api_key,
        }
        # kwargs varsa default'ların üzerine yaz
        call_params.update(kwargs)

        try:
            response = litellm.completion(**call_params)
            content: str = response.choices[0].message.content or ""
            logger.debug("LLM yanıtı alındı: %d karakter.", len(content))
            return content
        except Exception as exc:
            raise RuntimeError(f"LLM API çağrısı başarısız: {exc}") from exc

    def chat_with_context(
        self,
        manager: Any,  # ContextManager — döngüsel import önlemek için Any
        user_question: str,
        format_mode: str = "default",
        inject_system: bool = True,
    ) -> str:
        """ContextManager içeriğini mesajlara dönüştürüp LLM'e gönderir.

        Args:
            manager: ContextManager nesnesi.
            user_question: LLM'e sorulacak soru.
            format_mode: PromptFormatter modu ("default", "xml", "minimal").
            inject_system: True ise provenance sistem açıklaması eklenir.

        Returns:
            Modelin ürettiği metin.
        """
        from sca.formatter import PromptFormatter

        formatter = PromptFormatter(mode=format_mode)  # type: ignore[arg-type]
        system_prompt = formatter.get_system_prompt() if inject_system else None
        messages = manager.to_messages(system_prompt=system_prompt)
        messages.append({"role": "user", "content": user_question})
        return self.chat(messages)