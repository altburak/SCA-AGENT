"""Groq API key rotation manager.

Loads multiple GROQ_API_KEY_N env vars, rotates through them automatically
when one hits rate limit (TPM or TPD). Thread-safe singleton.

Usage:
    from benchmarks.runner.key_manager import get_key_manager

    mgr = get_key_manager()
    key = mgr.current_key()
    # ...call LLM...
    # if rate limit error:
    mgr.mark_exhausted(key, cooldown_seconds=60)  # TPM (short)
    # or:
    mgr.mark_exhausted(key, cooldown_seconds=3600)  # TPD (long)
    key = mgr.current_key()  # now returns next healthy key
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


from dotenv import load_dotenv

load_dotenv()

@dataclass
class KeyState:
    key: str
    index: int
    cooldown_until: float = 0.0  # unix timestamp
    exhaustion_count: int = 0
    total_calls: int = 0

    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until

    def seconds_until_available(self) -> float:
        return max(0.0, self.cooldown_until - time.time())


class KeyManager:
    """Round-robin with health tracking across multiple Groq API keys."""

    def __init__(self, keys: list[str]) -> None:
        if not keys:
            raise RuntimeError(
                "No Groq API keys found. Set GROQ_API_KEY or "
                "GROQ_API_KEY_1, GROQ_API_KEY_2, ... in your .env"
            )
        self._keys: list[KeyState] = [
            KeyState(key=k, index=i) for i, k in enumerate(keys)
        ]
        self._current_idx: int = 0
        self._lock = threading.Lock()
        print(f"[KeyManager] Loaded {len(keys)} Groq API key(s).")

    def current_key(self) -> str:
        """Return a healthy key, rotating if current one is on cooldown.

        Raises RuntimeError if ALL keys are on cooldown.
        """
        with self._lock:
            n = len(self._keys)
            for _ in range(n):
                state = self._keys[self._current_idx]
                if state.is_available():
                    state.total_calls += 1
                    return state.key
                # Try next
                self._current_idx = (self._current_idx + 1) % n

            # All keys exhausted — find the one with shortest wait
            soonest = min(self._keys, key=lambda s: s.cooldown_until)
            wait = soonest.seconds_until_available()
            raise RuntimeError(
                f"All {n} Groq API keys exhausted. "
                f"Soonest available in {wait:.0f}s "
                f"(key #{soonest.index + 1})."
            )

    def mark_exhausted(
        self,
        key: str,
        cooldown_seconds: float = 60.0,
        reason: str = "",
    ) -> None:
        """Mark a key as on cooldown. Rotates to next key immediately."""
        with self._lock:
            for state in self._keys:
                if state.key == key:
                    state.cooldown_until = time.time() + cooldown_seconds
                    state.exhaustion_count += 1
                    print(
                        f"[KeyManager] Key #{state.index + 1} on cooldown "
                        f"for {cooldown_seconds:.0f}s ({reason})"
                    )
                    # Rotate
                    self._current_idx = (self._current_idx + 1) % len(self._keys)
                    return

    def status(self) -> str:
        """Human-readable status of all keys."""
        lines = []
        for s in self._keys:
            avail = "✓" if s.is_available() else f"✗ (wait {s.seconds_until_available():.0f}s)"
            lines.append(
                f"  Key #{s.index + 1}: {avail} "
                f"calls={s.total_calls} exhaustions={s.exhaustion_count}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Error parsing
# ---------------------------------------------------------------------------

_RETRY_RE = re.compile(r"try again in ([\d.]+)(m)?([\d.]+)?s", re.IGNORECASE)
_TPD_RE = re.compile(r"tokens per day|TPD", re.IGNORECASE)
_TPM_RE = re.compile(r"tokens per minute|TPM", re.IGNORECASE)


def parse_rate_limit_error(err_msg: str) -> tuple[float, str]:
    """Parse a Groq rate limit error message.

    Returns (cooldown_seconds, reason_tag).
    Defaults to (60s, "unknown") if parsing fails.
    """
    if not err_msg:
        return (60.0, "unknown")

    # Determine kind
    is_tpd = bool(_TPD_RE.search(err_msg))
    is_tpm = bool(_TPM_RE.search(err_msg))

    # Extract retry-after — examples:
    #   "try again in 3.47s"      -> 3.47s
    #   "try again in 17m6.432s"  -> 17*60 + 6.432
    #   "try again in 34.56s"
    retry_seconds = None
    m = re.search(r"try again in ([\d]+)m([\d.]+)s", err_msg)
    if m:
        retry_seconds = int(m.group(1)) * 60 + float(m.group(2))
    else:
        m = re.search(r"try again in ([\d.]+)s", err_msg)
        if m:
            retry_seconds = float(m.group(1))

    if retry_seconds is None:
        # Fallback by type
        if is_tpd:
            retry_seconds = 3600.0  # 1 hour, err on the safe side
        elif is_tpm:
            retry_seconds = 60.0
        else:
            retry_seconds = 60.0

    # Add a small buffer to be safe
    retry_seconds = retry_seconds + 2.0

    if is_tpd:
        tag = "TPD"
    elif is_tpm:
        tag = "TPM"
    else:
        tag = "rate_limit"
    return (retry_seconds, tag)


def is_rate_limit_error(err: Exception | str) -> bool:
    msg = str(err)
    return "rate_limit" in msg.lower() or "ratelimiterror" in msg.lower()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_manager: Optional[KeyManager] = None
_manager_lock = threading.Lock()


def _load_keys_from_env() -> list[str]:
    """Collect all Groq keys from env: GROQ_API_KEY, GROQ_API_KEY_1..N."""
    keys: list[str] = []
    seen: set[str] = set()

    # Numbered keys first (1..20 max)
    for i in range(1, 21):
        k = os.environ.get(f"GROQ_API_KEY_{i}")
        if k and k.strip() and k not in seen:
            keys.append(k.strip())
            seen.add(k.strip())

    # Plain GROQ_API_KEY as fallback
    plain = os.environ.get("GROQ_API_KEY")
    if plain and plain.strip() and plain.strip() not in seen:
        keys.append(plain.strip())
        seen.add(plain.strip())

    return keys


def get_key_manager() -> KeyManager:
    """Get or create the global KeyManager singleton."""
    global _manager
    with _manager_lock:
        if _manager is None:
            keys = _load_keys_from_env()
            _manager = KeyManager(keys)
        return _manager


def reset_key_manager() -> None:
    """Reset the singleton (mostly for tests)."""
    global _manager
    with _manager_lock:
        _manager = None
