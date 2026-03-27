"""Shared retry/backoff and circuit-breaker utilities.

This module centralizes transient-error handling for runtime model calls and
critical remote tools (RAG/search/memory/example retrieval).
"""

from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
_STATUS_RE = re.compile(r"\b(408|429|500|502|503|504)\b")


@dataclass(frozen=True)
class RetrySettings:
    enabled: bool
    max_attempts: int
    initial_delay_seconds: float
    max_delay_seconds: float
    exp_base: float
    jitter: float


@dataclass(frozen=True)
class CircuitBreakerSettings:
    enabled: bool
    failure_threshold: int
    open_seconds: float


@dataclass
class CircuitBreaker:
    failure_threshold: int
    open_seconds: float
    state: str = "closed"
    failures: int = 0
    opened_at: float | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def allow_request(self) -> tuple[bool, float | None]:
        with self._lock:
            if self.state != "open":
                return True, None

            opened_at = self.opened_at or 0.0
            elapsed = time.monotonic() - opened_at
            if elapsed >= self.open_seconds:
                self.state = "half_open"
                return True, None

            return False, max(0.0, self.open_seconds - elapsed)

    def record_success(self) -> None:
        with self._lock:
            self.state = "closed"
            self.failures = 0
            self.opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            if self.state == "half_open":
                self.state = "open"
                self.opened_at = time.monotonic()
                self.failures = self.failure_threshold
                return

            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.state = "open"
                self.opened_at = time.monotonic()


class CircuitBreakerOpenError(RuntimeError):
    def __init__(self, operation: str, retry_after_seconds: float):
        self.operation = operation
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Circuit breaker open for {operation}. Retry after "
            f"{retry_after_seconds:.2f}s"
        )


_breaker_registry_lock = threading.Lock()
_breaker_registry: dict[str, CircuitBreaker] = {}


def load_retry_settings() -> RetrySettings:
    return RetrySettings(
        enabled=_env_bool("F1_RETRY_ENABLED", True),
        max_attempts=max(1, _env_int("F1_RETRY_MAX_ATTEMPTS", 3)),
        initial_delay_seconds=max(0.0, _env_float("F1_RETRY_INITIAL_DELAY_S", 0.4)),
        max_delay_seconds=max(0.1, _env_float("F1_RETRY_MAX_DELAY_S", 4.0)),
        exp_base=max(1.1, _env_float("F1_RETRY_EXP_BASE", 2.0)),
        jitter=max(0.0, _env_float("F1_RETRY_JITTER", 0.35)),
    )


def load_circuit_settings() -> CircuitBreakerSettings:
    return CircuitBreakerSettings(
        enabled=_env_bool("F1_CIRCUIT_ENABLED", True),
        failure_threshold=max(1, _env_int("F1_CIRCUIT_FAILURE_THRESHOLD", 5)),
        open_seconds=max(1.0, _env_float("F1_CIRCUIT_OPEN_SECONDS", 20.0)),
    )


def run_with_retry(
    operation: str,
    fn: Callable[[], Any],
    *,
    retry: RetrySettings | None = None,
    circuit: CircuitBreakerSettings | None = None,
    logger_instance: logging.Logger | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    random_fn: Callable[[], float] = random.random,
) -> Any:
    retry_settings = retry or load_retry_settings()
    circuit_settings = circuit or load_circuit_settings()
    log = logger_instance or logger

    breaker = _get_breaker(operation, circuit_settings)

    attempts = 1 if not retry_settings.enabled else retry_settings.max_attempts
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        if breaker is not None:
            allowed, retry_after = breaker.allow_request()
            if not allowed:
                raise CircuitBreakerOpenError(operation, retry_after or 0.0)

        try:
            result = fn()
            if breaker is not None:
                breaker.record_success()
            if attempt > 1:
                log.info(
                    "resilience_retry_recovered operation=%s attempt=%d",
                    operation,
                    attempt,
                )
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            transient, status_code, error_type = classify_transient_error(exc)

            if transient and breaker is not None:
                breaker.record_failure()

            if not transient or attempt >= attempts:
                log.warning(
                    "resilience_retry_exhausted operation=%s attempt=%d attempts=%d "
                    "transient=%s status_code=%s error_type=%s",
                    operation,
                    attempt,
                    attempts,
                    transient,
                    status_code,
                    error_type,
                    exc_info=True,
                )
                raise

            delay = backoff_delay_seconds(
                attempt=attempt,
                initial=retry_settings.initial_delay_seconds,
                exp_base=retry_settings.exp_base,
                max_delay=retry_settings.max_delay_seconds,
                jitter=retry_settings.jitter,
                random_fn=random_fn,
            )
            log.warning(
                "resilience_retry operation=%s attempt=%d/%d delay_s=%.3f "
                "status_code=%s error_type=%s",
                operation,
                attempt,
                attempts,
                delay,
                status_code,
                error_type,
            )
            sleep_fn(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Unexpected retry flow for operation={operation}")


def backoff_delay_seconds(
    *,
    attempt: int,
    initial: float,
    exp_base: float,
    max_delay: float,
    jitter: float,
    random_fn: Callable[[], float] = random.random,
) -> float:
    base = min(max_delay, initial * (exp_base ** max(0, attempt - 1)))
    if jitter <= 0:
        return base
    # Multiplicative jitter in [1 - jitter, 1 + jitter].
    jitter_factor = 1 + ((random_fn() * 2.0 - 1.0) * jitter)
    jittered = base * jitter_factor
    return max(0.0, min(max_delay, jittered))


def classify_transient_error(error: BaseException) -> tuple[bool, int | None, str]:
    status_code = _extract_status_code(error)
    error_type = type(error).__name__
    if status_code in _RETRYABLE_HTTP_CODES:
        return True, status_code, error_type

    text = str(error)
    lowered = text.lower()
    transient_tokens = (
        "resourceexhausted",
        "service unavailable",
        "unavailable",
        "deadline exceeded",
        "temporarily unavailable",
        "rate limit",
        "too many requests",
        "timeout",
        "timed out",
        "connection reset",
    )
    if any(token in lowered for token in transient_tokens):
        return True, status_code, error_type

    return False, status_code, error_type


def is_quota_or_unavailable_error(error: BaseException) -> bool:
    transient, status_code, _ = classify_transient_error(error)
    if not transient:
        return False
    if status_code in {429, 503}:
        return True
    text = str(error).lower()
    return "resourceexhausted" in text or "service unavailable" in text


def _extract_status_code(error: BaseException) -> int | None:
    for attr in ("status_code", "code", "status"):
        value = getattr(error, attr, None)
        parsed = _parse_status_like(value)
        if parsed is not None:
            return parsed

    response = getattr(error, "response", None)
    if response is not None:
        parsed = _parse_status_like(getattr(response, "status_code", None))
        if parsed is not None:
            return parsed

    message = str(error)
    match = _STATUS_RE.search(message)
    if match:
        return int(match.group(1))
    return None


def _parse_status_like(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, int):
        return value

    if callable(value):
        try:
            called = value()
        except Exception:  # noqa: BLE001
            called = None
        parsed = _parse_status_like(called)
        if parsed is not None:
            return parsed

    text = str(value)
    if text.isdigit():
        return int(text)
    match = _STATUS_RE.search(text)
    if match:
        return int(match.group(1))
    return None


def _get_breaker(
    operation: str,
    settings: CircuitBreakerSettings,
) -> CircuitBreaker | None:
    if not settings.enabled:
        return None

    with _breaker_registry_lock:
        existing = _breaker_registry.get(operation)
        if existing is not None:
            return existing
        breaker = CircuitBreaker(
            failure_threshold=settings.failure_threshold,
            open_seconds=settings.open_seconds,
        )
        _breaker_registry[operation] = breaker
        return breaker


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default
