"""LiteLLM proxy 回调：把每次成功调用记成一行 JSONL。

本模块只做两件事——从 litellm 回调参数里提取（标签, usage, 元信息），
然后委托给 schema 层归一化和落盘。litellm 相关的脏活全部收敛在这里，
schema/pricing/analyze 都不 import litellm。

配置方式（proxy.config.yaml）：
    litellm_settings:
      callbacks: ledger.callback.ledger_logger
"""

from __future__ import annotations

import os
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

from ledger.schema import JsonlSink, UsageRecord, normalize_usage, now

# 调用方注入归属标签的请求头（proxy 会把 header 名小写化）
_HEADER_MAP = {
    "x-ledger-run": "run",
    "x-ledger-task": "task",
    "x-ledger-episode": "episode",
    "x-ledger-step": "step",
}

DEFAULT_LOG_PATH = os.environ.get(
    "LEDGER_LOG_PATH",
    os.path.join(os.path.dirname(__file__), "..", "logs", "ledger.jsonl"),
)


def _extract_tags(kwargs: dict[str, Any]) -> dict[str, Any]:
    """提取调用归属标签：优先请求头，其次 litellm metadata 里的 ledger_* 键。

    注意 litellm 的结构：proxy_server_request 藏在 litellm_params 下面，
    不在 kwargs 顶层（这是实测确认的，不同版本位置可能漂移，所以两处都找）。
    """
    tags: dict[str, Any] = {}

    litellm_params = kwargs.get("litellm_params") or {}
    proxy_req = kwargs.get("proxy_server_request") or litellm_params.get("proxy_server_request") or {}
    headers = proxy_req.get("headers") or {}
    for header, field_name in _HEADER_MAP.items():
        value = headers.get(header)
        if value not in (None, ""):
            tags[field_name] = value

    metadata = litellm_params.get("metadata") or {}
    for key in ("run", "task", "episode", "step"):
        value = metadata.get(f"ledger_{key}")
        if value not in (None, ""):
            tags.setdefault(key, value)

    for int_key in ("episode", "step"):
        if int_key in tags:
            try:
                tags[int_key] = int(tags[int_key])
            except (TypeError, ValueError):
                tags.pop(int_key)
    return tags


class LedgerCallback(CustomLogger):
    """litellm CustomLogger：成功一次，记一行。失败调用也记（output=0 + finish_reason=error）。"""

    def __init__(self, log_path: str | None = None):
        super().__init__()
        self._sink = JsonlSink(log_path or DEFAULT_LOG_PATH)

    def _record(self, kwargs: dict[str, Any], response_obj: Any, error: str = "") -> None:
        start, end = kwargs.get("start_time"), kwargs.get("end_time")
        latency_ms = int((end - start).total_seconds() * 1000) if start and end else 0

        response = getattr(response_obj, "model_response", None) or response_obj
        usage = getattr(response, "usage", None)
        choices = getattr(response, "choices", None) or []
        finish_reason = (
            error or (getattr(choices[0], "finish_reason", "") if choices else "")
        )

        self._sink.write(
            UsageRecord(
                ts=now(),
                model=kwargs.get("model", ""),
                counts=normalize_usage(usage),
                latency_ms=latency_ms,
                finish_reason=finish_reason or "",
                **_extract_tags(kwargs),
            )
        )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, error="error")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, error="error")


# litellm 按配置里的 "ledger.callback.ledger_logger" 找到这个实例
ledger_logger = LedgerCallback()
