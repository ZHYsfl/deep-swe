"""用量记录的数据层：schema 定义 + 各家 API usage 字段归一化。

只存原始 token 计数，不存美元——定价换算是分析层的事（ledger/pricing.py）。
本模块不依赖 litellm，纯函数 + dataclass，可独立测试。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TokenCounts:
    """一次模型调用的 token 计数，输入侧按缓存来源分三档。

    不变量：input_fresh + input_cache_read + input_cache_write == input_total
    （DeepSeek/Kimi 都是自动前缀缓存，没有写缓存概念，cache_write 恒为 0；
    字段留着是为了 schema 稳定，万一以后接显式缓存的厂商不用改账本格式）。
    """

    input_fresh: int = 0        # 未命中任何缓存、按全价计费的输入 token
    input_cache_read: int = 0   # 命中前缀缓存、按缓存价计费的输入 token
    input_cache_write: int = 0  # 写入缓存的输入 token（DeepSeek/Kimi 恒为 0）
    output: int = 0             # 输出 token（不含 reasoning）
    output_reasoning: int = 0   # reasoning token（含在输出总数内，仅作明细）

    @property
    def input_total(self) -> int:
        return self.input_fresh + self.input_cache_read + self.input_cache_write

    @property
    def output_total(self) -> int:
        return self.output + self.output_reasoning


@dataclass
class UsageRecord:
    """一行 JSONL 的完整内容：一次模型调用 + 调用归属标签。

    标签（run/task/episode/step）由调用方通过请求头注入，proxy 原样透传：
      X-Ledger-Run     实验批次，如 "ab-nomem-20260727"
      X-Ledger-Task    任务 ID，如 "sales.multi_hop_lookup"
      X-Ledger-Episode episode 序号（学习曲线的 X 轴）
      X-Ledger-Step    任务内步数（可选）
    """

    ts: float
    model: str
    counts: TokenCounts
    run: str = ""
    task: str = ""
    episode: int | None = None
    step: int | None = None
    latency_ms: int = 0
    finish_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        d = asdict(self)
        d["counts"] = asdict(self.counts)
        return json.dumps(d, ensure_ascii=False)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "UsageRecord":
        d = dict(d)
        d["counts"] = TokenCounts(**d.get("counts", {}))
        return UsageRecord(**d)


def _get(obj: Any, name: str, default: int = 0) -> int:
    """从对象或 dict 里取 int 字段，None 视为 default。"""
    v = _sub(obj, name)
    return int(v) if v is not None else default


def _sub(obj: Any, name: str) -> Any:
    """取嵌套字段，兼容属性和 dict 两种形态；取不到返回 None。"""
    if obj is None:
        return None
    v = getattr(obj, name, None)
    if v is None and isinstance(obj, dict):
        v = obj.get(name)
    return v


def normalize_usage(usage: Any) -> TokenCounts:
    """把 usage 字段归一成 TokenCounts。只支持 OpenAI chat completions 规范。

    目标 provider 是 DeepSeek 和 Kimi，两家都走 OpenAI chat completions
    格式，差别只在缓存命中字段的位置：

      - DeepSeek: prompt_cache_hit/miss_tokens 互斥分解，
                  hit + miss == prompt_tokens（官方 schema 明示）。
      - Kimi:     cached_tokens 可能在顶层（官方 schema），也可能在
                  prompt_tokens_details（K3 实测），两处都兜底；
                  命中部分 ⊆ prompt_tokens。

    两家都是服务端自动前缀缓存，无写缓存概念，cache_write 恒为 0。
    reasoning（completion_tokens_details.reasoning_tokens）已含在
    completion_tokens 内，拆出仅作明细，绝不重复计费
    （AutomationBench 的 usage.py 在这里加了两遍，是本模块刻意避开的坑）。
    什么都不报的响应退化为全部 input_fresh。
    """
    if usage is None:
        return TokenCounts()

    completion = _get(usage, "completion_tokens")

    # --- 输入侧缓存分层 -------------------------------------------------
    if _get(usage, "prompt_cache_hit_tokens") or _get(usage, "prompt_cache_miss_tokens"):
        # DeepSeek：hit/miss 互斥分解，hit + miss == prompt_tokens
        cache_read = _get(usage, "prompt_cache_hit_tokens")
        fresh = _get(usage, "prompt_cache_miss_tokens")
    else:
        # OpenAI chat 规范：cached ⊆ prompt_tokens。Kimi 顶层字段兜底。
        total = _get(usage, "prompt_tokens")
        cache_read = _get(_sub(usage, "prompt_tokens_details"), "cached_tokens")
        if not cache_read:
            cache_read = _get(usage, "cached_tokens")
        fresh = max(total - cache_read, 0)

    # --- reasoning：从输出里拆出来，不重复计数 ---------------------------
    reasoning = _get(_sub(usage, "completion_tokens_details"), "reasoning_tokens")
    output = max(completion - reasoning, 0)

    return TokenCounts(
        input_fresh=fresh,
        input_cache_read=cache_read,
        output=output,
        output_reasoning=reasoning,
    )


class JsonlSink:
    """线程安全的 JSONL 追加写。append-only，坏了单行不影响整体。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        import threading

        self._lock = threading.Lock()

    def write(self, record: UsageRecord) -> None:
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(record.to_jsonl() + "\n")

    def read_all(self) -> list[UsageRecord]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(UsageRecord.from_dict(json.loads(line)))
        return records


def now() -> float:
    return time.time()
