"""定价层：token 计数 × 单价 = 成本。与分析层分离，价格表可整体替换。

原则（从 AutomationBench 学的 + 修正）：
  - 缓存三档分别计价，不用单一 input 价——否则"成本递减"全是缓存假象
  - 落盘数据（JSONL）只有 token 计数，美元永远在这里现算
  - 价格表是显式数据，每次分析打印来源，实验报告可溯源
"""

from __future__ import annotations

from dataclasses import dataclass

from ledger.schema import TokenCounts

# 单价：美元 / 百万 token。请按各 provider 官网刊例价维护，改了就是改了，
# 不要在代码里"偷偷更新"——实验间可比性靠价格表版本一致保证。
PRICE_TABLE: dict[str, dict[str, float]] = {
    # DeepSeek 官方刊例价（2026-07，直接用官方美元牌价：
    #   https://api-docs.deepseek.com/quick_start/pricing
    "deepseek-v4-flash": {
        "input_fresh": 0.14,           # $0.14   （缓存未命中，¥1.00）
        "input_cache_read": 0.0028,    # $0.0028 （缓存命中，¥0.02）
        "output": 0.28,                # $0.28   （¥2.00）
    },
    "deepseek-v4-pro": {
        "input_fresh": 0.435,          # $0.435   （¥3.00）
        "input_cache_read": 0.003625,  # $0.003625（¥0.025）
        "output": 0.87,                # $0.87    （¥6.00）
    },
    # Kimi（月之暗面）国际站官方美元刊例价（2026-07，platform.kimi.ai 直接以 USD 计价）：
    #   https://platform.kimi.ai/docs/pricing/chat-k3
    #   https://platform.kimi.ai/docs/pricing/code-k2.7
    #   https://platform.kimi.ai/docs/pricing/chat-k26
    "kimi-k3": {
        "input_fresh": 3.00,         # 缓存未命中
        "input_cache_read": 0.30,    # 缓存命中（1/10）
        "output": 15.00,
    },
    "kimi-k2.7-code": {
        "input_fresh": 0.95,
        "input_cache_read": 0.19,    # 1/5
        "output": 4.00,
    },
    "kimi-k2.7-code-highspeed": {  # 与 k2.7-code 同模型，~180 tok/s 输出
        "input_fresh": 1.90,
        "input_cache_read": 0.38,
        "output": 8.00,
    },
    "kimi-k2.6": {
        "input_fresh": 0.95,
        "input_cache_read": 0.16,    # ~1/6
        "output": 4.00,
    },
}


@dataclass(frozen=True)
class ModelPrice:
    """单个模型的分档单价（美元 / token）。"""

    name: str
    input_fresh: float
    input_cache_read: float
    input_cache_write: float
    output: float

    def cost(self, counts: TokenCounts) -> float:
        return (
            counts.input_fresh * self.input_fresh
            + counts.input_cache_read * self.input_cache_read
            + counts.input_cache_write * self.input_cache_write
            + counts.output_total * self.output
        )


class UnknownModelError(KeyError):
    pass


def get_price(model: str, table: dict[str, dict[str, float]] | None = None) -> ModelPrice:
    """按模型名查价。精确匹配 → 去 provider 前缀匹配，找不到抛 UnknownModelError。"""
    table = table if table is not None else PRICE_TABLE
    entry = table.get(model) or table.get(model.split("/")[-1])
    if entry is None:
        raise UnknownModelError(
            f"价格表里没有 {model!r}。请在 ledger/pricing.py 的 PRICE_TABLE 补充，"
            "或分析时通过 --price 参数显式给出。"
        )
    return ModelPrice(
        name=model,
        input_fresh=entry["input_fresh"] / 1_000_000,
        input_cache_read=entry["input_cache_read"] / 1_000_000,
        input_cache_write=entry.get("input_cache_write", entry["input_fresh"]) / 1_000_000,
        output=entry["output"] / 1_000_000,
    )
