"""cost-ledger：harness 无关的 token 成本账本。

分层（每层只依赖下一层）：
    schema.py   数据层：UsageRecord / TokenCounts / 各家 usage 归一化 / JSONL 读写
    callback.py 采集层：LiteLLM proxy 回调，唯一 import litellm 的地方
    pricing.py  定价层：缓存三档分别计价，价格表显式可溯源
    analyze.py  分析层：per-task 汇总 / per-episode 学习曲线
"""

__version__ = "0.1.0"
