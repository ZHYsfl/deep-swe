"""分析层：把 JSONL 账本读进来，产出 per-task 汇总和 per-episode 学习曲线。

命令行用法（即 --help 输出）：

    usage: ledger.analyze [-h] {summary,curve} ...

    positional arguments:
      {summary,curve}
        summary        按 task 分桶汇总 token 用量与成本
        curve          按 episode 分桶产出学习曲线（CSV + PNG）

    usage: ledger.analyze summary [-h] [--model MODEL] ledger

      ledger         ledger.jsonl 路径
      --model MODEL  计价用模型名，须存在于 pricing.py 的 PRICE_TABLE
                     （默认取账本里最后一条记录的 model 字段）

    usage: ledger.analyze curve [-h] [--model MODEL] [--out OUT] ledger

      ledger         ledger.jsonl 路径
      --model MODEL  计价用模型名（当前 curve 不计价，此参数保留与 summary 对齐）
      --out OUT      curve.csv / curve.png 的输出目录（默认 ./report，不存在会自动创建）

只读 schema 和 pricing，不知道 litellm 的存在。
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ledger.pricing import ModelPrice, UnknownModelError, get_price
from ledger.schema import JsonlSink, TokenCounts, UsageRecord


@dataclass
class Bucket:
    """一组调用的聚合（按 task 或按 episode）。"""

    name: str
    calls: int = 0
    counts: TokenCounts = field(default_factory=TokenCounts)
    latency_ms: int = 0

    def add(self, record: UsageRecord) -> None:
        self.calls += 1
        self.latency_ms += record.latency_ms
        self.add_counts(record.counts)

    def add_counts(self, c: TokenCounts) -> None:
        mine = self.counts
        mine.input_fresh += c.input_fresh
        mine.input_cache_read += c.input_cache_read
        mine.input_cache_write += c.input_cache_write
        mine.output += c.output
        mine.output_reasoning += c.output_reasoning


def aggregate(records: list[UsageRecord], key: str) -> list[Bucket]:
    """按 'task' 或 'episode' 分桶聚合。"""
    buckets: dict[str, Bucket] = {}
    for r in records:
        if key == "episode":
            name = "?" if r.episode is None else str(r.episode)
        else:
            name = r.task or "(untagged)"
        buckets.setdefault(name, Bucket(name=name)).add(r)
    order = (lambda b: int(b.name) if b.name.isdigit() else 1 << 30) if key == "episode" else (lambda b: b.name)
    return sorted(buckets.values(), key=order)


def _fmt_cost(price: ModelPrice | None, counts: TokenCounts) -> str:
    if price is None:
        return "N/A"
    return f"${price.cost(counts):.4f}"


def cmd_summary(path: Path, model: str | None, price_override: dict | None) -> None:
    records = JsonlSink(path).read_all()
    if not records:
        sys.exit(f"账本为空：{path}")
    price = None
    try:
        price = get_price(model or records[-1].model, table=price_override)
    except UnknownModelError as e:
        print(f"[warn] {e}", file=sys.stderr)

    rows = aggregate(records, key="task")
    header = f"{'task':40s} {'calls':>5s} {'fresh_in':>10s} {'cache_read':>11s} {'output':>8s} {'cache%':>7s} {'cost':>10s}"
    print(header)
    print("-" * len(header))
    total = Bucket(name="TOTAL")
    for b in rows:
        total.calls += b.calls
        total.latency_ms += b.latency_ms
        total.add_counts(b.counts)
        cache_pct = b.counts.input_cache_read / max(b.counts.input_total, 1)
        print(f"{b.name[:40]:40s} {b.calls:5d} {b.counts.input_fresh:10,d} "
              f"{b.counts.input_cache_read:11,d} {b.counts.output_total:8,d} "
              f"{cache_pct:6.0%} {_fmt_cost(price, b.counts):>10s}")
    print("-" * len(header))
    cache_pct = total.counts.input_cache_read / max(total.counts.input_total, 1)
    print(f"{'TOTAL':40s} {total.calls:5d} {total.counts.input_fresh:10,d} "
          f"{total.counts.input_cache_read:11,d} {total.counts.output_total:8,d} "
          f"{cache_pct:6.0%} {_fmt_cost(price, total.counts):>10s}")
    if price:
        print(f"\n价格来源：ledger/pricing.py PRICE_TABLE[{price.name!r}]")


def cmd_curve(path: Path, out_dir: Path, model: str | None) -> None:
    records = JsonlSink(path).read_all()
    episodes = aggregate(records, key="episode")
    if not any(b.name != "?" for b in episodes):
        sys.exit("账本里没有 episode 标签（X-Ledger-Episode 请求头），画不了学习曲线。")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "curve.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "calls", "fresh_input", "cache_read", "output", "cache_pct"])
        for b in episodes:
            w.writerow([b.name, b.calls, b.counts.input_fresh, b.counts.input_cache_read,
                        b.counts.output_total,
                        round(b.counts.input_cache_read / max(b.counts.input_total, 1), 4)])
    print(f"CSV 已写出：{csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] 无 matplotlib，跳过画图（CSV 已足够）")
        return

    xs = [int(b.name) if b.name.isdigit() else -1 for b in episodes]
    fresh = [b.counts.input_fresh for b in episodes]
    cached = [b.counts.input_cache_read for b in episodes]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.stackplot(xs, fresh, cached, labels=["fresh input", "cache read"], alpha=0.8)
    ax.set_xlabel("episode")
    ax.set_ylabel("input tokens / call")
    ax.set_title("Cost learning curve (fresh tokens = memory's real contribution)")
    ax.legend()
    fig.tight_layout()
    png = out_dir / "curve.png"
    fig.savefig(png, dpi=120)
    print(f"图已写出：{png}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ledger.analyze",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p_summary = sub.add_parser(
        "summary",
        help="按 task 分桶汇总 token 用量与成本",
        description="按 task 分桶打印 calls / fresh_in / cache_read / output / cache% / cost，"
                    "末尾附 TOTAL 行和价格来源。",
    )
    p_summary.add_argument("ledger", type=Path, help="ledger.jsonl 路径")
    p_summary.add_argument(
        "--model",
        help="计价用模型名，须存在于 pricing.py 的 PRICE_TABLE"
             "（默认取账本里最后一条记录的 model 字段）",
    )

    p_curve = sub.add_parser(
        "curve",
        help="按 episode 分桶产出学习曲线（CSV + PNG）",
        description="按 X-Ledger-Episode 标签分桶，输出 curve.csv（各 episode 的 token 分解）"
                    "和 curve.png（fresh input 随 episode 的堆积面积图，"
                    "fresh 的下降才是 memory 的净贡献）。账本没有 episode 标签时报错退出。",
    )
    p_curve.add_argument("ledger", type=Path, help="ledger.jsonl 路径")
    p_curve.add_argument(
        "--model",
        help="计价用模型名（当前 curve 不计价，此参数保留与 summary 对齐）",
    )
    p_curve.add_argument(
        "--out", type=Path, default=Path("report"),
        help="curve.csv / curve.png 的输出目录（默认 ./report，不存在会自动创建）",
    )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(2)

    if args.command == "summary":
        cmd_summary(args.ledger, args.model, price_override=None)
    else:
        cmd_curve(args.ledger, args.out, args.model)


if __name__ == "__main__":
    main()
