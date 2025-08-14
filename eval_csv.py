"""
评测脚本：读取与训练相同格式的 CSV（text,task,label），
按任务分组后调用服务端 /eval 计算任务与总体准确率。

用法示例：
  python eval_csv.py --csv test_data.csv
  python eval_csv.py --csv test_data.csv --url http://localhost:8000 --timeout 60 --fail-under 0.8
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class TaskEvalResult:
    task: str
    size: int
    accuracy: float
    per_label: dict[str, dict[str, float]]


class Evaluator:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def check_server(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/tasks", timeout=self.timeout)
            r.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ 服务器不可用: {e}")
            return False

    @staticmethod
    def load_csv(csv_path: Path) -> list[dict[str, str]]:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        data: list[dict[str, str]] = []
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"text", "task", "label"}
            fields = set(reader.fieldnames or [])
            if not required.issubset(fields):
                missing = required - fields
                raise ValueError(f"CSV文件缺少必需列: {missing}")
            for row_num, row in enumerate(reader, start=2):
                text = (row.get("text") or "").strip()
                task = (row.get("task") or "").strip()
                label = (row.get("label") or "").strip()
                if not text or not task or not label:
                    print(f"⚠️  第{row_num}行无效，跳过")
                    continue
                data.append({"text": text, "task": task, "label": label})
        return data

    @staticmethod
    def group_by_task(data: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
        groups: dict[str, list[dict[str, str]]] = {}
        for item in data:
            groups.setdefault(item["task"], []).append(item)
        return groups

    def eval_task(self, task: str, items: list[dict[str, str]]) -> TaskEvalResult | None:
        texts = [it["text"] for it in items]
        labels = [it["label"] for it in items]
        payload = {"task": task, "texts": texts, "labels": labels}
        try:
            r = self.session.post(f"{self.base_url}/eval", json=payload, timeout=self.timeout)
            if r.status_code == 404:
                print(f"❌ 任务不存在: {task}")
                return None
            r.raise_for_status()
            data = r.json()
            return TaskEvalResult(
                task=data["task"],
                size=int(data["size"]),
                accuracy=float(data["accuracy"]),
                per_label={
                    str(k): {
                        "support": float(v.get("support", 0.0)),
                        "accuracy": float(v.get("accuracy", 0.0)),
                    }
                    for k, v in (data.get("per_label") or {}).items()
                },
            )
        except requests.RequestException as e:
            print(f"❌ 评测任务 '{task}' 失败: {e}")
            return None


def main() -> int:
    parser = argparse.ArgumentParser(description="评测 CSV 准确率（按任务分组，服务端 /eval 计算）")
    parser.add_argument("--csv", type=str, required=True, help="CSV文件路径（包含 text,task,label）")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="AI Reviewer 服务 URL")
    parser.add_argument("--timeout", type=int, default=30, help="请求超时时间（秒）")
    parser.add_argument("--fail-under", type=float, default=None, help="总体准确率低于该阈值则以非零码退出")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    ev = Evaluator(base_url=args.url, timeout=args.timeout)

    if not ev.check_server():
        return 2

    try:
        data = ev.load_csv(csv_path)
    except Exception as e:
        print(f"❌ 读取CSV失败: {e}")
        return 2

    if not data:
        print("❌ CSV 中无有效数据")
        return 2

    groups = ev.group_by_task(data)
    if not groups:
        print("❌ 未找到任何任务数据")
        return 2

    print(f"📊 发现 {len(groups)} 个任务，开始评测…\n")

    results: list[TaskEvalResult] = []
    total = 0
    correct = 0.0

    for task, items in groups.items():
        res = ev.eval_task(task, items)
        if res is None:
            continue
        results.append(res)
        total += res.size
        correct += res.accuracy * res.size

        print(f"任务: {res.task}")
        print(f"  样本数: {res.size}")
        print(f"  准确率: {res.accuracy:.4f}")
        if res.per_label:
            print("  分标签:")
            for lb, m in res.per_label.items():
                print(f"    - {lb}: acc={m.get('accuracy', 0.0):.4f}, support={int(m.get('support', 0.0))}")
        print()

    if total == 0:
        print("❌ 没有成功评测的任务")
        return 2

    overall = correct / total if total else 0.0
    print("====================")
    print(f"总体: 样本数={total}, 准确率={overall:.4f}")

    if args.fail_under is not None and overall < args.fail_under:
        print(f"❌ 总体准确率 {overall:.4f} 低于阈值 {args.fail_under}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
