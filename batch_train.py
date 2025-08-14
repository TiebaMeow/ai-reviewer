"""
批量训练脚本

从CSV文件读取训练数据并批量提交给AI Reviewer服务进行增量学习。

CSV文件格式：
- text: 文本内容
- task: 任务名称
- label: 标签

使用示例：
    python batch_train.py --csv data/training_data.csv --url http://localhost:8000
    python batch_train.py --csv data/training_data.csv --batch-size 10 --delay 0.5
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class BatchTrainer:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        初始化批量训练器

        Args:
            base_url: AI Reviewer服务的基础URL
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # 配置重试策略
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)

        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def check_server(self) -> bool:
        """检查服务器是否可用"""
        try:
            response = self.session.get(f"{self.base_url}/tasks", timeout=self.timeout)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ 服务器连接失败: {e}")
            return False

    def get_available_tasks(self) -> list[str]:
        """获取可用任务列表"""
        try:
            response = self.session.get(f"{self.base_url}/tasks", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("tasks", [])
        except Exception as e:
            print(f"❌ 获取任务列表失败: {e}")
            return []

    def register_task_if_needed(self, task_name: str, labels: list[str]) -> bool:
        """如果任务不存在则注册任务"""
        available_tasks = self.get_available_tasks()

        if task_name in available_tasks:
            print(f"✓ 任务 '{task_name}' 已存在")
            return True

        # 注册新任务
        try:
            payload = {
                "name": task_name,
                "labels": labels
            }
            response = self.session.post(
                f"{self.base_url}/tasks/register",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            print(f"✓ 成功注册任务 '{task_name}' 标签: {labels}")
            return True
        except Exception as e:
            print(f"❌ 注册任务 '{task_name}' 失败: {e}")
            return False

    def update_task(self, texts: list[str], task: str, labels: list[str]) -> bool:
        """更新任务模型"""
        try:
            payload = {
                "texts": texts,
                "task": task,
                "labels": labels
            }
            response = self.session.post(
                f"{self.base_url}/update",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"❌ 更新任务 '{task}' 失败: {e}")
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 404:
                    print(f"  任务 '{task}' 不存在，请先注册")
                elif e.response.status_code == 400:
                    print("  标签错误或数据格式不正确")
            return False

    def load_csv_data(self, csv_file: Path) -> list[dict[str, str]]:
        """从CSV文件加载训练数据"""
        data = []

        if not csv_file.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_file}")

        try:
            with csv_file.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)

                # 检查必需的列
                required_columns = {"text", "task", "label"}
                fieldnames = reader.fieldnames or []
                if not required_columns.issubset(fieldnames):
                    missing = required_columns - set(fieldnames)
                    raise ValueError(f"CSV文件缺少必需的列: {missing}")

                for row_num, row in enumerate(reader, start=2):
                    # 验证数据
                    if not row["text"].strip():
                        print(f"⚠️  第{row_num}行: 文本内容为空，跳过")
                        continue
                    if not row["task"].strip():
                        print(f"⚠️  第{row_num}行: 任务名称为空，跳过")
                        continue
                    if not row["label"].strip():
                        print(f"⚠️  第{row_num}行: 标签为空，跳过")
                        continue

                    data.append({
                        "text": row["text"].strip(),
                        "task": row["task"].strip(),
                        "label": row["label"].strip()
                    })

        except Exception as e:
            raise Exception(f"读取CSV文件失败: {e}") from e

        return data

    def group_by_task(self, data: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
        """按任务分组数据"""
        groups = {}
        for item in data:
            task = item["task"]
            if task not in groups:
                groups[task] = []
            groups[task].append(item)
        return groups

    def batch_train(
        self,
        csv_file: Path,
        batch_size: int = 50,
        delay: float = 0.1,
        auto_register: bool = True,
        shuffle: bool = True,
        seed: int | None = None,
        epochs: int = 1,
    ):
        """批量训练"""
        print(f"📁 加载CSV文件: {csv_file}")

        # 加载数据
        try:
            data = self.load_csv_data(csv_file)
        except Exception as e:
            print(f"❌ {e}")
            return False

        if not data:
            print("❌ 没有有效的训练数据")
            return False

        print(f"✓ 成功加载 {len(data)} 条训练数据")

        # 检查服务器
        if not self.check_server():
            return False

        # 按任务分组
        task_groups = self.group_by_task(data)
        print(f"📊 发现 {len(task_groups)} 个不同的任务:")

        for task_name, items in task_groups.items():
            # Preserve label order by first occurrence in CSV to ensure deterministic mapping
            seen: set[str] = set()
            labels: list[str] = []
            for item in items:
                lb = item["label"]
                if lb not in seen:
                    seen.add(lb)
                    labels.append(lb)
            print(f"  - {task_name}: {len(items)} 条数据, 标签: {labels}")

            # 自动注册任务
            if auto_register:
                if not self.register_task_if_needed(task_name, labels):
                    print(f"⚠️  跳过任务 '{task_name}' 的训练")
                    continue

        # 定义分层打乱与交替混合（按标签分桶并轮询取样）
        def stratified_interleave(items: list[dict[str, str]], rnd: random.Random) -> list[dict[str, str]]:
            buckets: dict[str, list[dict[str, str]]] = {}
            for it in items:
                buckets.setdefault(it["label"], []).append(it)
            # 独立打乱每个标签桶
            for arr in buckets.values():
                rnd.shuffle(arr)
            order = list(buckets.keys())
            rnd.shuffle(order)
            mixed: list[dict[str, str]] = []
            # 轮询取样直到所有桶耗尽（每轮使用一次性 extend）
            while any(buckets[k] for k in order):
                round_elems = [buckets[k].pop() for k in order if buckets[k]]
                mixed.extend(round_elems)
            return mixed

        # 批量训练
        total_success = 0
        total_failed = 0

        for task_name, items in task_groups.items():
            print(f"\n🔄 开始训练任务: {task_name}")
            # 进行多个 epoch 的分层打乱并分批提交
            for ep in range(epochs):
                rnd = random.Random((seed if seed is not None else random.randrange(1 << 30)) + ep)
                items_epoch = stratified_interleave(items, rnd) if shuffle else list(items)

                for i in range(0, len(items_epoch), batch_size):
                    batch = items_epoch[i:i + batch_size]
                    texts = [item["text"] for item in batch]
                    labels = [item["label"] for item in batch]

                    print(
                        f"  Epoch {ep + 1}/{epochs} 批次 {i // batch_size + 1}: 处理 {len(batch)} 条数据...",
                        end="",
                    )

                    if self.update_task(texts, task_name, labels):
                        print(" ✓")
                        total_success += len(batch)
                    else:
                        print(" ❌")
                        total_failed += len(batch)

                    # 延迟避免过于频繁的请求
                    if delay > 0:
                        time.sleep(delay)

        # 总结
        print("\n📈 训练完成!")
        print(f"  ✓ 成功: {total_success} 条")
        print(f"  ❌ 失败: {total_failed} 条")

        return total_failed == 0


def create_sample_csv(file_path: Path):
    """创建示例CSV文件"""
    sample_data = [
        {"text": "这种观点会导致社会分裂", "task": "滑坡", "label": "有"},
        {"text": "我认为这个政策很好", "task": "滑坡", "label": "无"},
        {"text": "你这种想法就是想挑起争端", "task": "引战", "label": "是"},
        {"text": "让我们理性讨论这个问题", "task": "引战", "label": "否"},
        {"text": "某某明星根本不如另一个明星", "task": "拉踩", "label": "是"},
        {"text": "我喜欢这个明星的作品", "task": "拉踩", "label": "否"},
    ]

    with file_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "task", "label"])
        writer.writeheader()
        writer.writerows(sample_data)

    print(f"✓ 已创建示例CSV文件: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="AI Reviewer 批量训练脚本")
    parser.add_argument("--csv", type=str, required=True, help="CSV训练数据文件路径")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="AI Reviewer服务URL")
    parser.add_argument("--batch-size", type=int, default=50, help="批处理大小")
    parser.add_argument("--delay", type=float, default=0.1, help="批次间延迟时间（秒）")
    parser.add_argument("--timeout", type=int, default=30, help="请求超时时间（秒）")
    parser.add_argument("--no-auto-register", action="store_true", help="不自动注册新任务")
    parser.add_argument("--create-sample", action="store_true", help="创建示例CSV文件")
    parser.add_argument("--no-shuffle", action="store_true", help="不打乱样本（默认打乱并分层混合）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，用于重现实验")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（对同一CSV重复多轮）")

    args = parser.parse_args()

    csv_file = Path(args.csv)

    # 创建示例文件
    if args.create_sample:
        create_sample_csv(csv_file)
        return

    # 执行批量训练
    trainer = BatchTrainer(base_url=args.url, timeout=args.timeout)

    success = trainer.batch_train(
        csv_file=csv_file,
        batch_size=args.batch_size,
        delay=args.delay,
        auto_register=not args.no_auto_register,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        epochs=args.epochs,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
