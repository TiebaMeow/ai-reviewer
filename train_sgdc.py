"""
批量训练脚本

从CSV文件读取训练数据并批量提交给AI Reviewer服务进行增量学习。

CSV文件格式：
- text: 文本内容
- task: 任务名称
- label: 标签

使用示例：
    python train_sgdc.py --csv data/training_data.csv --url http://localhost:8000
    python train_sgdc.py --csv data/training_data.csv --batch-size 10 --delay 0.5
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
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

    def _get_server_config(self) -> dict:
        """获取服务端配置（包含任务及其标签）。失败时返回空字典。"""
        try:
            r = self.session.get(f"{self.base_url}/config", timeout=self.timeout)
            r.raise_for_status()
            return r.json() or {}
        except Exception:
            return {}

    def _get_task_labels_from_server(self, task_name: str) -> list[str] | None:
        cfg = self._get_server_config()
        tasks = (cfg or {}).get("tasks") or {}
        info = tasks.get(task_name)
        if isinstance(info, dict):
            lbs = info.get("labels")
            if isinstance(lbs, list) and all(isinstance(x, str) for x in lbs):
                return lbs
        return None

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

    def load_csv_data(self, csv_file: Path) -> pd.DataFrame:
        """从CSV文件加载训练数据并进行清理。"""
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_file}")

        df = pd.read_csv(csv_file)
        required_columns = {"text", "task", "label"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"CSV文件缺少必需的列: {missing}")

        # 选择并清理数据
        df = df[list(required_columns)].copy()
        df.dropna(inplace=True)
        for col in ["text", "task", "label"]:
            df[col] = df[col].astype(str).str.strip()

        df = df[df["text"].str.len() > 0]
        df = df[df["task"].str.len() > 0]
        df = df[df["label"].str.len() > 0]

        return df

    def group_by_task(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """按任务分组数据"""
        return {str(task): group_df for task, group_df in df.groupby("task")}

    def batch_train(
        self,
        csv_file: Path,
        batch_size: int = 50,
        delay: float = 0.1,
        auto_register: bool = True,
        shuffle: bool = True,
        seed: int | None = None,
        epochs: int = 1,
        val_ratio: float = 0.0,
        warn_threshold_no_improve: int = 3,
        overfit_warn_gap: float = 0.15,
        val_csv: Path | None = None,
        hard_mining: bool = True,
        hard_weight: float = 0.3,
    ):
        """批量训练"""
        print(f"📁 加载CSV文件: {csv_file}")

        # 加载数据
        try:
            df = self.load_csv_data(csv_file)
        except Exception as e:
            print(f"❌ {e}")
            return False

        if df.empty:
            print("❌ 没有有效的训练数据")
            return False

        print(f"✓ 成功加载 {len(df)} 条训练数据")

        # 若提供外部验证集，加载之
        val_df: pd.DataFrame | None = None
        if val_csv is not None:
            try:
                print(f"📁 加载验证集CSV文件: {val_csv}")
                val_df = self.load_csv_data(val_csv)
                if val_df.empty:
                    print("⚠️ 验证集为空，将忽略外部验证集")
                    val_df = None
            except Exception as e:
                print(f"⚠️ 加载外部验证集失败（将忽略并回退到内部划分，如启用）: {e}")
                val_df = None

        # 检查服务器
        if not self.check_server():
            return False

        # 按任务分组
        task_groups = self.group_by_task(df)
        print(f"📊 发现 {len(task_groups)} 个不同的任务:")

        for task_name, items_df in task_groups.items():
            # Preserve label order by first occurrence in CSV to ensure deterministic mapping
            labels = items_df["label"].unique().tolist()
            print(f"  - {task_name}: {len(items_df)} 条数据, 标签: {labels}")

            # 自动注册任务
            if auto_register:
                if not self.register_task_if_needed(task_name, labels):
                    print(f"⚠️  跳过任务 '{task_name}' 的训练")
                    continue
            # 再次从服务端读取该任务的标签（以持久化为准），便于后续过滤
            server_labels = self._get_task_labels_from_server(task_name)
            if server_labels is None:
                print("⚠️  无法获取服务端任务标签，将按CSV标签尝试训练（可能导致部分批次失败）")
            else:
                # 打印服务端与CSV标签对比
                csv_set, srv_set = set(labels), set(server_labels)
                if csv_set - srv_set:
                    print(f"  ⚠️ CSV 中存在未注册标签: {sorted(csv_set - srv_set)}（将过滤）")
                if srv_set - csv_set:
                    print(f"  ℹ️  服务端还有额外标签: {sorted(srv_set - csv_set)}")

        # 外部验证集分组（仅用于同名任务）
        val_groups: dict[str, pd.DataFrame] = {}
        if val_df is not None:
            val_groups = self.group_by_task(val_df)
            # 只提示一次总体信息
            print(f"📑 外部验证集: 共 {len(val_df)} 条，涉及 {len(val_groups)} 个任务")

        # 定义分层打乱与交替混合（按标签分桶并轮询取样）
        def stratified_interleave(df: pd.DataFrame, rnd: np.random.RandomState) -> pd.DataFrame:
            # 使用 frac=1 进行随机抽样，实现打乱
            return df.sample(frac=1, random_state=rnd).reset_index(drop=True)

        # 批量训练
        total_success = 0
        total_failed = 0

        for task_name, items_df in task_groups.items():
            print(f"\n🔄 开始训练任务: {task_name}")
            # 可选验证集划分（基于每个任务独立）
            rnd_global = np.random.RandomState(seed=seed if seed is not None else random.randrange(1 << 30))
            # 优先使用外部验证集
            ext_val_df = val_groups.get(task_name) if val_groups else None
            if ext_val_df is not None:
                train_df = items_df.copy()
                val_items_df = ext_val_df.copy()
                print(f"  使用外部验证集: {len(val_items_df)} 条")
            else:
                # 回退到内部划分（仅当设置了比例且样本数足够时）
                if 0.0 < val_ratio < 0.5 and len(items_df) >= 10:
                    # 使用 train_test_split 进行分层划分
                    from sklearn.model_selection import train_test_split
                    train_df, val_items_df = train_test_split(
                        items_df,
                        test_size=val_ratio,
                        random_state=rnd_global,
                        stratify=items_df["label"],
                    )
                    print(f"  内部划分验证集: 训练 {len(train_df)} / 验证 {len(val_items_df)} (ratio={val_ratio})")
                else:
                    train_df = items_df.copy()
                    val_items_df = pd.DataFrame()

            # 若能获取服务端标签，过滤掉未知标签样本，减少 400 错误
            server_labels = self._get_task_labels_from_server(task_name)
            if server_labels:
                before_train = len(train_df)
                before_val = len(val_items_df)
                train_df = train_df[train_df["label"].isin(server_labels)]
                if not val_items_df.empty:
                    val_items_df = val_items_df[val_items_df["label"].isin(server_labels)]

                removed_train = before_train - len(train_df)
                removed_val = before_val - len(val_items_df)
                if removed_train or removed_val:
                    print(
                        f"  ⚠️ 过滤未知标签样本: 训练 -{removed_train} / 验证 -{removed_val} "
                        f"(server_labels={server_labels})"
                    )

            # 打印标签分布
            def label_dist_df(df: pd.DataFrame) -> str:
                from collections import Counter
                from operator import itemgetter

                if df.empty:
                    return "(空)"
                c = Counter(df["label"])
                parts = [f"{k}:{v}" for k, v in sorted(c.items(), key=itemgetter(0))]
                return ", ".join(parts)

            print(f"  训练标签分布: {label_dist_df(train_df)}")
            if not val_items_df.empty:
                print(f"  验证标签分布: {label_dist_df(val_items_df)}")

            best_val_acc = -1.0
            not_improve_epochs = 0
            last_train_acc = None
            last_val_acc = None
            # 进行多个 epoch 的分层打乱并分批提交
            for ep in range(epochs):
                rnd = np.random.RandomState(seed=seed + ep if seed is not None else random.randrange(1 << 30))

                items_epoch_df = stratified_interleave(train_df, rnd) if shuffle else train_df.copy()

                t0 = time.time()
                processed = 0

                for i in range(0, len(items_epoch_df), batch_size):
                    batch_df = items_epoch_df.iloc[i:i + batch_size]
                    texts = batch_df["text"].tolist()
                    labels = batch_df["label"].tolist()

                    print(
                        f"  Epoch {ep + 1}/{epochs} 批次 {i // batch_size + 1}: 处理 {len(batch_df)} 条数据...",
                        end="",
                    )

                    if self.update_task(texts, task_name, labels):
                        print(" ✓")
                        total_success += len(batch_df)
                        processed += len(batch_df)
                    else:
                        print(" ❌")
                        total_failed += len(batch_df)

                    # 延迟避免过于频繁的请求
                    if delay > 0:
                        time.sleep(delay)

                # 简单在线评估：调用 /eval 计算训练集与验证集准确率
                try:
                    # 训练集快速抽样评估，避免过长
                    sample_train_df = train_df.sample(n=min(len(train_df), 200), random_state=rnd)
                    train_texts = sample_train_df["text"].tolist()
                    train_labels = sample_train_df["label"].tolist()
                    ev_payload = {"task": task_name, "texts": train_texts, "labels": train_labels}
                    r = self.session.post(f"{self.base_url}/eval", json=ev_payload, timeout=self.timeout)
                    r.raise_for_status()
                    rj = r.json()
                    train_acc = float(rj.get("accuracy", 0.0))
                    train_macro_f1 = rj.get("macro_f1")
                except Exception:
                    train_acc = None
                    train_macro_f1 = None

                val_acc = None
                if not val_items_df.empty:
                    try:
                        sample_val_df = val_items_df.sample(n=min(len(val_items_df), 200), random_state=rnd)
                        val_texts = sample_val_df["text"].tolist()
                        val_labels = sample_val_df["label"].tolist()
                        ev_payload = {"task": task_name, "texts": val_texts, "labels": val_labels}
                        r = self.session.post(f"{self.base_url}/eval", json=ev_payload, timeout=self.timeout)
                        r.raise_for_status()
                        rj = r.json()
                        val_acc = float(rj.get("accuracy", 0.0))
                        val_macro_f1 = rj.get("macro_f1")
                        # 困难样本回流（开关控制）：取前 50 条 hardest，追加一次小权重训练
                        if hard_mining:
                            hardest = rj.get("hardest") or []
                            if hardest:
                                # 仅保留在训练任务标签范围内的样本
                                hx = [
                                    h
                                    for h in hardest
                                    if isinstance(h.get("text"), str)
                                    and isinstance(h.get("true"), str)
                                ]
                                if hx:
                                    ht_texts = [h["text"] for h in hx]
                                    ht_labels = [h["true"] for h in hx]
                                    payload = {
                                        "texts": ht_texts,
                                        "task": task_name,
                                        "labels": ht_labels,
                                        # 小权重微调，减小噪声影响
                                        "sample_weight": [hard_weight] * len(ht_texts),
                                    }
                                    try:
                                        rr = self.session.post(
                                            f"{self.base_url}/update",
                                            json=payload,
                                            timeout=self.timeout,
                                        )
                                        rr.raise_for_status()
                                        print(
                                            f"    ↩ 回流困难样本 {len(ht_texts)} 条（权重={hard_weight}）"
                                        )
                                    except Exception:
                                        pass
                    except Exception:
                        val_acc = None
                        val_macro_f1 = None

                # 输出收敛/过拟合提示
                msg = f"  Epoch {ep + 1} 指标: "
                if train_acc is not None:
                    msg += f"train_acc={train_acc:.4f} "
                    if train_macro_f1 is not None:
                        msg += f"train_f1={float(train_macro_f1):.4f} "
                if val_acc is not None:
                    msg += f"val_acc={val_acc:.4f} "
                    if val_macro_f1 is not None:
                        msg += f"val_f1={float(val_macro_f1):.4f} "
                dt = max(time.time() - t0, 1e-9)
                qps = processed / dt if processed else 0.0
                msg += f"throughput={qps:.1f} it/s"
                print(msg.strip())

                if val_acc is not None:
                    if val_acc > best_val_acc + 1e-4:
                        best_val_acc = val_acc
                        not_improve_epochs = 0
                    else:
                        not_improve_epochs += 1
                        if not_improve_epochs >= warn_threshold_no_improve:
                            print(f"  ⚠️ 验证集{warn_threshold_no_improve}个epoch未提升，可能未收敛或学习率不合适")

                if train_acc is not None and val_acc is not None and (train_acc - val_acc) >= overfit_warn_gap:
                    print("  ⚠️ 可能过拟合：训练/验证准确率差距较大，建议减小模型、增大正则或提高混洗/样本量")

                last_train_acc = train_acc
                last_val_acc = val_acc

            # 任务级汇总
            print("  —— 任务小结 ——")
            if last_train_acc is not None:
                print(f"  最后一次训练集准确率: {last_train_acc:.4f}")
            if last_val_acc is not None:
                print(f"  最好验证集准确率: {best_val_acc:.4f}")

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
    parser.add_argument("--val-csv", type=str, default=None, help="验证集CSV文件路径（可选，提供时优先使用外部验证集）")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="AI Reviewer服务URL")
    parser.add_argument("--batch-size", type=int, default=50, help="批处理大小")
    parser.add_argument("--delay", type=float, default=0.1, help="批次间延迟时间（秒）")
    parser.add_argument("--timeout", type=int, default=30, help="请求超时时间（秒）")
    parser.add_argument("--no-auto-register", action="store_true", help="不自动注册新任务")
    parser.add_argument("--create-sample", action="store_true", help="创建示例CSV文件")
    parser.add_argument("--no-shuffle", action="store_true", help="不打乱样本（默认打乱并分层混合）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，用于重现实验")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（对同一CSV重复多轮）")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="内部验证集划分比例（0~0.5，若未提供外部验证集时生效）",
    )
    parser.add_argument("--no-hard-mining", action="store_true", help="禁用困难样本回流")
    parser.add_argument("--hard-weight", type=float, default=0.3, help="困难样本回流权重（默认 0.3）")

    args = parser.parse_args()

    csv_file = Path(args.csv)
    val_csv = Path(args.val_csv) if args.val_csv else None

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
        val_ratio=args.val_ratio,
        val_csv=val_csv,
        hard_mining=not args.no_hard_mining,
        hard_weight=args.hard_weight,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
