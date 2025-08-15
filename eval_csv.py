"""
评测脚本：读取与训练相同格式的 CSV（text,task,label），
按任务分组后调用服务端 /eval 计算任务与总体准确率。

用法示例：
  python eval_csv.py --csv test_data.csv
  python eval_csv.py --csv test_data.csv --url http://localhost:8000 --timeout 60 --fail-under 0.8
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
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

    def get_config(self) -> dict:
        try:
            r = self.session.get(f"{self.base_url}/config", timeout=self.timeout)
            r.raise_for_status()
            return r.json() or {}
        except Exception:
            return {}

    def get_task_labels(self, task: str) -> list[str] | None:
        cfg = self.get_config()
        tasks = (cfg or {}).get("tasks") or {}
        info = tasks.get(task)
        if isinstance(info, dict):
            lbs = info.get("labels")
            if isinstance(lbs, list) and all(isinstance(x, str) for x in lbs):
                return lbs
        return None

    @staticmethod
    def load_csv(csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        df = pd.read_csv(csv_path)
        required = {"text", "task", "label"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"CSV文件缺少必需列: {missing}")
        # 清理数据
        df = df[list(required)].copy()
        df.dropna(inplace=True)
        for col in ["text", "task", "label"]:
            df[col] = df[col].astype(str).str.strip()
        df = df[df["text"].str.len() > 0]
        df = df[df["task"].str.len() > 0]
        df = df[df["label"].str.len() > 0]
        return df

    @staticmethod
    def group_by_task(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {str(task): group for task, group in df.groupby("task")}

    def eval_task(self, task: str, items_df: pd.DataFrame) -> TaskEvalResult | None:
        texts = items_df["text"].tolist()
        labels = items_df["label"].tolist()
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

    # ---- Calibration helpers ----
    def get_probs(self, task: str, texts: list[str]) -> list[list[float]] | None:
        payload = {"task": task, "texts": texts}
        try:
            r = self.session.post(f"{self.base_url}/probs", json=payload, timeout=self.timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json() or {}
            return data.get("probs")
        except Exception:
            return None

    @staticmethod
    def _grid(spec: str) -> list[float]:
        """Parse range spec 'start:stop:step' to a float list."""
        try:
            start_s, stop_s, step_s = spec.split(":")
            start, stop, step = float(start_s), float(stop_s), float(step_s)
            n = int(math.floor((stop - start) / step)) + 1
            return [start + i * step for i in range(max(n, 0))]
        except Exception:
            return []

    @staticmethod
    def _macro_f1(y_true: list[int], y_pred: list[int], n_classes: int) -> float:
        try:
            from sklearn.metrics import f1_score

            return float(f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0))
        except Exception:
            # 简易实现：按每类F1再平均（当 sklearn 不可用时退化）
            tp = [0] * n_classes
            fp = [0] * n_classes
            fn = [0] * n_classes
            for yt, yp in zip(y_true, y_pred, strict=True):
                if yt == yp:
                    tp[yt] += 1
                else:
                    fp[yp] += 1
                    fn[yt] += 1
            f1s = []
            for k in range(n_classes):
                p = tp[k] / (tp[k] + fp[k]) if (tp[k] + fp[k]) > 0 else 0.0
                r = tp[k] / (tp[k] + fn[k]) if (tp[k] + fn[k]) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                f1s.append(f1)
            return float(sum(f1s) / len(f1s)) if f1s else 0.0

    @staticmethod
    def _nll(y_true: list[int], probs: list[list[float]]) -> float:
        eps = 1e-12
        loss = 0.0
        for i, yt in enumerate(y_true):
            p = probs[i][yt] if 0 <= yt < len(probs[i]) else eps
            p = max(p, eps)
            loss -= math.log(p)
        return loss / max(1, len(y_true))

    @staticmethod
    def _apply_temp_to_probs(probs: list[list[float]], temperature: float) -> list[list[float]]:
        import numpy as np

        arr = np.array(probs, dtype=float)
        eps = 1e-12
        logits = np.log(np.maximum(arr, eps)) / float(temperature)
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        out = ex / ex.sum(axis=1, keepdims=True)
        return out.tolist()

    def calibrate_threshold(
        self, task: str, items_df: pd.DataFrame, grid: str, target: str
    ) -> tuple[float | None, float]:
        labels = self.get_task_labels(task) or []
        if len(labels) != 2:
            return None, 0.0
        texts = items_df["text"].tolist()
        y_true = [labels.index(label) for label in items_df["label"] if label in labels]
        probs = self.get_probs(task, texts)
        if probs is None:
            return None, 0.0
        pos_idx = 1  # 与服务端一致，第二个标签视为正类
        best_thr, best_score = None, -1.0
        for thr in self._grid(grid):
            y_pred = [1 if (row[pos_idx] if pos_idx < len(row) else 0.0) >= thr else 0 for row in probs]
            if target == "f1":
                score = self._macro_f1(y_true, y_pred, 2)
            else:  # fallback to accuracy
                score = sum(int(a == b) for a, b in zip(y_true, y_pred, strict=True)) / max(1, len(y_true))
            if score > best_score + 1e-12:
                best_score, best_thr = score, float(thr)
        return best_thr, best_score

    def calibrate_temperature(
        self, task: str, items_df: pd.DataFrame, grid: str, target: str
    ) -> tuple[float | None, float]:
        labels = self.get_task_labels(task) or []
        if len(labels) < 2:
            return None, 0.0
        texts = items_df["text"].tolist()
        y_true = [labels.index(label) for label in items_df["label"] if label in labels]
        base_probs = self.get_probs(task, texts)
        if base_probs is None:
            return None, 0.0
        best_temp, best_obj = None, float("inf") if target == "nll" else -1.0
        for t in self._grid(grid):
            if t <= 0:
                continue
            probs = self._apply_temp_to_probs(base_probs, t)
            if target == "nll":
                obj = self._nll(y_true, probs)
                better = obj < best_obj
            else:
                y_pred = [int(max(range(len(p)), key=lambda k: p[k])) for p in probs]
                obj = self._macro_f1(y_true, y_pred, len(labels))
                better = obj > best_obj
            if best_temp is None or better:
                best_temp, best_obj = float(t), obj
        return best_temp, best_obj

    def write_calibration(self, task: str, threshold: float | None, temperature: float | None) -> bool:
        payload: dict[str, object] = {"task": task}
        if threshold is not None:
            payload["threshold"] = float(threshold)
        if temperature is not None:
            payload["temperature"] = float(temperature)
        try:
            r = self.session.post(f"{self.base_url}/tasks/calibrate", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return True
        except Exception as e:
            print(f"⚠️ 写回校准失败: {e}")
            return False


def main() -> int:
    parser = argparse.ArgumentParser(description="评测 CSV 准确率（按任务分组，服务端 /eval 计算）")
    parser.add_argument("--csv", type=str, required=True, help="CSV文件路径（包含 text,task,label）")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="AI Reviewer 服务 URL")
    parser.add_argument("--timeout", type=int, default=30, help="请求超时时间（秒）")
    parser.add_argument("--fail-under", type=float, default=None, help="总体准确率低于该阈值则以非零码退出")
    parser.add_argument("--auto-calibrate", action="store_true", help="自动校准：二分类阈值、多分类温度")
    parser.add_argument(
        "--calib-target",
        choices=["f1", "nll"],
        default="nll",
        help="温度或阈值优化目标：二分类建议 f1，多分类默认 nll",
    )
    parser.add_argument("--th-grid", type=str, default="0.2:0.8:0.02", help="阈值搜索范围，格式 start:stop:step")
    parser.add_argument("--temp-grid", type=str, default="0.5:2.0:0.1", help="温度搜索范围，格式 start:stop:step")
    parser.add_argument("--no-write-back", action="store_true", help="只计算最优参数但不写回服务端")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    ev = Evaluator(base_url=args.url, timeout=args.timeout)

    if not ev.check_server():
        return 2

    try:
        df = ev.load_csv(csv_path)
    except Exception as e:
        print(f"❌ 读取CSV失败: {e}")
        return 2

    if df.empty:
        print("❌ CSV 中无有效数据")
        return 2

    groups = ev.group_by_task(df)
    if not groups:
        print("❌ 未找到任何任务数据")
        return 2

    print(f"📊 发现 {len(groups)} 个任务，开始评测…\n")

    results: list[TaskEvalResult] = []
    total = 0
    correct = 0.0

    for task, items_df in groups.items():
        res = ev.eval_task(task, items_df)
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

        # 可选：自动校准
        if args.auto_calibrate:
            labels = ev.get_task_labels(task) or []
            if len(labels) == 2:
                thr, score = ev.calibrate_threshold(task, items_df, args.th_grid, target="f1")
                if thr is not None:
                    print(f"  建议阈值: {thr:.2f}（基于F1）")
                    if not args.no_write_back:
                        ok = ev.write_calibration(task, threshold=thr, temperature=None)
                        print("  ↳ 阈值写回: ", "成功" if ok else "失败")
            elif len(labels) > 2:
                temp, obj = ev.calibrate_temperature(task, items_df, args.temp_grid, target=args.calib_target)
                if temp is not None:
                    if args.calib_target == "nll":
                        print(f"  建议温度: {temp:.2f}（NLL={obj:.4f}）")
                    else:
                        print(f"  建议温度: {temp:.2f}（macroF1={obj:.4f}）")
                    if not args.no_write_back:
                        ok = ev.write_calibration(task, threshold=None, temperature=temp)
                        print("  ↳ 温度写回: ", "成功" if ok else "失败")

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
