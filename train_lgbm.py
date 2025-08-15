"""
使用 LightGBM 模型进行离线训练的脚本。

该脚本从 CSV 文件加载数据，为指定任务训练一个 LightGBM 分类器，
并将训练好的模型保存到磁盘。这个过程适用于需要更高性能、
不依赖在线学习的场景。

用法:
  python train_lgbm.py --csv training_data.csv --task "我的LGBM任务"
  python train_lgbm.py --csv data.csv --task "新任务" --params-file lgbm_params.json
"""
import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ 警告: optuna 未安装，无法使用贝叶斯优化功能。可通过 'pip install optuna' 安装。")

from ai_reviewer.config import AppConfig, load_config
from ai_reviewer.embeddings import EmbeddingBackend


def optimize_lgbm_params(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    n_classes: int,
    n_trials: int = 100,
    random_state: int = 42,
) -> dict:
    """
    使用 Optuna 进行 LightGBM 超参数优化
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna 未安装，无法使用贝叶斯优化功能")

    def objective(trial):
        # 定义参数搜索空间
        params = {
            "objective": "binary" if n_classes == 2 else "multiclass",
            "metric": "binary_logloss" if n_classes == 2 else "multi_logloss",
            "boosting_type": "gbdt",
            "seed": random_state,
            "n_jobs": -1,
            "verbose": -1,
            "force_col_wise": True,
            "use_gpu": "cuda",
        }

        if n_classes > 2:
            params["num_class"] = n_classes

        # 超参数搜索空间
        params.update({
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        })

        # 训练模型
        model = lgb.LGBMClassifier(**params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric=params["metric"],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        # 评估模型（使用 F1 作为优化目标）
        y_pred = np.array(model.predict(x_val))
        if n_classes == 2:
            score = f1_score(y_val, y_pred)
        else:
            score = f1_score(y_val, y_pred, average="macro")

        return float(score)

    print(f"🔍 开始贝叶斯优化，将进行 {n_trials} 次试验...")

    # 创建 Optuna 研究
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )

    # 执行优化
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"✅ 优化完成！最佳 F1 分数: {study.best_value:.4f}")
    print(f"🏆 最佳参数: {study.best_params}")

    # 构建最终参数
    best_params = {
        "objective": "binary" if n_classes == 2 else "multiclass",
        "metric": "binary_logloss" if n_classes == 2 else "multi_logloss",
        "boosting_type": "gbdt",
        "seed": random_state,
        "n_jobs": -1,
        "verbose": -1,
        "force_col_wise": True,
        "use_gpu": "cuda",
    }

    if n_classes > 2:
        best_params["num_class"] = n_classes

    best_params.update(study.best_params)

    return best_params


def get_default_params(n_samples: int, n_classes: int, random_state: int) -> dict:
    """根据数据规模获取默认参数"""
    base_params = {
        "boosting_type": "gbdt",
        "seed": random_state,
        "n_jobs": -1,
        "verbose": -1,
        "force_col_wise": True,
    }
    if n_classes == 2:
        base_params.update({
            "objective": "binary",
            "metric": "binary_logloss",
        })
    else:
        base_params.update({
            "objective": "multiclass",
            "num_class": n_classes,
            "metric": "multi_logloss",
        })
    # 根据样本量调整参数
    if n_samples < 1000:
        # 小数据集：防止过拟合
        base_params.update({
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 10,
            "reg_alpha": 0.2,
            "reg_lambda": 0.2,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
        })
    elif n_samples < 10000:
        # 中等数据集：平衡性能和效率
        base_params.update({
            "n_estimators": 500,
            "learning_rate": 0.1,
            "num_leaves": 64,
            "max_depth": 8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
        })
    else:
        # 大数据集：更复杂的模型
        base_params.update({
            "n_estimators": 1000,
            "learning_rate": 0.1,
            "num_leaves": 128,
            "max_depth": 10,
            "min_child_samples": 50,
            "reg_alpha": 0.05,
            "reg_lambda": 0.05,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 10,
        })
    return base_params


def train_lightgbm_task(
    csv_path: Path,
    task_name: str,
    config: AppConfig,
    val_csv_path: Path | None = None,
    params: dict | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    use_bayesian_optimization: bool = False,
    n_trials: int = 100,
):
    """
    为指定任务训练 LightGBM 模型。
    """
    print(f"🚀 开始为任务 '{task_name}' 训练 LightGBM 模型...")

    # 1. 加载并打乱数据
    try:
        df = pd.read_csv(csv_path)
        task_df = df[df["task"] == task_name]
        if task_df.empty:
            print(f"❌ 错误: 在 '{csv_path}' 中找不到任务 '{task_name}' 的数据。")
            return

        # 对数据进行随机打乱，以防原始数据按标签排序
        print("🔀 正在对数据进行随机打乱...")
        task_df = task_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        texts = task_df["text"].tolist()
        labels = task_df["label"].tolist()
        print(f"📚 找到 {len(texts)} 条关于任务 '{task_name}' 的数据。")
    except Exception as e:
        print(f"❌ 读取或处理 CSV 文件时出错: {e}")
        return

    # 2. 获取任务配置和标签映射
    task_config = config.tasks.get(task_name)
    if not task_config:
        print(f"❌ 错误: 任务 '{task_name}' 未在配置文件中定义。")
        print("请先在 config.toml 中添加该任务的配置。")
        return

    label_map = {label: i for i, label in enumerate(task_config.labels)}
    y = np.array([label_map.get(label) for label in labels])
    if None in y:
        print("❌ 错误: 数据中的某些标签未在任务配置中定义。")
        return

    # 3. 编码文本为向量
    print("🧠 正在将文本编码为向量...")
    embedder = EmbeddingBackend(config.embedding_model, config.device, config.embed_batch_size, config.preprocess)
    x_train_vec = embedder.encode(texts)
    y_train = np.array([label_map.get(label) for label in labels])
    print("✅ 训练数据编码完成。")

    # 将向量转换为带有特征名的 DataFrame 以避免警告
    feature_names = [f"emb_{i}" for i in range(x_train_vec.shape[1])]
    x_train_df = pd.DataFrame(x_train_vec, columns=feature_names)

    # 4. 准备验证集
    x_val_df = None
    y_val = None
    if val_csv_path:
        print(f"📖 正在从 '{val_csv_path}' 加载外部验证集...")
        try:
            val_df = pd.read_csv(val_csv_path)
            val_task_df = val_df[val_df["task"] == task_name]
            if val_task_df.empty:
                print(f"⚠️ 警告: 在验证文件 '{val_csv_path}' 中找不到任务 '{task_name}' 的数据。将继续但无验证。")
            else:
                val_texts = val_task_df["text"].tolist()
                val_labels = val_task_df["label"].tolist()
                x_val_vec = embedder.encode(val_texts)
                y_val = np.array([label_map.get(label) for label in val_labels])
                x_val_df = pd.DataFrame(x_val_vec, columns=feature_names)
                print(f"✅ 找到 {len(x_val_df)} 条验证数据。")
        except Exception as e:
            print(f"❌ 读取或处理验证 CSV 时出错: {e}")
            return
    else:
        print("🔪 正在从训练数据中划分验证集...")
        x_train_df, x_val_df, y_train, y_val = train_test_split(
            x_train_df, y_train, test_size=test_size, random_state=random_state, stratify=y_train
        )
        print(f"🔪 数据集划分: {len(x_train_df)} 训练, {len(x_val_df)} 验证。")

    # 5. 训练 LightGBM 模型
    print("🏋️ 开始训练 LightGBM 模型...")
    n_samples = len(x_train_df)
    n_classes = len(task_config.labels)

    # 如果启用贝叶斯优化且有验证集
    if use_bayesian_optimization and x_val_df is not None and y_val is not None and OPTUNA_AVAILABLE:
        print("🔍 使用贝叶斯优化进行超参数调优...")
        lgbm_params = optimize_lgbm_params(
            x_train_df, y_train, x_val_df, y_val, n_classes, n_trials, random_state
        )
    elif use_bayesian_optimization and not OPTUNA_AVAILABLE:
        print("⚠️ 无法使用贝叶斯优化：optuna 未安装，回退到默认参数")
        lgbm_params = params or get_default_params(n_samples, n_classes, random_state)
    elif use_bayesian_optimization and (x_val_df is None or y_val is None):
        print("⚠️ 无法使用贝叶斯优化：需要验证集，回退到默认参数")
        lgbm_params = params or get_default_params(n_samples, n_classes, random_state)
    else:
        # 使用默认参数或用户提供的参数
        lgbm_params = params or get_default_params(n_samples, n_classes, random_state)

    print(f"🔧 使用以下 LightGBM 参数: {lgbm_params}")

    model = lgb.LGBMClassifier(**lgbm_params)
    eval_set = [(x_val_df, y_val)] if x_val_df is not None and y_val is not None else None

    model.fit(
        x_train_df,
        y_train,
        eval_set=eval_set,
        eval_metric=lgbm_params.get("metric"),
        # callbacks=[lgb.early_stopping(100, verbose=True)],
    )
    print("✅ 模型训练完成。")

    # 6. 在验证集上评估并保存结果
    eval_results = {}
    if x_val_df is not None and y_val is not None:
        print("📊 正在验证集上评估模型...")
        y_pred = np.array(model.predict(x_val_df))
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average="macro")
        recall = recall_score(y_val, y_pred, average="macro")
        f1 = f1_score(y_val, y_pred, average="macro")
        eval_results = {
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
        }
        print(f"📈 验证结果: Accuracy={accuracy:.4f}, Macro-F1={f1:.4f}")

    # 7. 保存模型和元数据
    model_path = Path(task_config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"💾 模型已保存到: {model_path}")

    meta_path = model_path.with_suffix(model_path.suffix + ".labels.json")
    meta_data = {
        "labels": task_config.labels,
        "classifier": "lightgbm",
        "source_csv": str(csv_path),
        "val_source_csv": str(val_csv_path) if val_csv_path else "from_training_set",
        "eval_results": eval_results,
    }
    if task_config.threshold is not None:
        meta_data["threshold"] = task_config.threshold
    if task_config.temperature is not None:
        meta_data["temperature"] = task_config.temperature

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)
    print(f"📄 元数据已保存到: {meta_path}")
    print("\n🎉 训练成功完成！")


def main():
    parser = argparse.ArgumentParser(description="为 AI Reviewer 训练 LightGBM 模型")
    parser.add_argument("--csv", type=str, required=True, help="包含 'text', 'task', 'label' 列的 CSV 文件路径")
    parser.add_argument("--val-csv", type=str, default=None, help="可选的外部验证集 CSV 文件路径")
    parser.add_argument("--task", type=str, required=True, help="要训练的任务名称 (必须在 config.toml 中定义)")
    parser.add_argument("--config", type=str, default=None, help="AI Reviewer 配置文件路径 (默认: config.toml)")
    parser.add_argument("--params-file", type=str, default=None, help="包含 LightGBM 参数的 JSON 文件路径 (可选)")
    parser.add_argument("--test-size", type=float, default=0.2, help="验证集所占比例")
    parser.add_argument("--seed", type=int, default=42, help="用于数据划分和模型训练的随机种子")
    parser.add_argument("--bayesian-optimization", action="store_true", help="启用贝叶斯优化进行超参数调优")
    parser.add_argument("--n-trials", type=int, default=100, help="贝叶斯优化的试验次数")

    args = parser.parse_args()

    config = load_config(args.config)
    lgbm_params = None
    if args.params_file:
        try:
            params_path = Path(args.params_file)
            with params_path.open(encoding="utf-8") as f:
                lgbm_params = json.load(f)
            print(f"🔧 已从 '{args.params_file}' 加载 LightGBM 参数。")
        except Exception as e:
            print(f"⚠️ 无法加载参数文件: {e}。将使用默认参数。")

    train_lightgbm_task(
        csv_path=Path(args.csv),
        task_name=args.task,
        config=config,
        val_csv_path=Path(args.val_csv) if args.val_csv else None,
        params=lgbm_params,
        test_size=args.test_size,
        random_state=args.seed,
        use_bayesian_optimization=args.bayesian_optimization,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
