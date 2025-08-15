<div align="center">

# AI Reviewer

_一个可扩展的文本分类服务，基于向量嵌入 + 线性分类器，支持多任务检测与在线增量学习。_

</div>

~~纯纯的氛围编程（）~~

## 功能特性

- 默认内容审查任务（二分类）：违规内容
- 动态注册自定义任务与标签，在线增量学习（/update）
- 文本预处理与向量池化（支持自定义模型、批大小、是否启用预处理）
- 二分类支持阈值（threshold），多分类支持温度缩放（temperature）
- 评测增强：macro 指标与困难样本（hardest）输出；支持困难样本回流训练
- 简易自动校准：阈值/温度一键搜索并写回

## 项目结构

- `ai_reviewer/`
  - `api.py`：FastAPI 路由与应用工厂
  - `config.py`：配置加载与数据类
  - `embeddings.py`：Embedding 抽象
  - `tasks.py`：任务注册与模型持久化
- `config.toml`：配置文件
- `main.py`：应用入口
- `batch_train.py`：批量训练脚本
- `eval_csv.py`：离线评测与自动校准脚本

## 快速开始

1. 安装依赖

    ```bash
    uv sync --extra cpu
    ```

    或使用

    ```bash
    uv sync --extra gpu
    ```

    以启用 GPU 支持

2. 运行服务

    ```bash
    uv run python main.py
    # 或
    uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```

可使用环境变量 `AI_REVIEWER_CONFIG` 指定自定义 TOML 配置文件路径。

## 配置说明

```toml
embedding_model = "richinfoai/ritrieve_zh_v1"  # 可替换为其他中文句向量模型
device = "auto"
embed_batch_size = 32         # 可选：向量batch_size
preprocess = true             # 可选：是否启用文本预处理

# 中文任务名请使用双引号
[tasks."违规内容"]
# 可添加多个标签
labels = ["否", "是"]
# 任务分类器权重文件路径
model_path = "models/违规内容.joblib"

```

## 批量训练脚本

### 使用方法

1. 创建示例 CSV 文件

```bash
uv run python batch_train.py --csv sample_data.csv --create-sample
```

CSV文件格式：

| 列 | 说明 | 示例 |
|-----|------|------|
| text | 要分类的文本内容 | "难道说？" |
| task | 分类任务名称 | "违规内容" |
| label | 对应的标签 | "否" |

2. 运行批量训练

```bash
uv run python batch_train.py --csv training_data.csv
```

3. 常用参数

```bash
uv run python batch_train.py \
    --csv training_data.csv \
    --url http://localhost:8000 \
    --batch-size 50 \
    --val-csv eval_data.csv \
    --val-ratio 0.2 \
    --epochs 3 \
    --hard-mining --hard-weight 0.3
```

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--csv` | 必需 | CSV 训练数据文件路径 |
| `--url` | `http://localhost:8000` | AI Reviewer 服务 URL |
| `--batch-size` | `50` | 每批处理的数据条数 |
| `--delay` | `0.1` | 批次间延迟时间（秒） |
| `--timeout` | `30` | 请求超时时间（秒） |
| `--no-auto-register` | `False` | 不自动注册新任务 |
| `--create-sample` | `False` | 创建示例 CSV 文件 |
| `--val-csv` |  | 外部验证集 CSV |
| `--val-ratio` | `0.0` | 内部分割比例（无外部验证集时生效） |
| `--epochs` | `1` | 训练轮数 |
| `--no-hard-mining` | `False` | 关闭困难样本回流 |
| `--hard-weight` | `0.3` | 回流样本权重 |

## 评测脚本

### 使用方法

1. 准备评测数据，格式与训练数据相同

2. 运行评测脚本

  ```bash
  uv run python eval_csv.py --csv eval_data.csv
  ```

3. 自动校准（可选）

  - 二分类阈值（F1 目标）
  ```bash
  uv run python eval_csv.py --csv eval_data.csv --auto-calibrate --calib-target f1
  ```
  - 多分类温度（NLL 目标）
  ```bash
  uv run python eval_csv.py --csv eval_data.csv --auto-calibrate --temp-grid 0.5:2.0:0.1
  ```

4. 输出示例

- 每个任务：样本数、准确率、各标签支持度与准确率。
- 总体：宏观样本数与加权准确率。
- 若开启 `--fail-under` 且总体准确率低于阈值，会以 1 退出。

> 任务: 违规内容
>   样本数: 100
>   准确率: 0.7000
>   分标签:
>     - 否: acc=0.7400, support=50
>     - 是: acc=0.6600, support=50
> 
> ====================
> 总体: 样本数=100, 准确率=0.7000

## API

### `GET /config`

返回当前配置与已注册任务

**返回示例：**

```json
{
  "embedding_model": "richinfoai/ritrieve_zh_v1",
  "device": "cuda:0",
  "tasks": {
    "违规内容": {
      "labels": ["否", "是"],
      "model_path": "models/违规内容.joblib",
      "threshold": 0.50,
      "temperature": 1.00
    },
    "AI生成": {
      "labels": ["否", "是"],
      "model_path": "models/AI生成.joblib",
      "threshold": 0.55,
      "temperature": 1.00
    }
  }
}
```

### `GET /tasks`

列出任务名称

**返回示例：**

```json
{
  "tasks": ["违规内容", "AI生成"]
}
```

### `POST /tasks/register`

注册/创建自定义任务

**请求示例：**

```json
{
  "name": "我的任务",
  "labels": ["A", "B"],
  "model_path": "models/my.joblib"
}
```

**返回示例：**

```json
{
  "message": "task registered",
  "task": {
    "name": "我的任务",
    "labels": ["A", "B"],
    "model_path": "models/my.joblib"
  }
}
```

### `POST /predict`

预测文本分类

**请求示例：**

```json
{
  "texts": ["难道说？", "我是AI"],
  "tasks": ["违规内容", "AI生成"]
}
```

> `tasks` 为空时执行所有任务

**返回示例：**

```json
{
  "results": [
    {
      "违规内容": {
        "label": "是",
        "confidence": 0.85
      },
      "AI生成": {
        "label": "否",
        "confidence": 0.72
      }
    },
    {
      "违规内容": {
        "label": "否",
        "confidence": 0.91
      },
      "AI生成": {
        "label": "是",
        "confidence": 0.88
      }
    }
  ]
}
```

### `POST /update`

增量学习更新指定任务

**请求示例：**

```json
{
  "texts": ["我是一条违规内容"],
  "task": "违规内容",
  "labels": ["是"]
}
```

**返回示例：**

```json
{
  "message": "task 违规内容 updated"
}
```

### `POST /eval`

按任务评测，返回 accuracy、per_label，以及 macro 指标与困难样本（hardest）。

### `POST /probs`

返回基础概率（未应用温度与阈值），用于外部校准。

### `POST /tasks/calibrate`

写回任务的 `threshold` 与 `temperature`（可二选一）。

## 注意事项

- 首次注册任务时，会用任务名+标签的模板文本进行最小化冷启动，以便分类器具备类别先验。需要使用真实样本调用 `/update` 强化训练后，才能实现实际的分类效果。
- 完成基础训练后，使用评测脚本 `--auto-calibrate` 进行阈值/温度校准，再根据 hardest 做少量困难样本回流，可获得更稳的提升。
