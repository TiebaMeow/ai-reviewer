<div align="center">

# AI Reviewer

_一个可扩展的文本分类服务，基于向量嵌入 + 线性分类器，支持多任务检测与在线增量学习。_

</div>

## 功能特性

- 四类默认内容审查任务（二分类）：滑坡、引战、拉踩、AI生成
- 动态注册自定义任务与标签
- 支持自定义配置（Embedding 模型、设备、任务与模型路径）

## 项目结构

- `ai_reviewer/`
  - `api.py`：FastAPI 路由与应用工厂
  - `config.py`：配置加载与数据类
  - `embeddings.py`：Embedding 抽象
  - `tasks.py`：任务注册与模型持久化
- `config.toml`：配置文件
- `main.py`：应用入口

## 快速开始

1. 安装依赖

```bash
uv sync
```

2. 运行服务

```bash
uv run python main.py
# 或
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

可使用环境变量 `AI_REVIEWER_CONFIG` 指定自定义 TOML 配置文件路径。

## 配置说明

```toml
embedding_model = "Qwen/Qwen3-Embedding-0.6B"
device = "auto"

# 中文任务名请使用双引号
[tasks."滑坡"]
# 可添加多个标签
labels = ["无", "有"]
# 任务分类器权重文件路径
model_path = "models/slippery_slope.joblib"

```

## 批量训练脚本

### 使用方法

1. 创建示例 CSV 文件

```bash
python batch_train.py --csv sample_data.csv --create-sample
```

2. 运行批量训练

```bash
python batch_train.py --csv training_data.csv
```

3. 自定义参数

```bash
python batch_train.py \
    --csv training_data.csv \
    --url http://localhost:8000 \
    --batch-size 20 \
    --delay 0.2 \
    --timeout 60
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

### CSV文件格式

| 列 | 说明 | 示例 |
|-----|------|------|
| text | 要分类的文本内容 | "难道说？" |
| task | 分类任务名称 | "滑坡" |
| label | 对应的标签 | "无" |

## API

### `GET /config`

返回当前配置与已注册任务

**返回示例：**

```json
{
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "device": "cuda:0",
  "tasks": {
    "滑坡": {
      "labels": ["无", "有"],
      "model_path": "models/slippery_slope.joblib"
    },
    "引战": {
      "labels": ["否", "是"],
      "model_path": "models/inflammatory.joblib"
    }
  }
}
```

### `GET /tasks`

列出任务名称

**返回示例：**

```json
{
  "tasks": ["滑坡", "引战", "拉踩", "AI生成"]
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
  "texts": ["这个观点可能会导致严重后果", "这是一个正常的讨论"],
  "tasks": ["滑坡", "引战"]
}
```

> `tasks` 为空时执行所有任务

**返回示例：**

```json
{
  "results": [
    {
      "滑坡": {
        "label": "有",
        "confidence": 0.85
      },
      "引战": {
        "label": "无",
        "confidence": 0.72
      }
    },
    {
      "滑坡": {
        "label": "无",
        "confidence": 0.91
      },
      "引战": {
        "label": "无",
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
  "texts": ["这种说法明显是滑坡"],
  "task": "滑坡",
  "labels": ["有"]
}
```

**返回示例：**

```json
{
  "message": "task 滑坡 updated"
}
```

## 批量训练

项目提供了 `batch_train.py` 脚本，支持从 CSV 文件批量导入训练数据：

### 创建示例训练数据

```bash
uv run python batch_train.py --csv sample_data.csv --create-sample
```

### 批量训练使用

```bash
# 基本用法
uv run python batch_train.py --csv training_data.csv

# 自定义配置
uv run python batch_train.py \
    --csv training_data.csv \
    --batch-size 20 \
    --delay 0.2 \
    --url http://localhost:8000
```

### CSV 文件格式

```csv
text,task,label
这种观点会导致社会分裂,滑坡,有
我认为这个政策很好,滑坡,无
你这种想法就是想挑起争端,引战,是
让我们理性讨论这个问题,引战,否
```

详细使用说明请参考 [BATCH_TRAINING_GUIDE.md](BATCH_TRAINING_GUIDE.md)

## 注意事项

- 首次注册任务时，会用任务名+标签的模板文本进行最小化冷启动，以便分类器具备类别先验。需要使用真实样本调用 `/update` 强化训练后，才能实现实际的分类效果。
