# CachedRequestsTool 使用说明

## 概述

`CachedRequestsTool` 是一个为 `eval-anything` 框架设计的 API 请求缓存工具。它能够自动缓存 API 响应，避免重复的网络请求，提高评估效率并降低 API 调用成本。

## 安装要求

```bash
pip install requests
```

该工具依赖于 `eval-anything` 框架的以下组件：
- `eval_anything.evaluate_tools.base_tools.BaseTool`
- `eval_anything.utils.uuid.UUIDGenerator`
- `eval_anything.utils.logger.EvalLogger`

## 使用方法

### 1. 作为工具类使用（推荐）

```python
from eval_anything.evaluate_tools.cached_requests import CachedRequestsTool

# 创建工具实例
tool = CachedRequestsTool(
    cache_dir="./my_cache",  # 缓存目录
    max_try=3,               # 最大重试次数
    timeout=30               # 请求超时时间（秒）
)

# 准备消息
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
]

# 发起请求
response = tool.apply(
    messages=messages,
    model="gpt-3.5-turbo",
    max_completion_tokens=100,
    temperature=0.7,
    api_key="your-api-key",
    api_base="https://api.openai.com/v1/chat/completions"
)

print(response)
```

### 2. 作为便利函数使用

```python
from eval_anything.evaluate_tools.cached_requests import cached_requests

# 直接调用函数
response = cached_requests(
    messages=[{"role": "user", "content": "What is 2+2?"}],
    model="gpt-3.5-turbo",
    max_completion_tokens=50,
    temperature=0.7,
    api_key="your-api-key",
    api_base="https://api.openai.com/v1/chat/completions",
    cache_dir="./cache",
    max_try=3,
    timeout=30
)

print(response)
```

## 配置参数

### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cache_dir` | str | `"./cache"` | 缓存文件存储目录 |
| `max_try` | int | `3` | API 请求失败时的最大重试次数 |
| `timeout` | int | `3600` | 单次请求的超时时间（秒） |

### API 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `messages` | List[Dict] | 必需 | 发送给 API 的消息列表 |
| `model` | str | 必需 | 使用的模型名称 |
| `max_completion_tokens` | int | `4096` | 生成的最大 token 数 |
| `temperature` | float | `0.7` | 采样温度（0-2） |
| `repetition_penalty` | float | `1.0` | 重复惩罚因子 |
| `top_p` | float | `0.9` | 核采样参数 |
| `api_key` | str | None | API 密钥（可通过环境变量 `API_KEY` 设置） |
| `api_base` | str | None | API 端点 URL（可通过环境变量 `API_BASE` 设置） |

## 环境变量配置

你可以通过环境变量来设置默认的 API 配置：

```bash
export API_KEY="your-api-key-here"
export API_BASE="https://api.openai.com/v1/chat/completions"
```

或在 Python 中设置：

```python
import os
os.environ["API_KEY"] = "your-api-key-here"
os.environ["API_BASE"] = "https://api.openai.com/v1/chat/completions"
```

## 缓存机制

### 缓存键生成

缓存键基于以下参数生成：
- `messages`：输入消息
- `model`：模型名称
- `max_completion_tokens`：最大生成 token 数
- `temperature`：采样温度
- `repetition_penalty`：重复惩罚
- `top_p`：核采样参数

相同参数组合将生成相同的缓存键，从而复用之前的 API 响应。

### 缓存文件

- 缓存文件以 JSON 格式存储在指定目录
- 文件名为 SHA256 哈希值，确保唯一性
- 自动处理缓存文件的读写和错误恢复

## 完整示例

### 基础使用示例

```python
from eval_anything.evaluate_tools.cached_requests import CachedRequestsTool
import os

# 设置 API 配置
os.environ["API_KEY"] = "sk-your-api-key"
os.environ["API_BASE"] = "https://api.openai.com/v1/chat/completions"

# 创建工具
tool = CachedRequestsTool(cache_dir="./eval_cache")

# 测试消息
test_messages = [
    {"role": "system", "content": "你是一个有用的助手，请简洁回答问题。"},
    {"role": "user", "content": "请解释什么是机器学习？"}
]

# 第一次调用（会请求 API）
print("第一次调用...")
response1 = tool.apply(
    messages=test_messages,
    model="gpt-3.5-turbo",
    max_completion_tokens=200,
    temperature=0.7
)
print(f"响应: {response1}")

# 第二次调用（使用缓存）
print("\n第二次调用（应该使用缓存）...")
response2 = tool.apply(
    messages=test_messages,
    model="gpt-3.5-turbo",
    max_completion_tokens=200,
    temperature=0.7
)
print(f"响应: {response2}")
print(f"响应一致: {response1 == response2}")
```

### 批量处理示例

```python
from eval_anything.evaluate_tools.cached_requests import CachedRequestsTool

def batch_evaluate_questions(questions, model="gpt-3.5-turbo"):
    """批量评估问题列表"""
    tool = CachedRequestsTool(cache_dir="./batch_cache")

    results = []
    for i, question in enumerate(questions):
        print(f"处理问题 {i+1}/{len(questions)}: {question[:50]}...")

        messages = [
            {"role": "user", "content": question}
        ]

        try:
            response = tool.apply(
                messages=messages,
                model=model,
                max_completion_tokens=150,
                temperature=0.3  # 较低温度获得更一致的结果
            )
            results.append({
                "question": question,
                "answer": response,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "question": question,
                "answer": None,
                "status": "error",
                "error": str(e)
            })

    return results

# 使用示例
questions = [
    "什么是人工智能？",
    "Python 和 Java 的主要区别是什么？",
    "如何提高代码的可读性？"
]

results = batch_evaluate_questions(questions)
for result in results:
    print(f"问题: {result['question']}")
    print(f"答案: {result['answer']}")
    print(f"状态: {result['status']}")
    print("-" * 50)
```

## 错误处理

工具内置了多种错误处理机制：

1. **API 密钥验证**：缺少 API 密钥时抛出 `ValueError`
2. **网络错误重试**：自动重试失败的请求
3. **缓存文件恢复**：自动处理损坏的缓存文件
4. **超时保护**：防止请求长时间挂起

## 性能优化建议

1. **合理设置缓存目录**：使用 SSD 存储以提高缓存读写速度
2. **调整重试参数**：根据网络环境调整 `max_try` 和 `timeout`
3. **定期清理缓存**：删除过期或不需要的缓存文件
4. **批量处理**：对于大量请求，使用批量处理减少overhead

## 注意事项

1. **API 密钥安全**：不要在代码中硬编码 API 密钥，使用环境变量
2. **缓存一致性**：参数变化会产生新的缓存，注意参数的一致性
3. **存储空间**：缓存文件会占用磁盘空间，定期清理不需要的缓存
4. **并发安全**：当前实现不支持多进程并发写入同一缓存文件

## 与 eval-anything 框架集成

该工具完全集成到 eval-anything 框架中：

```python
# 在 benchmark 中使用
from eval_anything.evaluate_tools.cached_requests import CachedRequestsTool

class MyBenchmark(T2TBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_requests = CachedRequestsTool(
            cache_dir=self.output_path + "/api_cache"
        )

    def evaluate_predictions(self, predictions, task_name):
        # 使用缓存的 API 请求进行评估
        for pred in predictions:
            evaluation_prompt = f"请评估以下回答的质量：{pred}"
            messages = [{"role": "user", "content": evaluation_prompt}]

            evaluation_result = self.cached_requests.apply(
                messages=messages,
                model="gpt-4",
                max_completion_tokens=100
            )
            # 处理评估结果...
```

## 更新日志

- **v1.0.0**：初始版本，支持基础缓存功能
- 适配 eval-anything 框架
- 集成统一日志系统
- 支持多种 API 端点配置

## 贡献和反馈

如果你在使用过程中遇到问题或有改进建议，欢迎提出 issue 或贡献代码。
