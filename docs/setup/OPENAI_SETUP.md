# OpenAI LLM 配置指南

本指南将帮助您将系统从 Ollama 切换到 OpenAI 的 GPT 模型。

## 📋 前提条件

1. **获取 OpenAI API 密钥**
   - 访问 [OpenAI Platform](https://platform.openai.com/api-keys)
   - 注册/登录您的账户
   - 创建新的 API 密钥
   - 复制并保存密钥（只会显示一次）

2. **安装 OpenAI 库**
   ```bash
   pip install openai>=1.0.0
   # 或者重新安装所有依赖
   pip install -r requirements.txt
   ```

## 🔧 配置步骤

### 方法 1: 使用环境变量（推荐）

1. **创建 .env 文件**
   ```bash
   cp .env.example .env
   ```

2. **编辑 .env 文件，设置以下变量**
   ```bash
   # 选择后端
   LLM_BACKEND=openai

   # 设置您的 API 密钥
   OPENAI_API_KEY=sk-your-actual-api-key-here

   # 选择模型（可选，默认为 gpt-4o-mini）
   LLM_MODEL=gpt-4o-mini

   # 其他 OpenAI 配置（可选）
   OPENAI_API_BASE=https://api.openai.com/v1
   # OPENAI_ORGANIZATION=org-your-org-id
   ```

3. **可用的 OpenAI 模型**
   - `gpt-4o` - 最新最强大的模型（较贵）
   - `gpt-4o-mini` - 性价比高，推荐用于大多数任务
   - `gpt-3.5-turbo` - 更便宜，速度快
   - `gpt-4` - 旧版 GPT-4

### 方法 2: 直接修改配置文件

编辑 `configs/config.yaml`:

```yaml
generation:
  # 设置后端为 openai
  llm_backend: "openai"

  # OpenAI 配置
  openai_api_key: "${OPENAI_API_KEY}"  # 仍建议从环境变量读取
  openai_api_base: "https://api.openai.com/v1"

  # 选择模型
  model: "gpt-4o-mini"
  fallback_models:
    - "gpt-3.5-turbo"
    - "gpt-4"

  # 生成参数
  temperature: 0.1
  max_tokens: 2000
```

## 🚀 使用示例

### 命令行交互模式

```bash
# 确保已设置环境变量
export OPENAI_API_KEY=sk-your-api-key-here

# 运行 CLI
python -m app.cli query --interactive

# 或使用 Makefile
make run
```

### Python 代码示例

```python
from src.generator.llm_client import create_llm_manager_from_config

# 从配置自动创建管理器
manager = create_llm_manager_from_config()

# 生成回答
prompt = """基于以下学术资料回答问题：

问题：什么是 transformer 架构？

上下文资料：
Transformer architecture uses self-attention mechanism...

请直接回答问题："""

response = manager.generate_answer(prompt)
print(f"回答: {response.text}")
print(f"使用模型: {response.model}")
print(f"Token 用量: {response.tokens_used}")
```

### FastAPI Web 服务

```bash
# 启动 API 服务
make serve

# 或直接使用 uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

然后访问 `http://localhost:8000/docs` 查看 API 文档。

## 💰 成本估算

OpenAI 按 token 计费，以下是参考价格（2024年价格）：

| 模型 | 输入价格 | 输出价格 | 推荐场景 |
|------|---------|---------|---------|
| gpt-4o-mini | $0.15/1M tokens | $0.60/1M tokens | 日常使用，性价比最高 |
| gpt-4o | $2.50/1M tokens | $10.00/1M tokens | 需要最高质量时 |
| gpt-3.5-turbo | $0.50/1M tokens | $1.50/1M tokens | 简单任务，追求速度 |

**估算示例**：
- 一次 RAG 查询平均使用约 2000-4000 tokens
- 使用 gpt-4o-mini，每1000次查询约花费 $0.50-$1.00
- 使用 gpt-4o，每1000次查询约 $10-$20

## 🔄 切换回 Ollama

如果需要切换回本地 Ollama：

1. **修改 .env 文件**
   ```bash
   LLM_BACKEND=ollama
   LLM_MODEL=llama3.1:8b
   ```

2. **或修改 configs/config.yaml**
   ```yaml
   generation:
     llm_backend: "ollama"
     model: "llama3.1:8b"
   ```

## 🧪 测试配置

运行测试脚本验证配置：

```bash
# 测试 OpenAI 连接
python -c "
from src.generator.llm_client import OpenAIClient
import os

client = OpenAIClient()
print(f'OpenAI 可用: {client.is_available()}')
if client.is_available():
    models = client.get_available_models()
    print(f'可用模型: {models[:5]}')  # 显示前5个
"
```

## ❓ 常见问题

### Q: API 密钥无效怎么办？
A: 检查以下几点：
- 确保 API 密钥正确复制（以 `sk-` 开头）
- 检查账户是否有余额
- 验证密钥没有被撤销
- 环境变量是否正确设置

### Q: 请求超时怎么办？
A: 可以调整超时设置：
```yaml
generation:
  timeout: 120  # 增加到120秒
```

### Q: 如何使用自定义 API 端点（如 Azure OpenAI）？
A: 修改 `OPENAI_API_BASE`:
```bash
OPENAI_API_BASE=https://your-resource.openai.azure.com/
OPENAI_ORGANIZATION=your-org-id
```

### Q: 如何监控 API 使用量和成本？
A: 访问 [OpenAI Usage Dashboard](https://platform.openai.com/usage)

## 📚 更多资源

- [OpenAI API 文档](https://platform.openai.com/docs)
- [定价详情](https://openai.com/api/pricing/)
- [Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
- [最佳实践](https://platform.openai.com/docs/guides/production-best-practices)

## 🔒 安全建议

1. **永远不要在代码中硬编码 API 密钥**
2. **不要提交 .env 文件到 Git**（已在 .gitignore 中）
3. **定期轮换 API 密钥**
4. **为不同环境使用不同的密钥**
5. **启用 API 使用限制**，防止意外高额账单

---

如有问题，请参考 [主 README](README.md) 或提交 Issue。
