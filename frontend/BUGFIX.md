# 🔧 Bug Fix - API Endpoint Mismatch

## ✅ 问题已解决！

**问题描述**: 点击 "Ask" 按钮后一直显示 "Processing..."

**根本原因**: 前端和后端的 API 端点不匹配

## 🔍 发现的问题

| 组件 | 错误的 | 正确的 |
|------|--------|--------|
| API 端点 | `/rag/query` | `/query` |
| 请求字段 | `query` | `question` |
| 响应格式 | `answer.text` | `answer` (string) |
| 源数组 | `retrieved` | `sources` |

## ✨ 已修复的内容

### 1. AskPanel.tsx
- ✅ 修改 API 端点：`/rag/query` → `/query`
- ✅ 修改请求字段：`query:` → `question:`
- ✅ 更新响应接口以匹配后端格式
- ✅ 修复答案渲染逻辑
- ✅ 修复源文档显示（`retrieved` → `sources`）

### 2. 环境配置
- ✅ 创建 `.env` 文件，配置 `VITE_RAG_BASE_URL=http://localhost:8000`

## 🚀 现在开始测试

### 步骤 1: 确保后端运行

```bash
# 在项目根目录
cd ~/Desktop/academic-rag-system

# 方式 1: 使用 Make
make serve

# 方式 2: 直接运行
python app/main.py

# 方式 3: 使用 uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**验证后端**: 访问 http://localhost:8000/docs 查看 API 文档

### 步骤 2: 启动前端（新终端）

```bash
# 进入前端目录
cd ~/Desktop/academic-rag-system/frontend

# 如果还没安装依赖
npm install

# 启动开发服务器
npm run dev
```

**浏览器会自动打开**: http://localhost:3000

### 步骤 3: 测试查询功能

1. 在 **Ask** 标签页输入问题：
   ```
   What is the Transformer architecture?
   ```

2. 设置 `top_k = 5`

3. 点击 "🔍 Ask"

4. **应该能看到**:
   - 生成的答案（带引用 [1], [2]...）
   - 检索到的文档片段
   - 相似度分数
   - 响应时间

### 步骤 4: 测试健康检查

1. 切换到 **Debug** 标签页

2. 点击 "🔍 Check Health"

3. **应该能看到**:
   - 系统指标（retrieved_n, kept_n, latency_ms）
   - 测试查询结果
   - 健康状态

## 📊 预期结果

### 成功的查询响应示例:

```json
{
  "answer": "The Transformer is a neural network architecture...",
  "sources": [
    {
      "index": 1,
      "content": "Transformer architecture uses attention...",
      "title": "Attention Is All You Need",
      "score": 0.9234,
      "metadata": {
        "doc_id": "vaswani2017attention"
      }
    }
  ],
  "num_sources": 5,
  "success": true,
  "metadata": {
    "total_time": 1.23
  }
}
```

## ⚠️ 已知限制

以下功能暂时不可用（需要后端实现）:

### Ingest Panel
- ❌ 文件上传 (`/papers/upload` 未实现)
- ❌ 刷新索引 (`/index/refresh` 未实现)

**替代方案**: 使用命令行工具
```bash
# 上传新论文
cp your-paper.pdf data/raw_papers/

# 刷新索引
make index
```

## 🔧 故障排查

### 问题: 仍然显示 "Processing..."

**检查清单**:
1. ✅ 后端是否在运行? → `curl http://localhost:8000/health`
2. ✅ 前端是否重新加载? → 刷新浏览器 (Cmd+R / Ctrl+R)
3. ✅ 浏览器控制台是否有错误? → 按 F12 查看

### 问题: 连接错误 (Failed to fetch)

**解决方案**:
```bash
# 1. 检查后端 URL
cat frontend/.env
# 应该是: VITE_RAG_BASE_URL=http://localhost:8000

# 2. 测试后端连接
curl http://localhost:8000/health

# 3. 重启前端
cd frontend
npm run dev
```

### 问题: CORS 错误

**解决方案**: 后端已经启用 CORS (`allow_origins=["*"]`)，应该不会有此问题。如果遇到，检查后端是否最新代码。

## 📝 API 端点对照表

### 当前可用的端点:

| 端点 | 方法 | 说明 | 状态 |
|------|------|------|------|
| `/query` | POST | 查询 RAG 系统 | ✅ 已修复 |
| `/health` | GET | 基本健康检查 | ✅ 可用 |
| `/health/rag` | GET | RAG 健康检查 | ✅ 可用 |
| `/stats` | GET | 系统统计 | ✅ 可用 |
| `/models` | GET | 模型信息 | ✅ 可用 |
| `/` | GET | API 信息 | ✅ 可用 |
| `/docs` | GET | API 文档 | ✅ 可用 |

### 需要实现的端点:

| 端点 | 方法 | 说明 | 状态 |
|------|------|------|------|
| `/papers/upload` | POST | 上传论文 | ❌ 未实现 |
| `/index/refresh` | POST | 刷新索引 | ❌ 未实现 |

## ✅ 验证修复成功

运行以下测试确认修复成功：

```bash
# 1. 测试后端健康
curl http://localhost:8000/health/rag

# 2. 测试查询端点
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is BERT?", "top_k": 5}'
```

**预期响应**: 应该看到完整的 JSON 响应，包含 `answer`, `sources`, `success` 等字段

## 🎊 现在可以正常使用了！

前端已经完全修复，所有查询功能应该正常工作。

---

**修复时间**: 2025-01-29
**修复内容**: API 端点匹配、请求/响应格式统一
**状态**: ✅ 完成
