# 🚀 快速启动后端服务

## 问题：前端一直显示 "Processing..."

**原因**: 后端服务器没有运行

## ✅ 解决方案（5 步）

### 步骤 1: 打开终端

- **Mac**: `Cmd + Space` → 输入 "Terminal" → 回车
- **Windows**: `Win + R` → 输入 "cmd" → 回车

### 步骤 2: 进入项目目录

```bash
cd ~/Desktop/academic-rag-system
```

### 步骤 3: 启动后端

选择以下任一方式：

**方式 A - 直接运行** (推荐):
```bash
python app/main.py
```

**方式 B - 使用 Make**:
```bash
make serve
```

**方式 C - 使用 Uvicorn**:
```bash
uvicorn app.main:app --reload --port 8000
```

### 步骤 4: 等待启动完成

您应该看到以下输出：

```
🚀 Starting Academic RAG System API...
🔧 Initializing RAG components...
✅ RAG System API ready!
```

**重要**: 首次启动可能需要 1-2 分钟，因为需要加载模型

### 步骤 5: 刷新前端

1. 回到浏览器 (http://localhost:3000)
2. 页面顶部应该显示绿色的 "✓ 后端服务器已连接"
3. 现在可以输入问题并点击 "Ask"

## 🔍 验证后端是否运行

打开新的终端窗口，运行：

```bash
curl http://localhost:8000/health
```

**预期输出**:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "rag_available": true,
  ...
}
```

或者在浏览器访问：http://localhost:8000/docs

## ⚠️ 常见问题

### 问题 1: "Address already in use"（端口被占用）

**解决方案**: 杀掉占用 8000 端口的进程

```bash
# Mac/Linux
lsof -ti:8000 | xargs kill -9

# 或使用不同端口
uvicorn app.main:app --reload --port 8001
# 然后更新前端 .env 文件: VITE_RAG_BASE_URL=http://localhost:8001
```

### 问题 2: "No module named 'rag'"（模块未找到）

**解决方案**: 确认在正确的目录

```bash
pwd  # 应该显示 .../academic-rag-system
ls   # 应该看到 app/, rag/, frontend/ 等目录
```

### 问题 3: 后端启动但前端仍无法连接

**解决方案**: 检查防火墙和端口

```bash
# 1. 确认后端在监听
netstat -an | grep 8000

# 2. 测试连接
curl http://localhost:8000/health

# 3. 检查前端环境变量
cat frontend/.env
# 应该是: VITE_RAG_BASE_URL=http://localhost:8000
```

### 问题 4: 模型加载失败

**原因**: 缺少依赖或模型文件

**解决方案**:
```bash
# 重新安装依赖
pip install -r requirements.txt

# 检查模型
ls ~/.cache/huggingface/  # 应该有模型文件
```

## 📊 正常运行时的状态

### 后端终端输出：
```
🚀 Starting Academic RAG System API...
🔧 Initializing RAG components...
🔥 嵌入模型使用设备: mps
📥 加载嵌入模型: BAAI/bge-m3
✅ 模型加载成功
   - 维度: 1024
   - 设备: mps
✅ RAG System API ready!
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 前端界面应该显示：
- 顶部绿色横幅：**"✓ 后端服务器已连接 - http://localhost:8000"**
- "Ask" 按钮可点击
- 没有红色错误提示

## 💡 提示

### 1. 保持后端运行

启动后端后，**不要关闭**该终端窗口。保持它运行，打开新的终端窗口进行其他操作。

### 2. 后台运行（可选）

```bash
# 使用 nohup 后台运行
nohup python app/main.py > backend.log 2>&1 &

# 查看日志
tail -f backend.log

# 停止后台进程
ps aux | grep "app/main.py"
kill <PID>
```

### 3. 开发模式

如果您在修改代码，使用 `--reload` 模式：

```bash
uvicorn app.main:app --reload --port 8000
```

这样代码修改后会自动重启。

## 🎯 快速检查清单

在提问前确保：

- [ ] 后端终端显示 "✅ RAG System API ready!"
- [ ] 浏览器顶部显示绿色的连接状态
- [ ] http://localhost:8000/docs 可以访问
- [ ] 前端页面已刷新

## 🆘 还是不行？

如果按照以上步骤仍然无法启动：

1. **检查 Python 版本**:
   ```bash
   python --version  # 应该是 3.8+
   ```

2. **检查虚拟环境**:
   ```bash
   which python  # 应该指向 venv
   ```

3. **重新安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

4. **查看完整错误**:
   ```bash
   python app/main.py 2>&1 | tee error.log
   # 然后检查 error.log
   ```

5. **检查系统日志**:
   ```bash
   tail -100 logs/rag_system.log  # 如果有日志文件
   ```

## 📞 获取帮助

如果仍然有问题，请提供以下信息：

- 操作系统版本
- Python 版本
- 完整的错误输出
- `pip list` 的输出

---

**记住**: 前端（React）和后端（Python FastAPI）是两个独立的服务，都需要运行才能正常工作！

- **后端**: `python app/main.py` (端口 8000)
- **前端**: `npm run dev` (端口 3000)
