# Phase 1: 核心增强功能使用指南

## 🎉 新增功能概览

Phase 1 已成功实现以下三个核心功能模块：

1. **📚 Library Panel** - 论文库浏览管理
2. **🔍 Ask Panel Enhanced** - 查询历史 + 智能建议
3. **⚙️ Settings Panel** - 实时参数配置

---

## 🚀 快速启动

### 1. 启动后端服务

```bash
# 进入项目目录
cd ~/Desktop/academic-rag-system

# 激活虚拟环境
source venv_m3max/bin/activate

# 启动后端API
python app/main.py

# 等待看到:
# ✅ RAG System API ready!
# 🚀 Starting server on 0.0.0.0:8000
```

后端将运行在 http://localhost:8000

### 2. 启动前端服务

```bash
# 新开一个终端窗口
cd ~/Desktop/academic-rag-system/frontend

# 安装依赖 (首次运行)
npm install

# 启动开发服务器
npm run dev

# 前端将运行在 http://localhost:3000
```

### 3. 打开浏览器

访问 http://localhost:3000 即可看到增强后的Dashboard！

---

## 📚 功能详解

### 1. Library Panel - 论文库管理

**位置**: Dashboard顶部导航 → `📚 Library`

**主要功能**:

#### ✅ 论文列表浏览
- 显示所有59篇已索引论文
- 展示论文标题、作者、年份、会议等元信息
- 显示每篇论文的chunk数量

#### ✅ 搜索与过滤
- 全文搜索论文标题和作者
- 实时搜索结果更新
- 清除搜索快捷按钮

#### ✅ 分页浏览
- 每页显示10篇论文
- 智能分页导航
- 显示总数和当前页信息

#### ✅ 论文详情
- 点击任意论文查看详细信息
- 显示完整作者列表
- ArXiv ID直接链接
- 文件路径查看

#### 📊 技术实现
**前端**: `frontend/src/components/LibraryPanel.tsx`
**后端API**:
- `GET /papers` - 获取论文列表 (支持分页和搜索)
- `GET /papers/{paper_id}` - 获取单篇论文详情

**使用示例**:
```typescript
// API调用示例
fetch('http://localhost:8000/papers?limit=10&offset=0&search=transformer')
  .then(res => res.json())
  .then(data => console.log(data.papers))
```

---

### 2. Ask Panel Enhanced - 智能问答增强

**位置**: Dashboard顶部导航 → `🔍 Ask` (已增强)

**新增功能**:

#### ✅ 查询历史
- 自动保存最近50条查询记录
- 本地存储 (localStorage)，无需后端
- 点击历史记录快速复用问题
- 一键清空历史

**使用方法**:
1. 在Ask Panel提问后，问题会自动保存
2. 下次访问时，顶部会显示"🕐 Recent Queries"
3. 点击"Show X"展开历史记录
4. 点击任意历史问题自动填充到输入框

#### ✅ 智能建议
- 基于论文库自动生成5个推荐问题
- 动态从论文标题提取关键概念
- 点击建议直接填充到输入框

**建议生成逻辑**:
- 从papers_info.json读取论文信息
- 提取论文标题中的核心概念
- 应用问题模板生成建议：
  - "What is {concept}?"
  - "How does {concept} work?"
  - "What are the key innovations in {concept}?"
  - ...

#### 📊 技术实现
**前端**:
- `frontend/src/components/AskPanel.tsx` (增强版)
- `frontend/src/utils/queryHistory.ts` (历史管理工具)

**后端API**:
- `GET /papers/suggest/questions?limit=5` - 获取智能建议

**使用示例**:
```typescript
// 获取建议
fetch('http://localhost:8000/papers/suggest/questions?limit=5')
  .then(res => res.json())
  .then(data => console.log(data.suggestions))

// 手动添加历史
import { addToHistory } from './utils/queryHistory'
addToHistory({
  question: "What is BERT?",
  answer: "BERT is...",
  num_sources: 5
})
```

---

### 3. Settings Panel - 实时配置管理

**位置**: Dashboard顶部导航 → `⚙️ Settings`

**主要功能**:

#### ✅ 检索参数配置
- **Top K**: 调整检索文档数量 (1-20)
- **Vector Weight**: 语义检索权重 (0.0-1.0)
- **BM25 Weight**: 关键词检索权重 (0.0-1.0)
- **Enable Reranking**: 开关重排序功能
- **Enable MMR**: 开关多样性优化

#### ✅ 生成参数配置
- **LLM Model**: 切换模型 (gpt-4o-mini / llama3.1:8b)
- **Temperature**: 调整生成随机性 (0.0-2.0)
- **Max Tokens**: 最大生成长度 (100-4000)

#### ✅ 配置管理
- **实时保存**: 配置立即生效，无需重启
- **重置默认**: 一键恢复默认配置
- **变更提示**: 未保存变更会有警告提示

#### 📊 技术实现
**前端**: `frontend/src/components/SettingsPanel.tsx`

**后端API**:
- `GET /config` - 获取当前配置
- `PUT /config` - 更新配置 (热更新)
- `POST /config/reset` - 重置为默认值

**热更新原理**:
```python
# 后端实现 (app/main.py)
@app.put("/config")
async def update_config(request: ConfigUpdateRequest):
    # 直接修改内存中的pipeline config
    for key, value in config_updates.items():
        setattr(rag_pipeline.config, key, value)
    # 下次查询即时使用新配置
    return {"message": "Configuration updated successfully"}
```

**使用示例**:
```typescript
// 更新配置
fetch('http://localhost:8000/config', {
  method: 'PUT',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    top_k: 10,
    llm_temperature: 0.2
  })
})
```

---

## 🧪 测试验证

### 测试清单

#### 1. Library Panel测试
```bash
# 1. 打开Library面板
✅ 能看到59篇论文列表
✅ 搜索"transformer"能过滤结果
✅ 点击分页按钮能翻页
✅ 点击论文能查看详情模态框
✅ 点击详情中的ArXiv链接能跳转
```

#### 2. Ask Panel测试
```bash
# 1. 提交一个问题
✅ 问题自动保存到历史
✅ 刷新页面后历史仍在
✅ 点击"Show X"能展开历史
✅ 点击历史记录能填充到输入框

# 2. 智能建议
✅ 首次加载能看到5个建议问题
✅ 点击建议能填充到输入框
✅ 建议内容与论文相关
```

#### 3. Settings Panel测试
```bash
# 1. 修改配置
✅ 拖动滑块修改权重
✅ 页面显示"⚠️ Unsaved changes"
✅ 点击"Save Changes"成功保存
✅ 成功消息显示3秒后消失

# 2. 配置生效
✅ 在Ask Panel提问，使用新的top_k
✅ 返回结果数量符合配置

# 3. 重置功能
✅ 点击"Reset to Defaults"
✅ 所有参数恢复默认值
```

### 手动测试脚本

```bash
# 测试后端API
# 1. 论文列表
curl http://localhost:8000/papers?limit=5

# 2. 搜索论文
curl "http://localhost:8000/papers?search=transformer"

# 3. 获取建议
curl http://localhost:8000/papers/suggest/questions?limit=5

# 4. 查看配置
curl http://localhost:8000/config

# 5. 更新配置
curl -X PUT http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"top_k": 10, "llm_temperature": 0.3}'

# 6. 重置配置
curl -X POST http://localhost:8000/config/reset
```

---

## 📝 文件清单

### 新增文件
```
frontend/src/components/LibraryPanel.tsx        # 论文库组件
frontend/src/components/SettingsPanel.tsx       # 设置组件
frontend/src/utils/queryHistory.ts              # 历史管理工具
PHASE1_GUIDE.md                                  # 本文档
```

### 修改文件
```
app/main.py                                      # 新增API endpoints
frontend/src/components/AskPanel.tsx             # 增强版(+历史+建议)
frontend/src/RAGDashboard.tsx                    # 集成新面板
```

---

## 🎯 下一步 (Phase 2)

Phase 1完成后，可以考虑：

1. **📊 Analytics Panel** - 数据统计可视化
   - 查询量趋势图
   - 热门论文Top 10
   - 性能指标监控

2. **⚖️ Comparison Panel** - 检索策略对比
   - A/B测试不同参数
   - 可视化对比结果

3. **🎨 UI优化**
   - 添加动画效果 (Framer Motion)
   - 响应式布局优化
   - 深色模式支持

---

## 🐛 常见问题

### Q1: 前端启动失败
```bash
# 错误: Cannot find module './utils/queryHistory'
# 解决: 确保创建了queryHistory.ts文件

cd frontend/src
mkdir -p utils
# 然后创建queryHistory.ts文件
```

### Q2: 后端API 404
```bash
# 错误: GET /papers 404
# 解决: 确保使用了修改后的app/main.py

# 检查后端日志
python app/main.py
# 应该看到新的endpoints注册信息
```

### Q3: 历史记录不保存
```bash
# 原因: localStorage被禁用或清除
# 解决:
# 1. 检查浏览器控制台是否有错误
# 2. 确保浏览器允许localStorage
# 3. 尝试隐身模式测试
```

### Q4: 配置修改不生效
```bash
# 原因: 后端未正确更新配置
# 解决:
# 1. 检查Network面板，确认PUT请求成功
# 2. 查看后端日志
# 3. 尝试重启后端服务
```

---

## 💡 使用技巧

1. **快速提问**: 点击智能建议直接提问，无需手动输入

2. **复用历史**: 点击历史记录微调后再次提问

3. **参数实验**: 在Settings中调整参数，观察Ask结果变化

4. **论文探索**: 在Library中搜索关键词，发现相关论文

5. **性能优化**: 通过Settings降低top_k提升速度

---

## ✅ 完成标志

Phase 1成功完成的标志：

- [x] 后端API新增5个endpoints
- [x] Library Panel显示论文列表
- [x] Library Panel支持搜索和分页
- [x] Ask Panel显示查询历史
- [x] Ask Panel显示智能建议
- [x] Settings Panel可以修改配置
- [x] 配置修改实时生效
- [x] 所有功能测试通过

---

**🎊 恭喜！Phase 1核心增强已完成！**

现在您的Academic RAG System拥有了：
- 📚 专业的论文库管理
- 🔍 智能的问答辅助
- ⚙️ 灵活的参数配置

项目从"基础可用"升级到了"生产级Dashboard"！ 🚀
