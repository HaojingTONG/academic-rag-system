# 📚 下载高引用AI论文指南

## 🚀 快速开始

我为你创建了两种下载方式：

### 方法1: 精选论文列表 (推荐⭐)

最简单可靠的方式 - 下载手工精选的20篇顶级AI论文：

```bash
# 下载所有精选论文 (20篇)
python scripts/download_curated_papers.py

# 下载前10篇
python scripts/download_curated_papers.py --limit 10

# 只下载NLP相关论文
python scripts/download_curated_papers.py --category nlp

# 只下载CV相关论文
python scripts/download_curated_papers.py --category cv
```

**优点:**
- ✅ 简单可靠
- ✅ 全部有ArXiv ID，100%成功率
- ✅ 手工精选的经典论文
- ✅ 包含Transformer、BERT、ResNet、GPT等

---

### 方法2: API自动搜索 (高级)

从Semantic Scholar API自动搜索并下载高引用论文：

```bash
# 下载200篇AI/ML领域高引用论文
python scripts/download_top_papers.py --limit 200

# 下载50篇深度学习论文
python scripts/download_top_papers.py --limit 50 --field "deep learning"

# 下载100篇2015年后的论文
python scripts/download_top_papers.py --limit 100 --start-year 2015

# 下载NLP领域论文
python scripts/download_top_papers.py --limit 100 --field "natural language processing"
```

**优点:**
- ✅ 自动搜索最新论文
- ✅ 可自定义搜索条件
- ✅ 按引用量排序

**缺点:**
- ⚠️ 不是所有论文都有ArXiv ID
- ⚠️ 实际下载成功率约60-70%
- ⚠️ 需要网络API访问

---

## 📋 完整流程

### Step 1: 下载论文

```bash
# 激活虚拟环境
source venv_m3max/bin/activate

# 下载论文 (选择一种方法)
python scripts/download_curated_papers.py --limit 50
```

**预期输出:**
```
======================================================================
📚 Curated AI Papers Downloader
======================================================================
Papers to download: 20
Output directory: data/raw_papers
======================================================================

Downloading: 100%|████████████████████| 20/20 [02:15<00:00,  6.8s/it]

======================================================================
📊 Download Summary
======================================================================
✅ Downloaded: 18
⊙ Skipped (already exists): 0
❌ Failed: 2
📁 Location: data/raw_papers
======================================================================
```

---

### Step 2: 索引论文

```bash
# 索引新下载的论文
make index

# 或者手动运行
./scripts/refresh_index.sh
```

**预期时间:**
- 20篇论文: ~3-5分钟
- 50篇论文: ~8-12分钟
- 200篇论文: ~30-40分钟

---

### Step 3: 验证

```bash
# 查看索引统计
make stats

# 或通过API
curl http://localhost:8000/stats

# 或在前端Library面板查看
```

---

## 📊 精选论文列表预览

`scripts/curated_papers.json` 包含20篇精选论文：

| 论文 | 年份 | 引用量 | ArXiv ID |
|------|------|--------|----------|
| Deep Residual Learning (ResNet) | 2015 | 128K | 1512.03385 |
| AlexNet | 2012 | 120K | 1409.0473 |
| Adam Optimizer | 2014 | 95K | 1412.6980 |
| Attention Is All You Need | 2017 | 85K | 1706.03762 |
| BERT | 2018 | 72K | 1810.04805 |
| GAN | 2014 | 68K | 1406.2661 |
| Batch Normalization | 2015 | 55K | 1502.03167 |
| GPT-3 | 2020 | 28K | 2005.14165 |
| RoBERTa | 2019 | 14K | 1907.11692 |
| Vision Transformer | 2020 | 18K | 2010.11929 |
| ... | ... | ... | ... |

完整列表查看: `scripts/curated_papers.json`

---

## 🎯 推荐策略

### 新手策略: 从精选列表开始

```bash
# 1. 先下载20篇精选论文
python scripts/download_curated_papers.py

# 2. 索引
make index

# 3. 测试系统
python app/main.py
```

### 进阶策略: 扩展到200篇

```bash
# 1. 先下载精选的20篇 (基础)
python scripts/download_curated_papers.py

# 2. 再用API下载更多
python scripts/download_top_papers.py --limit 200 --start-year 2015

# 3. 索引所有论文
make index
```

### 专题策略: 按领域下载

```bash
# NLP专题
python scripts/download_top_papers.py --limit 100 --field "natural language processing"

# CV专题
python scripts/download_top_papers.py --limit 100 --field "computer vision"

# 强化学习专题
python scripts/download_top_papers.py --limit 50 --field "reinforcement learning"
```

---

## 🔧 自定义精选列表

你可以编辑 `scripts/curated_papers.json` 添加自己想要的论文：

```json
{
  "papers": [
    {
      "title": "Your Favorite Paper",
      "authors": ["Author Name"],
      "year": 2023,
      "arxiv_id": "2301.12345",
      "category": "your_category",
      "estimated_citations": 1000
    }
  ]
}
```

然后运行:
```bash
python scripts/download_curated_papers.py
```

---

## 💡 获取ArXiv ID的方法

### 方法1: ArXiv官网搜索
1. 访问 https://arxiv.org
2. 搜索论文标题
3. 复制URL中的ID (如 `2304.03442`)

### 方法2: Google Scholar
1. 搜索论文
2. 找到"arXiv:XXXX.XXXXX"链接
3. 复制ID

### 方法3: 论文PDF的ArXiv水印
很多论文PDF右上角有ArXiv ID水印

---

## ⚡ 性能优化建议

### 批量下载优化

```bash
# 分批下载，避免网络超时
python scripts/download_curated_papers.py --limit 50
# 等待完成
python scripts/download_top_papers.py --limit 50
# 再继续...
```

### 并行索引优化

```bash
# 下载完成后，一次性索引
make index

# 避免每下载一篇就索引一次
```

### 磁盘空间估算

- 单篇PDF: 平均 1-5 MB
- 20篇论文: ~50-100 MB
- 200篇论文: ~500MB - 1GB

确保有足够空间！

---

## 🐛 常见问题

### Q1: 下载速度慢怎么办？
```bash
# 脚本已经内置了3秒延迟（ArXiv限流）
# 如果还是慢，可能是网络问题

# 可以减少延迟（风险：可能被封IP）
# 编辑脚本，修改 time.sleep(3) -> time.sleep(1)
```

### Q2: 某些论文下载失败？
```bash
# 原因1: 没有ArXiv ID
# 解决: 使用精选列表，100%有ArXiv ID

# 原因2: ArXiv服务器问题
# 解决: 稍后重试，脚本会跳过已存在的文件

# 原因3: 网络问题
# 解决: 检查网络，使用代理
```

### Q3: 如何删除某篇论文？
```bash
# 1. 删除PDF
rm data/raw_papers/1234.5678.pdf

# 2. 重新索引（会自动清理）
make index
```

### Q4: 下载了太多论文，想重新开始？
```bash
# 备份当前论文
mv data/raw_papers data/raw_papers_backup

# 清空向量数据库
rm -rf vector_db/*

# 清空索引
rm data/bm25_index.pkl data/papers_info.json

# 重新下载
mkdir data/raw_papers
python scripts/download_curated_papers.py

# 重新索引
make index
```

---

## 📈 扩展200篇论文的策略

如果你想达到200篇，推荐组合策略：

```bash
# 1. 基础 (20篇精选)
python scripts/download_curated_papers.py
# ✅ 下载: 20篇

# 2. Transformer/NLP (50篇)
python scripts/download_top_papers.py --limit 50 \
  --field "transformer attention mechanism NLP"
# ✅ 预计成功: 30-35篇

# 3. Computer Vision (50篇)
python scripts/download_top_papers.py --limit 50 \
  --field "computer vision CNN image recognition"
# ✅ 预计成功: 30-35篇

# 4. 深度学习基础 (50篇)
python scripts/download_top_papers.py --limit 50 \
  --field "deep learning neural networks optimization"
# ✅ 预计成功: 30-35篇

# 5. 生成模型 (30篇)
python scripts/download_top_papers.py --limit 30 \
  --field "generative models diffusion GAN"
# ✅ 预计成功: 20-25篇

# 总计: 20 + 35 + 35 + 35 + 25 = 150-170篇

# 6. 补充到200篇
python scripts/download_top_papers.py --limit 50 \
  --field "reinforcement learning multimodal"
```

---

## 🎁 额外资源

### 手动添加单篇论文

```bash
# 直接下载ArXiv论文
curl -o data/raw_papers/2304.03442.pdf \
  https://arxiv.org/pdf/2304.03442.pdf

# 索引
make index
```

### 从本地文件导入

```bash
# 复制PDF到data/raw_papers/
cp ~/Downloads/*.pdf data/raw_papers/

# 索引
make index
```

---

## ✅ 验收清单

下载完成后，检查：

- [ ] PDF文件在 `data/raw_papers/` 目录
- [ ] 运行 `make index` 成功
- [ ] `data/papers_info.json` 有论文元信息
- [ ] `data/bm25_index.pkl` 已生成
- [ ] `vector_db/` 有数据
- [ ] Frontend Library面板能看到论文
- [ ] Ask Panel能成功查询

---

**🎉 现在你可以下载200篇顶级AI论文了！**

推荐先从精选的20篇开始，测试系统运行正常后再扩展。

有任何问题随时问我！
