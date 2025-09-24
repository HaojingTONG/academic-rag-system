# 🔧 双栏PDF处理方案总结

## 📋 问题描述
**问题**: CVPR/ICLR等双栏PDF使用`page.get_text()`会导致文本顺序混乱
- 默认提取: 左栏全部内容 → 右栏全部内容
- 正确阅读: 左栏第1段 → 右栏第1段 → 左栏第2段 → 右栏第2段...

## 💡 解决方案

### 1. 核心策略
- 使用`page.get_text("blocks")`获取文本块位置信息
- 基于x坐标检测双栏布局
- 智能重排文本块，保持自然阅读顺序

### 2. 关键排序字段
```python
block = {
    'x0': x0,           # 左边界
    'y0': y0,           # 上边界
    'x1': x1,           # 右边界
    'y1': y1,           # 下边界
    'center_x': (x0+x1)/2,  # 水平中心点 ⭐ 主要排序字段
    'center_y': (y0+y1)/2,  # 垂直中心点
    'text': text            # 文本内容
}
```

## 🎯 核心代码片段

### 双栏检测算法
```python
def _detect_two_column_layout(self, text_blocks: List[Dict], page_width: float) -> Tuple[bool, float]:
    """检测是否为双栏布局并找到分栏位置"""

    if len(text_blocks) < 6:  # 文本块太少
        return False, 0.0

    # 分析x坐标分布，寻找最大间隙
    center_xs = [block['center_x'] for block in text_blocks]
    center_xs.sort()

    max_gap = 0
    best_divider = page_width / 2

    for i in range(len(center_xs) - 1):
        gap = center_xs[i + 1] - center_xs[i]
        if gap > max_gap:
            max_gap = gap
            best_divider = (center_xs[i] + center_xs[i + 1]) / 2

    # 判断标准：间隙>页面宽度8% + 左右平衡
    if max_gap > page_width * 0.08:
        left_blocks = sum(1 for x in center_xs if x < best_divider)
        right_blocks = sum(1 for x in center_xs if x >= best_divider)

        if left_blocks >= 2 and right_blocks >= 2:
            balance_ratio = min(left_blocks, right_blocks) / max(left_blocks, right_blocks)
            if balance_ratio > 0.3:  # 左右相对平衡
                return True, best_divider

    return False, 0.0
```

### 双栏文本重排序
```python
def _reorder_two_column_text(self, text_blocks: List[Dict], column_divider: float) -> str:
    """重新排序双栏文本，按阅读顺序"""

    # 分离左右栏
    left_blocks = [b for b in text_blocks if b['center_x'] < column_divider]
    right_blocks = [b for b in text_blocks if b['center_x'] >= column_divider]

    # 按y坐标排序
    left_blocks.sort(key=lambda b: b['y0'])
    right_blocks.sort(key=lambda b: b['y0'])

    # 智能交替合并
    result_text = []
    left_idx = right_idx = 0
    y_threshold = 20  # 避免频繁切换的阈值

    while left_idx < len(left_blocks) or right_idx < len(right_blocks):
        if left_idx >= len(left_blocks):
            result_text.append(right_blocks[right_idx]['text'])
            right_idx += 1
        elif right_idx >= len(right_blocks):
            result_text.append(left_blocks[left_idx]['text'])
            left_idx += 1
        else:
            left_y = left_blocks[left_idx]['y0']
            right_y = right_blocks[right_idx]['y0']

            if left_y + y_threshold < right_y:
                result_text.append(left_blocks[left_idx]['text'])
                left_idx += 1
            elif right_y + y_threshold < left_y:
                result_text.append(right_blocks[right_idx]['text'])
                right_idx += 1
            else:
                # 差距很小，优先左栏
                result_text.append(left_blocks[left_idx]['text'])
                left_idx += 1

    return '\n\n'.join(result_text)
```

### 使用方式
```python
from src.processor.pdf_processor import AcademicPDFProcessor

processor = AcademicPDFProcessor()

# 开启双栏感知 (默认)
content = processor.extract_pdf_content("cvpr_paper.pdf", use_column_aware=True)

# 关闭双栏感知
content = processor.extract_pdf_content("cvpr_paper.pdf", use_column_aware=False)
```

## 📊 效果对比

### 测试结果 (1502.01852.pdf)
- **第2页**: 单栏检测 ✅
- **第3页**: 双栏检测 ✅ (分栏位置: 257.4, 左栏7块, 右栏8块)
- **第4页**: 单栏检测 ✅

### 性能指标
- **检测准确率**: 基于间隙分析 + 分布平衡验证
- **排序稳定性**: y坐标阈值避免频繁切换
- **兼容性**: 自动回退到单栏处理

## 🎯 关键参数调优

### 检测阈值
```python
# 双栏间隙阈值 (页面宽度百分比)
gap_threshold = page_width * 0.08  # 8%

# 左右平衡比例阈值
balance_threshold = 0.3  # 30%

# y坐标切换阈值 (避免频繁切换)
y_threshold = 20  # 20个单位
```

### 可调参数建议
- **gap_threshold**: 0.06~0.12 (6%~12%)
- **balance_threshold**: 0.2~0.5 (20%~50%)
- **y_threshold**: 10~30 (根据PDF分辨率调整)

## ✅ 实现完成

双栏PDF处理方案已完全集成到`AcademicPDFProcessor`中，支持：
- ✅ 自动双栏布局检测
- ✅ 智能文本重排序
- ✅ 单栏/双栏兼容处理
- ✅ 可配置开关控制

**使用场景**: CVPR、ICLR、NeurIPS等双栏学术论文的高质量文本提取。