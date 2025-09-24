#!/usr/bin/env python3
"""
双栏PDF文本提取和重组测试脚本
演示不同PyMuPDF提取模式的效果
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Tuple, Any

def demonstrate_text_extraction_modes(pdf_path: str, page_num: int = 0):
    """演示不同文本提取模式的效果"""

    print(f"🔍 测试PDF: {Path(pdf_path).name}")
    print(f"📄 测试页面: {page_num + 1}")
    print("=" * 80)

    doc = fitz.open(pdf_path)
    page = doc[page_num]

    print("1️⃣ 【默认模式】 page.get_text():")
    default_text = page.get_text()
    print(f"   前200字符: {default_text[:200].replace(chr(10), '↵')}")
    print()

    print("2️⃣ 【Blocks模式】 page.get_text('blocks'):")
    blocks = page.get_text("blocks")
    print(f"   总block数: {len(blocks)}")

    # 显示前3个block的信息
    for i, block in enumerate(blocks[:3]):
        x0, y0, x1, y1, text, block_no, block_type = block
        print(f"   Block {i}: 位置({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})")
        print(f"            类型: {block_type}, 内容: {text[:50].replace(chr(10), '↵')}...")
    print()

    print("3️⃣ 【Dict模式】 page.get_text('dict'):")
    page_dict = page.get_text("dict")
    print(f"   总block数: {len(page_dict['blocks'])}")

    # 分析第一个文本block
    for i, block in enumerate(page_dict["blocks"][:3]):
        if "lines" in block:  # 文本block
            print(f"   Block {i}: 位置({block['bbox'][0]:.1f}, {block['bbox'][1]:.1f})")
            print(f"            行数: {len(block['lines'])}")
            if block['lines']:
                first_line = block['lines'][0]
                if 'spans' in first_line:
                    text_content = ''.join(span['text'] for span in first_line['spans'])
                    print(f"            首行: {text_content[:50]}...")

    doc.close()
    print()

def extract_column_aware_text(pdf_path: str, page_num: int = 0, debug: bool = True) -> str:
    """基于列感知的文本提取"""

    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # 获取页面尺寸
    page_rect = page.rect
    page_width = page_rect.width
    page_height = page_rect.height

    if debug:
        print(f"📏 页面尺寸: {page_width:.1f} x {page_height:.1f}")

    # 使用blocks模式获取文本块
    blocks = page.get_text("blocks")
    text_blocks = []

    for block in blocks:
        x0, y0, x1, y1, text, block_no, block_type = block

        # 只处理文本块 (block_type = 0)
        if block_type == 0 and text.strip():
            text_blocks.append({
                'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                'text': text.strip(),
                'center_x': (x0 + x1) / 2,
                'center_y': (y0 + y1) / 2,
                'width': x1 - x0,
                'height': y1 - y0
            })

    if debug:
        print(f"📝 文本块数量: {len(text_blocks)}")

    # 检测是否为双栏布局
    is_two_column, column_divider = detect_two_column_layout(text_blocks, page_width, debug)

    if is_two_column:
        return reorder_two_column_text(text_blocks, column_divider, debug)
    else:
        # 单栏布局，按y坐标排序
        text_blocks.sort(key=lambda b: b['y0'])
        return '\n\n'.join(block['text'] for block in text_blocks)

def detect_two_column_layout(text_blocks: List[Dict], page_width: float, debug: bool = True) -> Tuple[bool, float]:
    """检测是否为双栏布局并找到分栏位置"""

    if len(text_blocks) < 4:  # 文本块太少，可能不是双栏
        return False, 0.0

    # 分析文本块的x坐标分布
    center_xs = [block['center_x'] for block in text_blocks]

    # 尝试用K-means简单聚类找到两列
    left_centers = [x for x in center_xs if x < page_width * 0.6]
    right_centers = [x for x in center_xs if x > page_width * 0.4]

    # 判断是否有明显的左右分布
    if len(left_centers) >= 2 and len(right_centers) >= 2:
        # 计算列的边界
        left_max = max([block['x1'] for block in text_blocks if block['center_x'] < page_width * 0.6])
        right_min = min([block['x0'] for block in text_blocks if block['center_x'] > page_width * 0.4])

        # 检查是否有明显的列间距
        column_gap = right_min - left_max

        if column_gap > 10:  # 至少10个单位的间距
            column_divider = (left_max + right_min) / 2

            if debug:
                print(f"✅ 检测到双栏布局:")
                print(f"   左栏范围: ~{left_max:.1f}")
                print(f"   右栏范围: {right_min:.1f}~")
                print(f"   列分割线: {column_divider:.1f}")
                print(f"   列间距: {column_gap:.1f}")

            return True, column_divider

    if debug:
        print("📄 检测为单栏布局")

    return False, 0.0

def reorder_two_column_text(text_blocks: List[Dict], column_divider: float, debug: bool = True) -> str:
    """重新排序双栏文本"""

    # 分离左右栏
    left_blocks = [b for b in text_blocks if b['center_x'] < column_divider]
    right_blocks = [b for b in text_blocks if b['center_x'] >= column_divider]

    # 按y坐标排序
    left_blocks.sort(key=lambda b: b['y0'])
    right_blocks.sort(key=lambda b: b['y0'])

    if debug:
        print(f"📊 左栏块数: {len(left_blocks)}, 右栏块数: {len(right_blocks)}")

    # 交替合并文本 - 改进版
    result_text = []

    # 方法1: 基于y坐标范围匹配
    left_idx = 0
    right_idx = 0

    while left_idx < len(left_blocks) or right_idx < len(right_blocks):
        # 选择y坐标更小（更靠上）的块
        if left_idx >= len(left_blocks):
            # 左栏用完，添加右栏剩余
            result_text.append(right_blocks[right_idx]['text'])
            right_idx += 1
        elif right_idx >= len(right_blocks):
            # 右栏用完，添加左栏剩余
            result_text.append(left_blocks[left_idx]['text'])
            left_idx += 1
        else:
            left_y = left_blocks[left_idx]['y0']
            right_y = right_blocks[right_idx]['y0']

            if left_y <= right_y:
                result_text.append(left_blocks[left_idx]['text'])
                left_idx += 1
            else:
                result_text.append(right_blocks[right_idx]['text'])
                right_idx += 1

    return '\n\n'.join(result_text)

def compare_extraction_methods(pdf_path: str, page_num: int = 0):
    """比较不同提取方法的效果"""

    print("🔄 比较不同文本提取方法")
    print("=" * 80)

    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # 方法1: 默认提取
    default_text = page.get_text()

    # 方法2: 列感知提取
    column_aware_text = extract_column_aware_text(pdf_path, page_num, debug=False)

    doc.close()

    print("📊 提取结果对比:")
    print(f"   默认方法字符数: {len(default_text)}")
    print(f"   列感知方法字符数: {len(column_aware_text)}")

    print("\n📝 默认方法前300字符:")
    print(f"   {default_text[:300].replace(chr(10), '↵')}")

    print("\n🎯 列感知方法前300字符:")
    print(f"   {column_aware_text[:300].replace(chr(10), '↵')}")

if __name__ == "__main__":
    # 测试用例
    import glob

    pdf_files = glob.glob("/Users/haojingtong/Desktop/academic-rag-system/data/raw_papers/*.pdf")

    if pdf_files:
        test_file = pdf_files[0]  # 使用第一个PDF文件

        print("🚀 双栏PDF处理测试")
        print("=" * 80)

        # 1. 演示不同提取模式
        demonstrate_text_extraction_modes(test_file, 1)  # 使用第2页

        # 2. 测试列感知提取
        print("🎯 列感知文本提取测试:")
        print("=" * 80)
        extracted_text = extract_column_aware_text(test_file, 1)
        print(f"✅ 提取完成，总字符数: {len(extracted_text)}")

        # 3. 比较提取效果
        print("\n")
        compare_extraction_methods(test_file, 1)

    else:
        print("❌ 未找到测试PDF文件")