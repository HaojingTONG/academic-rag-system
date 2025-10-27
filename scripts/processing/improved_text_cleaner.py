#!/usr/bin/env python3
"""
改进的文本清洗函数
保留换行与结构，同时清理噪声和重复空格
"""

import re
from typing import List, Tuple

class ImprovedTextCleaner:
    """改进的文本清洗器"""

    def __init__(self):
        # 列表标识符模式
        self.list_patterns = [
            r'^\s*[-•·]\s+',           # 项目符号列表
            r'^\s*[0-9]+\.\s+',        # 数字列表 (1. 2. 3.)
            r'^\s*[0-9]+\)\s+',        # 数字列表 (1) 2) 3))
            r'^\s*[a-zA-Z]\.\s+',      # 字母列表 (a. b. c.)
            r'^\s*[a-zA-Z]\)\s+',      # 字母列表 (a) b) c))
            r'^\s*[ivx]+\.\s+',        # 罗马数字列表 (i. ii. iii.)
            r'^\s*\*\s+',              # 星号列表
            r'^\s*\+\s+',              # 加号列表
        ]

        # 特殊结构模式
        self.structure_patterns = {
            'section_header': r'^\s*(?:\d+\.?\s*)?[A-Z][^.!?]*$',  # 章节标题
            'equation': r'\$.*?\$|\\[a-zA-Z]+\{.*?\}',              # 数学公式
            'citation': r'\[[0-9,\-\s]+\]|\([A-Za-z]+,?\s*[0-9]{4}\)',  # 引用
            'url': r'http[s]?://\S+|www\.\S+',                     # URL
            'code_block': r'```[\s\S]*?```|`[^`]+`',                # 代码块
        }

        # 噪声模式
        self.noise_patterns = [
            r'^\s*\d+\s*$',                    # 单独的页码
            r'^\s*Figure\s+\d+[:\.]?\s*$',     # 单独的图片标题行
            r'^\s*Table\s+\d+[:\.]?\s*$',      # 单独的表格标题行
            r'^\s*Fig\.\s*\d+\s*$',            # 简化图片引用
            r'^\s*[^\w\s]{3,}\s*$',            # 纯符号行(如分隔线)
        ]

    def clean_text_preserving_structure(self, text: str) -> str:
        """
        改进的文本清洗，保留结构和格式

        Args:
            text: 原始文本

        Returns:
            清洗后的文本，保留换行和列表结构
        """
        if not text or not text.strip():
            return ""

        # 第1步：预处理 - 标准化行结束符
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 第2步：按行处理，保留重要结构
        lines = text.split('\n')
        processed_lines = []

        for i, line in enumerate(lines):
            processed_line = self._process_line(line, i, lines)
            if processed_line is not None:  # None表示跳过该行
                processed_lines.append(processed_line)

        # 第3步：智能合并行，保留段落结构
        structured_text = self._merge_lines_intelligently(processed_lines)

        # 第4步：最终清理
        final_text = self._final_cleanup(structured_text)

        return final_text.strip()

    def _process_line(self, line: str, line_index: int, all_lines: List[str]) -> str:
        """
        处理单行文本

        Returns:
            处理后的行，或 None 表示跳过该行
        """
        # 跳过完全空白的行（但保留单个换行作为段落分隔）
        if not line.strip():
            return ""

        # 检查是否是噪声行
        if self._is_noise_line(line):
            return None  # 跳过噪声行

        # 保留特殊结构行的格式
        if self._is_structure_line(line):
            return self._clean_structure_line(line)

        # 检查是否是列表项
        if self._is_list_item(line):
            return self._clean_list_item(line)

        # 普通文本行清理
        return self._clean_normal_line(line)

    def _is_noise_line(self, line: str) -> bool:
        """检查是否是噪声行"""
        line_stripped = line.strip()

        # 空行不算噪声
        if not line_stripped:
            return False

        # 检查噪声模式
        for pattern in self.noise_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True

        # 长度过短且没有实际内容的行
        if len(line_stripped) < 3 and not re.search(r'\w', line_stripped):
            return True

        return False

    def _is_structure_line(self, line: str) -> bool:
        """检查是否是结构化行（章节标题等）"""
        line_stripped = line.strip()

        # 检查章节标题模式
        if re.match(self.structure_patterns['section_header'], line_stripped):
            return True

        # 检查是否包含公式、引用等特殊内容
        for pattern_name, pattern in self.structure_patterns.items():
            if pattern_name != 'section_header' and re.search(pattern, line):
                return True

        return False

    def _is_list_item(self, line: str) -> bool:
        """检查是否是列表项"""
        for pattern in self.list_patterns:
            if re.match(pattern, line):
                return True
        return False

    def _clean_structure_line(self, line: str) -> str:
        """清理结构化行"""
        # 移除行首尾多余空格，但保留内部格式
        line = line.strip()

        # 清理多余的空格（但不破坏特殊格式）
        line = re.sub(r'[ \t]+', ' ', line)

        return line

    def _clean_list_item(self, line: str) -> str:
        """清理列表项"""
        # 保留列表项的缩进和标识符
        # 只清理内容部分的多余空格

        # 找到列表标识符
        for pattern in self.list_patterns:
            match = re.match(pattern, line)
            if match:
                prefix = match.group(0)  # 列表标识符部分
                content = line[match.end():].strip()  # 内容部分

                # 清理内容部分
                content = re.sub(r'\s+', ' ', content)

                return prefix + content

        # 如果没有匹配到模式，按普通行处理
        return self._clean_normal_line(line)

    def _clean_normal_line(self, line: str) -> str:
        """清理普通文本行"""
        # 移除行首尾空格
        line = line.strip()

        # 清理多余的空格和制表符
        line = re.sub(r'[ \t]+', ' ', line)

        # 清理多余的标点符号
        line = re.sub(r'([.!?]){2,}', r'\1', line)  # 重复标点
        line = re.sub(r'([,;:]){2,}', r'\1', line)   # 重复标点

        return line

    def _merge_lines_intelligently(self, lines: List[str]) -> str:
        """智能合并行，保留段落结构"""
        if not lines:
            return ""

        result_lines = []
        i = 0

        while i < len(lines):
            current_line = lines[i]

            # 空行作为段落分隔符
            if current_line == "":
                result_lines.append("")
                i += 1
                continue

            # 结构化行（标题、列表等）单独成行
            if (self._is_structure_line(current_line) or
                self._is_list_item(current_line)):
                result_lines.append(current_line)
                i += 1
                continue

            # 普通文本行：检查是否需要与下一行合并
            merged_line = current_line
            j = i + 1

            while j < len(lines):
                next_line = lines[j]

                # 遇到空行、结构化行或列表项，停止合并
                if (next_line == "" or
                    self._is_structure_line(next_line) or
                    self._is_list_item(next_line)):
                    break

                # 检查是否应该合并
                if self._should_merge_lines(merged_line, next_line):
                    merged_line += " " + next_line
                    j += 1
                else:
                    break

            result_lines.append(merged_line)
            i = j

        return '\n'.join(result_lines)

    def _should_merge_lines(self, line1: str, line2: str) -> bool:
        """判断两行是否应该合并"""
        # 如果第一行以句号、感叹号或问号结尾，通常不合并
        if re.search(r'[.!?]\s*$', line1.strip()):
            return False

        # 如果第二行看起来像新句子的开始（大写字母），可能不应该合并
        if re.match(r'^[A-Z]', line2.strip()):
            # 但如果第一行很短（可能是断行），还是要合并
            return len(line1.strip()) < 50

        # 如果第二行是数字或特殊字符开头，可能不应该合并
        if re.match(r'^\d+|^[^\w\s]', line2.strip()):
            return False

        # 默认情况下合并短行
        return True

    def _final_cleanup(self, text: str) -> str:
        """最终清理"""
        # 清理多个连续空行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 清理行尾空格
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)

        # 移除特殊字符（但保留基本标点和数学符号）
        # 更温和的清理，避免破坏重要内容
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\"\'\/\\\$\%\#\@\&\*\+\=\<\>\~\`\n]', ' ', text)

        # 最后清理：移除多余空格，但保留换行
        text = re.sub(r'[ \t]+', ' ', text)

        return text

# 测试和对比函数
def compare_cleaning_methods():
    """对比新旧文本清洗方法的效果"""

    # 测试文本 - 包含各种格式
    test_text = """
    Title: Attention Is All You Need

    Abstract: The dominant sequence transduction models are based on complex recurrent
    or convolutional neural networks that include an encoder and a decoder.

    1. Introduction

    Machine learning has made significant progress in recent years. Key contributions include:

    • Deep learning architectures
    • Attention mechanisms
    • Transformer models

    2. Method

    Our approach consists of three steps:
    a) Data preprocessing
    b) Model training
    c) Evaluation

    The mathematical formulation is: $y = f(x; \\theta)$

    3. Experiments

    We evaluated on multiple datasets [1, 2, 3].
    Results are shown in Table 1.

    Figure 1

    Some conclusions can be drawn from this analysis.

    4. Conclusion

    In conclusion, we have shown that our approach works well.

    References
    [1] Author et al. "Title", Conference 2023.
    """

    print("🧪 对比文本清洗方法")
    print("=" * 80)

    # 旧方法（原来的_clean_text）
    def old_clean_text(text):
        """旧的清洗方法"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)

        # 移除特殊字符（保留基本标点）
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\"\'\/\\\$\%\#\@\&\*\+\=\<\>\~\`]', ' ', text)

        # 移除多余的换行和空格
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())

        return text.strip()

    # 新方法
    cleaner = ImprovedTextCleaner()

    print("📄 原始文本:")
    print("-" * 40)
    print(repr(test_text))

    print(f"\n🔧 旧方法清洗结果:")
    print("-" * 40)
    old_result = old_clean_text(test_text)
    print(repr(old_result))
    print(f"\n显示效果:")
    print(old_result)

    print(f"\n✨ 新方法清洗结果:")
    print("-" * 40)
    new_result = cleaner.clean_text_preserving_structure(test_text)
    print(repr(new_result))
    print(f"\n显示效果:")
    print(new_result)

    # 统计对比
    print(f"\n📊 统计对比:")
    print(f"原始文本：{len(test_text)} 字符, {len(test_text.split())} 词")
    print(f"旧方法：{len(old_result)} 字符, {len(old_result.split())} 词")
    print(f"新方法：{len(new_result)} 字符, {len(new_result.split())} 词")

    print(f"\n🔍 结构保留对比:")
    print(f"原始换行数：{test_text.count(chr(10))}")
    print(f"旧方法换行数：{old_result.count(chr(10))}")
    print(f"新方法换行数：{new_result.count(chr(10))}")

if __name__ == "__main__":
    compare_cleaning_methods()