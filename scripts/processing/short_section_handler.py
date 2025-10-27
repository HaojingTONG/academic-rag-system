#!/usr/bin/env python3
"""
短章节处理策略实现
智能处理长度不足的重要章节，避免信息丢失
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class ShortSectionInfo:
    """短章节信息"""
    original_section: Any  # PDFSection对象
    is_important: bool
    merge_strategy: str  # 'standalone', 'merge_prev', 'merge_next', 'context_extend'
    target_length: int

class ShortSectionHandler:
    """短章节智能处理器"""

    def __init__(self):
        self.important_section_types = {
            'conclusion', 'acknowledgment', 'limitation',
            'future_work', 'summary', 'discussion',
            'contribution', 'abstract', 'appendix'
        }

        self.important_keywords = [
            'conclude', 'conclusion', 'in conclusion',
            'acknowledge', 'thanks', 'thank',
            'limitation', 'future work', 'future research',
            'contribution', 'our contribution', 'we contribute',
            'summary', 'in summary', 'to summarize'
        ]

    def process_short_sections(self, sections: List[Any], min_length: int = 200) -> List[Dict]:
        """处理所有短章节"""

        print(f"🔍 检测短章节 (阈值: {min_length} 字符)")

        # 1. 分析所有章节
        section_info = []
        short_sections = []

        for i, section in enumerate(sections):
            content_length = len(section.content.strip())
            is_short = content_length <= min_length
            is_important = self._is_important_section(section)

            info = {
                'index': i,
                'section': section,
                'length': content_length,
                'is_short': is_short,
                'is_important': is_important,
                'merge_strategy': 'keep' if not is_short else self._decide_merge_strategy(section, i, sections)
            }

            section_info.append(info)

            if is_short:
                short_sections.append(info)
                print(f"   📏 短章节: {section.section_type} '{section.title[:30]}...' ({content_length} 字符)")

        print(f"   发现短章节: {len(short_sections)}/{len(sections)}")

        # 2. 执行处理策略
        processed_sections = self._execute_merge_strategies(section_info)

        return processed_sections

    def _is_important_section(self, section: Any) -> bool:
        """判断章节是否重要"""

        # 基于章节类型判断
        if section.section_type.lower() in self.important_section_types:
            return True

        # 基于章节标题判断
        title_lower = section.title.lower()
        if any(keyword in title_lower for keyword in self.important_keywords):
            return True

        # 基于内容关键词判断
        content_lower = section.content.lower()
        keyword_count = sum(1 for keyword in self.important_keywords if keyword in content_lower)

        # 如果内容虽短但包含多个重要关键词
        if keyword_count >= 2:
            return True

        return False

    def _decide_merge_strategy(self, section: Any, index: int, sections: List[Any]) -> str:
        """决定合并策略"""

        is_important = self._is_important_section(section)
        section_type = section.section_type.lower()
        content_length = len(section.content.strip())

        # 重要章节优先保留
        if is_important:
            if content_length >= 50:  # 至少50字符，独立保留
                return 'standalone_short'
            elif content_length >= 20:  # 20-50字符，尝试扩展上下文
                return 'context_extend'
            else:  # <20字符，与相邻章节合并
                return self._choose_merge_direction(index, sections)

        # 非重要短章节的处理
        if content_length >= 100:  # 100-200字符的非重要章节
            if self._has_meaningful_content(section.content):
                return 'standalone_short'  # 有意义内容，保留
            else:
                return 'skip'  # 跳过

        # <100字符的非重要章节，默认合并
        return self._choose_merge_direction(index, sections)

    def _choose_merge_direction(self, index: int, sections: List[Any]) -> str:
        """选择合并方向"""

        # 优先向前合并（与前面的章节合并）
        if index > 0:
            prev_section = sections[index - 1]
            if len(prev_section.content) < 2000:  # 前章节不太长
                return 'merge_prev'

        # 其次向后合并
        if index < len(sections) - 1:
            next_section = sections[index + 1]
            if len(next_section.content) < 2000:  # 后章节不太长
                return 'merge_next'

        # 都不行则尝试扩展上下文
        return 'context_extend'

    def _has_meaningful_content(self, content: str) -> bool:
        """判断内容是否有意义"""

        # 过滤掉只有页码、标题等的内容
        lines = [line.strip() for line in content.split('\\n') if line.strip()]
        meaningful_lines = []

        for line in lines:
            # 跳过页码
            if line.isdigit() and len(line) <= 3:
                continue
            # 跳过单纯的节标题
            if len(line) < 5:
                continue
            # 跳过图表引用
            if line.lower().startswith(('figure', 'table', 'fig.')):
                continue

            meaningful_lines.append(line)

        # 至少有2行有意义的内容
        return len(meaningful_lines) >= 2

    def _execute_merge_strategies(self, section_info: List[Dict]) -> List[Dict]:
        """执行合并策略"""

        processed_sections = []
        i = 0

        while i < len(section_info):
            info = section_info[i]
            strategy = info['merge_strategy']

            if strategy == 'keep':
                # 正常长度章节，直接保留
                processed_sections.append(self._create_section_dict(info['section']))

            elif strategy == 'standalone_short':
                # 重要短章节，独立保留并标记
                section_dict = self._create_section_dict(info['section'])
                section_dict['metadata']['is_short_section'] = True
                section_dict['metadata']['short_section_reason'] = 'important_standalone'
                processed_sections.append(section_dict)

            elif strategy == 'merge_prev' and processed_sections:
                # 与前一个章节合并
                prev_section = processed_sections[-1]
                merged_section = self._merge_sections(
                    prev_section, info['section'], 'append'
                )
                processed_sections[-1] = merged_section

            elif strategy == 'merge_next' and i + 1 < len(section_info):
                # 与下一个章节合并
                next_info = section_info[i + 1]
                merged_section = self._merge_sections(
                    info['section'], next_info['section'], 'prepend'
                )
                processed_sections.append(merged_section)
                i += 1  # 跳过下一个章节

            elif strategy == 'context_extend':
                # 扩展上下文保留
                extended_section = self._extend_section_context(info, section_info)
                processed_sections.append(extended_section)

            elif strategy == 'skip':
                # 跳过非重要短章节
                print(f"   ⏭️ 跳过短章节: {info['section'].section_type}")

            i += 1

        print(f"   ✅ 处理完成: {len(section_info)} → {len(processed_sections)} 个章节")

        return processed_sections

    def _create_section_dict(self, section: Any) -> Dict:
        """创建章节字典"""
        return {
            'content': f"Section: {section.title}\\n\\n{section.content}",
            'metadata': {
                'section_type': section.section_type,
                'section_title': section.title,
                'page_range': f"{section.page_range[0]}-{section.page_range[1]}",
                'confidence': section.confidence,
                'original_length': len(section.content),
                'is_short_section': False
            }
        }

    def _merge_sections(self, section1: Any, section2: Any, mode: str) -> Dict:
        """合并两个章节"""

        if mode == 'append':
            # section2 追加到 section1
            if isinstance(section1, dict):
                base_section = section1
                append_section = section2
                new_content = f"{base_section['content']}\\n\\n--- Merged Content ---\\n\\nSection: {append_section.title}\\n\\n{append_section.content}"
                merged_types = f"{base_section['metadata']['section_type']},{append_section.section_type}"
            else:
                base_section = section1
                append_section = section2
                new_content = f"Section: {base_section.title}\\n\\n{base_section.content}\\n\\n--- Merged Content ---\\n\\nSection: {append_section.title}\\n\\n{append_section.content}"
                merged_types = f"{base_section.section_type},{append_section.section_type}"
        else:  # prepend
            # section1 前置到 section2
            new_content = f"Section: {section1.title}\\n\\n{section1.content}\\n\\n--- Merged Content ---\\n\\nSection: {section2.title}\\n\\n{section2.content}"
            merged_types = f"{section1.section_type},{section2.section_type}"

        return {
            'content': new_content,
            'metadata': {
                'section_type': 'merged_section',
                'section_title': f"Merged: {section1.title if hasattr(section1, 'title') else 'Section1'} + {section2.title if hasattr(section2, 'title') else 'Section2'}",
                'page_range': f"{section1.page_range[0] if hasattr(section1, 'page_range') else 'N/A'}-{section2.page_range[1] if hasattr(section2, 'page_range') else 'N/A'}",
                'confidence': min(section1.confidence if hasattr(section1, 'confidence') else 0.5,
                                section2.confidence if hasattr(section2, 'confidence') else 0.5),
                'is_short_section': True,
                'short_section_reason': f'merged_{mode}',
                'original_types': merged_types
            }
        }

    def _extend_section_context(self, info: Dict, all_section_info: List[Dict]) -> Dict:
        """扩展章节上下文"""

        section = info['section']
        index = info['index']

        # 收集前后的一些上下文
        context_before = ""
        context_after = ""

        # 前面的上下文（最多100字符）
        if index > 0:
            prev_section = all_section_info[index - 1]['section']
            prev_content = prev_section.content.strip()
            if prev_content:
                context_before = f"...{prev_content[-100:]}\\n\\n"

        # 后面的上下文（最多100字符）
        if index < len(all_section_info) - 1:
            next_section = all_section_info[index + 1]['section']
            next_content = next_section.content.strip()
            if next_content:
                context_after = f"\\n\\n{next_content[:100]}..."

        extended_content = f"{context_before}Section: {section.title}\\n\\n{section.content}{context_after}"

        return {
            'content': extended_content,
            'metadata': {
                'section_type': section.section_type,
                'section_title': section.title,
                'page_range': f"{section.page_range[0]}-{section.page_range[1]}",
                'confidence': section.confidence,
                'is_short_section': True,
                'short_section_reason': 'context_extended',
                'original_length': len(section.content),
                'extended_length': len(extended_content)
            }
        }

def test_short_section_handler():
    """测试短章节处理器"""

    print("🧪 测试短章节处理器")
    print("=" * 50)

    # 模拟章节数据
    from dataclasses import dataclass
    from typing import Tuple

    @dataclass
    class MockSection:
        title: str
        content: str
        section_type: str
        page_range: Tuple[int, int]
        confidence: float

    # 创建测试章节
    test_sections = [
        MockSection("Introduction", "This is a long introduction with many details..." * 10, "introduction", (1, 2), 0.9),
        MockSection("Method", "Our method works as follows..." * 15, "method", (2, 4), 0.8),
        MockSection("Conclusion", "In conclusion, we showed that our approach works.", "conclusion", (10, 10), 0.7),  # 短但重要
        MockSection("Acknowledgment", "We thank the reviewers.", "acknowledgment", (11, 11), 0.6),  # 很短但重要
        MockSection("Figure Caption", "Fig 1.", "content", (5, 5), 0.3),  # 短且不重要
        MockSection("References", "Reference list..." * 20, "reference", (12, 13), 0.9)
    ]

    handler = ShortSectionHandler()
    processed_sections = handler.process_short_sections(test_sections, min_length=200)

    print(f"\\n📊 处理结果:")
    for i, section in enumerate(processed_sections):
        metadata = section['metadata']
        is_short = metadata.get('is_short_section', False)
        reason = metadata.get('short_section_reason', 'normal')
        content_preview = section['content'][:100].replace('\\n', ' ')

        print(f"   {i+1}. {metadata['section_type']} ({len(section['content'])} 字符)")
        print(f"      短章节: {is_short} ({reason})")
        print(f"      预览: {content_preview}...")
        print()

if __name__ == "__main__":
    test_short_section_handler()