# src/processor/pdf_processor.py
"""
PDF全文处理模块 - 提取学术论文的完整内容
支持智能章节识别、内容清理和结构化处理
"""

import fitz  # PyMuPDF
import re
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import logging

@dataclass
class PDFSection:
    """PDF章节数据结构"""
    title: str
    content: str
    section_type: str  # abstract, introduction, related_work, background, method, implementation, experiment, dataset, result, discussion, limitation, conclusion, acknowledgment, appendix, reference, content, header_content, footer_content, transition
    page_range: Tuple[int, int]
    confidence: float  # 识别置信度

@dataclass 
class PDFContent:
    """PDF完整内容数据结构"""
    title: str
    abstract: str
    sections: List[PDFSection]
    total_pages: int
    total_words: int
    has_formulas: bool
    has_tables: bool
    has_figures: bool
    language: str  # en, zh, etc.
    
class AcademicPDFProcessor:
    """学术论文PDF处理器"""
    
    def __init__(self):
        """初始化PDF处理器"""
        # 常见章节标题模式（支持中英文）
        self.section_patterns = {
            'abstract': [
                r'^\s*abstract\s*$',
                r'^\s*摘\s*要\s*$',
                r'^\s*summary\s*$'
            ],
            'introduction': [
                r'^\s*1\.?\s*introduction\s*$',
                r'^\s*introduction\s*$',
                r'^\s*1\.?\s*引\s*言\s*$',
                r'^\s*引\s*言\s*$'
            ],
            'related_work': [
                r'^\s*\d+\.?\s*related\s+work\s*$',
                r'^\s*related\s+work\s*$',
                r'^\s*\d+\.?\s*literature\s+review\s*$',
                r'^\s*literature\s+review\s*$',
                r'^\s*\d+\.?\s*prior\s+work\s*$',
                r'^\s*\d+\.?\s*相\s*关\s*工\s*作\s*$',
                r'^\s*相\s*关\s*工\s*作\s*$'
            ],
            'background': [
                r'^\s*\d+\.?\s*background\s*$',
                r'^\s*background\s*$',
                r'^\s*\d+\.?\s*preliminaries\s*$',
                r'^\s*preliminaries\s*$',
                r'^\s*\d+\.?\s*problem\s+formulation\s*$',
                r'^\s*\d+\.?\s*背\s*景\s*$',
                r'^\s*背\s*景\s*$'
            ],
            'method': [
                r'^\s*\d+\.?\s*method[s]?\s*$',
                r'^\s*\d+\.?\s*approach\s*$',
                r'^\s*\d+\.?\s*methodology\s*$',
                r'^\s*\d+\.?\s*proposed\s+method\s*$',
                r'^\s*\d+\.?\s*model\s*$',
                r'^\s*\d+\.?\s*algorithm\s*$',
                r'^\s*方\s*法\s*$',
                r'^\s*\d+\.?\s*方\s*法\s*$'
            ],
            'implementation': [
                r'^\s*\d+\.?\s*implementation\s*$',
                r'^\s*implementation\s*$',
                r'^\s*\d+\.?\s*system\s*$',
                r'^\s*\d+\.?\s*architecture\s*$',
                r'^\s*\d+\.?\s*design\s*$',
                r'^\s*\d+\.?\s*实\s*现\s*$',
                r'^\s*实\s*现\s*$'
            ],
            'experiment': [
                r'^\s*\d+\.?\s*experiment[s]?\s*$',
                r'^\s*\d+\.?\s*evaluation\s*$',
                r'^\s*\d+\.?\s*experimental\s+setup\s*$',
                r'^\s*\d+\.?\s*实\s*验\s*$',
                r'^\s*实\s*验\s*$'
            ],
            'dataset': [
                r'^\s*\d+\.?\s*dataset[s]?\s*$',
                r'^\s*dataset[s]?\s*$',
                r'^\s*\d+\.?\s*data\s*$',
                r'^\s*\d+\.?\s*benchmark\s*$',
                r'^\s*\d+\.?\s*数\s*据\s*集\s*$',
                r'^\s*数\s*据\s*集\s*$'
            ],
            'result': [
                r'^\s*\d+\.?\s*result[s]?\s*$',
                r'^\s*\d+\.?\s*finding[s]?\s*$',
                r'^\s*\d+\.?\s*analysis\s*$',
                r'^\s*\d+\.?\s*performance\s*$',
                r'^\s*结\s*果\s*$',
                r'^\s*\d+\.?\s*结\s*果\s*$'
            ],
            'discussion': [
                r'^\s*\d+\.?\s*discussion\s*$',
                r'^\s*discussion\s*$',
                r'^\s*\d+\.?\s*analysis\s+and\s+discussion\s*$',
                r'^\s*\d+\.?\s*讨\s*论\s*$',
                r'^\s*讨\s*论\s*$'
            ],
            'limitation': [
                r'^\s*\d+\.?\s*limitation[s]?\s*$',
                r'^\s*limitation[s]?\s*$',
                r'^\s*\d+\.?\s*future\s+work\s*$',
                r'^\s*future\s+work\s*$',
                r'^\s*\d+\.?\s*局\s*限\s*性\s*$',
                r'^\s*局\s*限\s*性\s*$'
            ],
            'conclusion': [
                r'^\s*\d+\.?\s*conclusion[s]?\s*$',
                r'^\s*conclusion[s]?\s*$',
                r'^\s*\d+\.?\s*concluding\s+remarks\s*$',
                r'^\s*\d+\.?\s*summary\s+and\s+conclusion\s*$',
                r'^\s*结\s*论\s*$',
                r'^\s*\d+\.?\s*结\s*论\s*$'
            ],
            'acknowledgment': [
                r'^\s*acknowledgment[s]?\s*$',
                r'^\s*acknowledgement[s]?\s*$',
                r'^\s*致\s*谢\s*$'
            ],
            'appendix': [
                r'^\s*appendix\s*[a-z]?\s*$',
                r'^\s*[a-z]\.?\s*appendix\s*$',
                r'^\s*supplementary\s+material\s*$',
                r'^\s*附\s*录\s*$'
            ],
            'reference': [
                r'^\s*reference[s]?\s*$',
                r'^\s*bibliography\s*$',
                r'^\s*参\s*考\s*文\s*献\s*$'
            ]
        }
        
        # 编译正则表达式
        self.compiled_patterns = {}
        for section_type, patterns in self.section_patterns.items():
            self.compiled_patterns[section_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def extract_pdf_content(self, pdf_path: str, use_column_aware: bool = True) -> Optional[PDFContent]:
        """提取PDF完整内容"""
        try:
            print(f"📄 处理PDF: {Path(pdf_path).name}")
            
            # 打开PDF文档
            doc = fitz.open(pdf_path)
            
            # 提取基本信息
            metadata = doc.metadata
            total_pages = doc.page_count
            
            print(f"   📊 总页数: {total_pages}")
            
            # 提取全文内容 - 支持双栏感知
            full_text = ""
            page_texts = []

            for page_num in range(total_pages):
                page = doc[page_num]

                if use_column_aware:
                    # 使用双栏感知提取
                    page_text = self._extract_column_aware_text(page)
                else:
                    # 使用默认提取
                    page_text = page.get_text()

                page_texts.append((page_num + 1, page_text))
                full_text += page_text + "\n"
            
            doc.close()
            
            if not full_text.strip():
                print("   ❌ PDF内容为空")
                return None
            
            # 分析文档特征
            has_formulas = self._detect_formulas(full_text)
            has_tables = self._detect_tables(full_text)  
            has_figures = self._detect_figures(full_text)
            language = self._detect_language(full_text)
            total_words = len(full_text.split())
            
            print(f"   📝 总词数: {total_words}")
            print(f"   🧮 包含公式: {'是' if has_formulas else '否'}")
            print(f"   📊 包含表格: {'是' if has_tables else '否'}")  
            print(f"   🖼️ 包含图片: {'是' if has_figures else '否'}")
            print(f"   🌐 语言: {language}")
            
            # 提取标题
            title = self._extract_title(page_texts[0][1] if page_texts else "")
            print(f"   📋 标题: {title[:50]}...")
            
            # 提取摘要
            abstract = self._extract_abstract(full_text)
            print(f"   📄 摘要: {len(abstract)} 字符")
            
            # 智能章节分割
            sections = self._extract_sections(page_texts)
            print(f"   📑 识别章节: {len(sections)} 个")
            
            for section in sections:
                print(f"      - {section.section_type}: {section.title[:30]}... ({len(section.content)} 字符)")
            
            return PDFContent(
                title=title,
                abstract=abstract,
                sections=sections,
                total_pages=total_pages,
                total_words=total_words,
                has_formulas=has_formulas,
                has_tables=has_tables,
                has_figures=has_figures,
                language=language
            )
            
        except Exception as e:
            print(f"   ❌ PDF处理失败: {e}")
            return None
    
    def _extract_title(self, first_page_text: str) -> str:
        """提取论文标题"""
        lines = first_page_text.split('\n')
        
        # 寻找最可能的标题行
        for i, line in enumerate(lines[:20]):  # 只检查前20行
            line = line.strip()
            if len(line) > 10 and len(line) < 200:  # 标题长度合理
                # 排除常见的非标题内容
                if not any(exclude in line.lower() for exclude in 
                          ['arxiv:', 'doi:', 'page', 'abstract', 'www.', 'http']):
                    return line
        
        return "Unknown Title"
    
    def _extract_abstract(self, full_text: str) -> str:
        """提取论文摘要"""
        # 查找Abstract章节
        abstract_pattern = r'(?i)abstract\s*[:\-]?\s*\n(.*?)(?=\n\s*(?:1\.?\s*introduction|keywords|key\s*words|\n\s*\n))'
        
        match = re.search(abstract_pattern, full_text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            # 清理摘要内容
            abstract = re.sub(r'\s+', ' ', abstract)  # 规范化空白字符
            abstract = abstract.replace('\n', ' ')
            return abstract[:2000]  # 限制长度
        
        return ""
    
    def _extract_sections(self, page_texts: List[Tuple[int, str]]) -> List[PDFSection]:
        """智能提取章节内容"""
        sections = []
        full_text = "\n".join([text for _, text in page_texts])
        
        # 寻找章节分界点
        section_boundaries = []
        
        for page_num, page_text in page_texts:
            lines = page_text.split('\n')
            
            for line_num, line in enumerate(lines):
                line_clean = line.strip()
                if len(line_clean) < 3 or len(line_clean) > 100:
                    continue
                
                # 检查是否匹配章节模式
                for section_type, patterns in self.compiled_patterns.items():
                    for pattern in patterns:
                        if pattern.match(line_clean):
                            confidence = self._calculate_section_confidence(line_clean, section_type)
                            section_boundaries.append({
                                'page': page_num,
                                'line': line_num,
                                'title': line_clean,
                                'type': section_type,
                                'confidence': confidence
                            })
                            break
        
        # 按页面和行号排序
        section_boundaries.sort(key=lambda x: (x['page'], x['line']))
        
        # 提取章节内容
        for i, boundary in enumerate(section_boundaries):
            if boundary['confidence'] < 0.5:  # 过滤低置信度的识别
                continue
                
            # 确定章节内容范围
            start_page = boundary['page']
            start_line = boundary['line']
            
            if i + 1 < len(section_boundaries):
                end_page = section_boundaries[i + 1]['page']
                end_line = section_boundaries[i + 1]['line']
            else:
                end_page = page_texts[-1][0]
                end_line = float('inf')
            
            # 提取章节内容
            content = self._extract_section_content(
                page_texts, start_page, start_line, end_page, end_line
            )
            
            if content.strip():
                section = PDFSection(
                    title=boundary['title'],
                    content=content,
                    section_type=boundary['type'],
                    page_range=(start_page, end_page),
                    confidence=boundary['confidence']
                )
                sections.append(section)
        
        # ⭐ 添加fallback机制：处理未被章节覆盖的文本
        fallback_sections = self._extract_fallback_content(page_texts, sections)
        sections.extend(fallback_sections)

        # 按页面顺序重新排序所有章节
        sections.sort(key=lambda x: x.page_range[0])

        return sections
    
    def _extract_section_content(self, page_texts: List[Tuple[int, str]], 
                                start_page: int, start_line: int,
                                end_page: int, end_line: int) -> str:
        """提取指定范围的章节内容"""
        content_lines = []
        
        for page_num, page_text in page_texts:
            if page_num < start_page or page_num > end_page:
                continue
                
            lines = page_text.split('\n')
            
            for line_num, line in enumerate(lines):
                # 跳过章节标题行
                if page_num == start_page and line_num <= start_line:
                    continue
                if page_num == end_page and line_num >= end_line:
                    break
                    
                content_lines.append(line)
        
        content = '\n'.join(content_lines)
        
        # 清理内容
        content = self._clean_text_content(content)
        
        return content
    
    def _clean_text_content(self, text: str) -> str:
        """清理文本内容"""
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除页码和页眉页脚
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # 移除过短的行（可能是格式噪声）
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 3:  # 保留有意义的内容
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _calculate_section_confidence(self, line: str, section_type: str) -> float:
        """计算章节识别置信度"""
        confidence = 0.5  # 基础置信度
        
        # 如果有数字编号，增加置信度
        if re.match(r'^\s*\d+\.?\s*', line):
            confidence += 0.2
        
        # 如果格式规整，增加置信度
        if line.isupper() or line.istitle():
            confidence += 0.1
        
        # 根据章节类型调整
        if section_type == 'abstract' and 'abstract' in line.lower():
            confidence += 0.3
        elif section_type == 'reference' and any(word in line.lower() 
                                               for word in ['reference', 'bibliography']):
            confidence += 0.3
        
        return min(confidence, 1.0)

    def _extract_fallback_content(self, page_texts: List[Tuple[int, str]],
                                 sections: List[PDFSection]) -> List[PDFSection]:
        """提取未被章节覆盖的内容作为fallback段落"""
        fallback_sections = []

        # 1. 构建已覆盖区域的映射
        covered_ranges = []
        for section in sections:
            covered_ranges.append({
                'start_page': section.page_range[0],
                'end_page': section.page_range[1],
                'type': section.section_type
            })

        # 按页面排序
        covered_ranges.sort(key=lambda x: x['start_page'])

        # 2. 识别未覆盖的区域
        uncovered_regions = []
        total_pages = len(page_texts)

        if not covered_ranges:
            # 如果没有识别到任何章节，整个文档都是未覆盖区域
            uncovered_regions.append({
                'start_page': page_texts[0][0] if page_texts else 1,
                'end_page': page_texts[-1][0] if page_texts else 1,
                'reason': 'no_sections_detected'
            })
        else:
            # 检查第一个章节前的内容
            first_section = covered_ranges[0]
            if first_section['start_page'] > page_texts[0][0]:
                uncovered_regions.append({
                    'start_page': page_texts[0][0],
                    'end_page': first_section['start_page'] - 1,
                    'reason': 'before_first_section'
                })

            # 检查章节之间的空隙
            for i in range(len(covered_ranges) - 1):
                current_end = covered_ranges[i]['end_page']
                next_start = covered_ranges[i + 1]['start_page']

                if next_start > current_end + 1:  # 有空隙
                    uncovered_regions.append({
                        'start_page': current_end + 1,
                        'end_page': next_start - 1,
                        'reason': 'between_sections'
                    })

            # 检查最后一个章节后的内容
            last_section = covered_ranges[-1]
            if last_section['end_page'] < page_texts[-1][0]:
                uncovered_regions.append({
                    'start_page': last_section['end_page'] + 1,
                    'end_page': page_texts[-1][0],
                    'reason': 'after_last_section'
                })

        # 3. 为每个未覆盖区域创建fallback section
        fallback_counter = 1
        for region in uncovered_regions:
            content = self._extract_fallback_region_content(
                page_texts, region['start_page'], region['end_page']
            )

            if content.strip() and len(content.strip()) > 100:  # 只保留有意义的内容
                # 智能推断内容类型
                inferred_type = self._infer_content_type(content, region['reason'])

                fallback_section = PDFSection(
                    title=f"Content Block {fallback_counter}",
                    content=content,
                    section_type=inferred_type,
                    page_range=(region['start_page'], region['end_page']),
                    confidence=0.3  # 低置信度标记为fallback内容
                )
                fallback_sections.append(fallback_section)
                fallback_counter += 1

        return fallback_sections

    def _extract_fallback_region_content(self, page_texts: List[Tuple[int, str]],
                                        start_page: int, end_page: int) -> str:
        """提取指定页面范围的内容"""
        content_lines = []

        for page_num, page_text in page_texts:
            if start_page <= page_num <= end_page:
                lines = page_text.split('\n')

                for line in lines:
                    line_clean = line.strip()
                    # 过滤噪声内容
                    if len(line_clean) > 5 and not self._is_noise_line(line_clean):
                        content_lines.append(line_clean)

        content = '\n'.join(content_lines)
        return self._clean_text_content(content)

    def _infer_content_type(self, content: str, reason: str) -> str:
        """智能推断fallback内容的类型"""
        content_lower = content.lower()

        # 基于位置推断
        if reason == 'before_first_section':
            if 'abstract' in content_lower or 'summary' in content_lower:
                return 'abstract'
            elif 'introduction' in content_lower:
                return 'introduction'
            else:
                return 'header_content'

        elif reason == 'after_last_section':
            if 'reference' in content_lower or 'bibliography' in content_lower:
                return 'reference'
            elif 'appendix' in content_lower:
                return 'appendix'
            else:
                return 'footer_content'

        else:  # between_sections
            # 基于内容特征推断
            if any(keyword in content_lower for keyword in ['figure', 'table', 'algorithm']):
                return 'content'
            elif len(content.split()) < 50:  # 短内容可能是标题或过渡
                return 'transition'
            else:
                return 'content'

    def _is_noise_line(self, line: str) -> bool:
        """判断是否为噪声行（页码、页眉页脚等）"""
        line_lower = line.lower()

        # 常见的噪声模式
        noise_patterns = [
            r'^\d+$',  # 纯数字（页码）
            r'^page \d+',  # "page 1"
            r'^\d+\s*/\s*\d+$',  # "1/10"
            r'^[a-z\s]{1,3}$',  # 极短的字母组合
            r'^[\W\s]*$',  # 纯标点符号或空白
        ]

        for pattern in noise_patterns:
            if re.match(pattern, line_lower):
                return True

        # 检查是否为页眉页脚常见内容
        if any(keyword in line_lower for keyword in ['arxiv:', 'doi:', 'www.', 'http', '©', 'copyright']):
            return True

        return False

    def _extract_column_aware_text(self, page) -> str:
        """双栏感知的文本提取"""

        # 获取页面尺寸
        page_rect = page.rect
        page_width = page_rect.width

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

        if len(text_blocks) < 4:
            # 文本块太少，使用默认提取
            return page.get_text()

        # 检测双栏布局
        is_two_column, column_divider = self._detect_two_column_layout(text_blocks, page_width)

        if is_two_column:
            return self._reorder_two_column_text(text_blocks, column_divider)
        else:
            # 单栏布局，按y坐标排序
            text_blocks.sort(key=lambda b: b['y0'])
            return '\n\n'.join(block['text'] for block in text_blocks)

    def _detect_two_column_layout(self, text_blocks: List[Dict], page_width: float) -> Tuple[bool, float]:
        """检测是否为双栏布局并找到分栏位置"""

        if len(text_blocks) < 6:  # 文本块太少
            return False, 0.0

        # 分析文本块的x坐标分布
        center_xs = [block['center_x'] for block in text_blocks]

        # 计算可能的分栏位置
        center_xs.sort()

        # 寻找最大的间隙作为潜在的列分割
        max_gap = 0
        best_divider = page_width / 2

        for i in range(len(center_xs) - 1):
            gap = center_xs[i + 1] - center_xs[i]
            if gap > max_gap:
                max_gap = gap
                best_divider = (center_xs[i] + center_xs[i + 1]) / 2

        # 判断是否为双栏
        if max_gap > page_width * 0.08:  # 间隙占页面宽度的8%以上
            # 验证左右分布是否平衡
            left_blocks = sum(1 for x in center_xs if x < best_divider)
            right_blocks = sum(1 for x in center_xs if x >= best_divider)

            if left_blocks >= 2 and right_blocks >= 2:
                balance_ratio = min(left_blocks, right_blocks) / max(left_blocks, right_blocks)
                if balance_ratio > 0.3:  # 左右相对平衡
                    return True, best_divider

        return False, 0.0

    def _reorder_two_column_text(self, text_blocks: List[Dict], column_divider: float) -> str:
        """重新排序双栏文本，按阅读顺序"""

        # 分离左右栏
        left_blocks = [b for b in text_blocks if b['center_x'] < column_divider]
        right_blocks = [b for b in text_blocks if b['center_x'] >= column_divider]

        # 按y坐标排序
        left_blocks.sort(key=lambda b: b['y0'])
        right_blocks.sort(key=lambda b: b['y0'])

        # 智能交替合并 - 基于y坐标匹配
        result_text = []
        left_idx = 0
        right_idx = 0

        while left_idx < len(left_blocks) or right_idx < len(right_blocks):
            left_available = left_idx < len(left_blocks)
            right_available = right_idx < len(right_blocks)

            if not left_available:
                # 左栏用完，添加右栏剩余
                result_text.append(right_blocks[right_idx]['text'])
                right_idx += 1
            elif not right_available:
                # 右栏用完，添加左栏剩余
                result_text.append(left_blocks[left_idx]['text'])
                left_idx += 1
            else:
                # 两栏都有内容，选择y坐标更小的
                left_y = left_blocks[left_idx]['y0']
                right_y = right_blocks[right_idx]['y0']

                # 使用一定的阈值来避免频繁切换
                y_threshold = 20  # 20个单位的阈值

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

    def _detect_formulas(self, text: str) -> bool:
        """检测文档是否包含数学公式"""
        formula_indicators = [
            r'\$.*\$',  # LaTeX公式
            r'\\[a-zA-Z]+\{',  # LaTeX命令
            r'∑|∏|∫|∂|∇|≈|≤|≥|±|α|β|γ|δ|ε|θ|λ|μ|π|σ|φ|ψ|ω',  # 数学符号
            r'[a-zA-Z]\s*[=]\s*[a-zA-Z0-9\+\-\*/\(\)]+',  # 简单等式
        ]
        
        for pattern in formula_indicators:
            if re.search(pattern, text):
                return True
        return False
    
    def _detect_tables(self, text: str) -> bool:
        """检测文档是否包含表格"""
        table_indicators = [
            r'table\s+\d+',
            r'表\s*\d+',
            r'\|.*\|.*\|',  # 简单的表格格式
            r'(\w+\s+){3,}\n(\w+\s+){3,}',  # 列对齐的数据
        ]
        
        for pattern in table_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _detect_figures(self, text: str) -> bool:
        """检测文档是否包含图片"""
        figure_indicators = [
            r'figure\s+\d+',
            r'fig\.\s*\d+',
            r'图\s*\d+',
            r'illustration\s+\d+',
        ]
        
        for pattern in figure_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _detect_language(self, text: str) -> str:
        """检测文档主要语言"""
        # 简单的语言检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        
        if chinese_chars > english_words * 0.5:
            return 'zh'
        else:
            return 'en'

def test_pdf_processor():
    """测试PDF处理器"""
    processor = AcademicPDFProcessor()
    
    # 测试一个PDF文件
    pdf_files = list(Path("data/raw_papers").glob("*.pdf"))
    if pdf_files:
        test_file = pdf_files[0]
        print(f"🧪 测试PDF处理器: {test_file.name}")
        
        content = processor.extract_pdf_content(str(test_file))
        if content:
            print(f"✅ 处理成功!")
            print(f"   标题: {content.title}")
            print(f"   章节数: {len(content.sections)}")
            print(f"   总页数: {content.total_pages}")
            print(f"   总词数: {content.total_words}")
        else:
            print("❌ 处理失败")
    else:
        print("❌ 未找到PDF文件")

if __name__ == "__main__":
    test_pdf_processor()