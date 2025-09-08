# src/processor/pdf_processor.py
"""
PDF全文处理模块 - 提取学术论文的完整内容
支持智能章节识别、内容清理和结构化处理
"""

import fitz  # PyMuPDF
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging

@dataclass
class PDFSection:
    """PDF章节数据结构"""
    title: str
    content: str
    section_type: str  # abstract, introduction, method, experiment, result, conclusion, reference
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
            'method': [
                r'^\s*\d+\.?\s*method[s]?\s*$',
                r'^\s*\d+\.?\s*approach\s*$',
                r'^\s*\d+\.?\s*methodology\s*$',
                r'^\s*方\s*法\s*$',
                r'^\s*\d+\.?\s*方\s*法\s*$'
            ],
            'experiment': [
                r'^\s*\d+\.?\s*experiment[s]?\s*$',
                r'^\s*\d+\.?\s*evaluation\s*$',
                r'^\s*\d+\.?\s*实\s*验\s*$',
                r'^\s*实\s*验\s*$'
            ],
            'result': [
                r'^\s*\d+\.?\s*result[s]?\s*$',
                r'^\s*\d+\.?\s*finding[s]?\s*$',
                r'^\s*结\s*果\s*$',
                r'^\s*\d+\.?\s*结\s*果\s*$'
            ],
            'conclusion': [
                r'^\s*\d+\.?\s*conclusion[s]?\s*$',
                r'^\s*\d+\.?\s*discussion\s*$',
                r'^\s*结\s*论\s*$',
                r'^\s*\d+\.?\s*结\s*论\s*$'
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
    
    def extract_pdf_content(self, pdf_path: str) -> Optional[PDFContent]:
        """提取PDF完整内容"""
        try:
            print(f"📄 处理PDF: {Path(pdf_path).name}")
            
            # 打开PDF文档
            doc = fitz.open(pdf_path)
            
            # 提取基本信息
            metadata = doc.metadata
            total_pages = doc.page_count
            
            print(f"   📊 总页数: {total_pages}")
            
            # 提取全文内容
            full_text = ""
            page_texts = []
            
            for page_num in range(total_pages):
                page = doc[page_num]
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