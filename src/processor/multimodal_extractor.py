# src/processor/multimodal_extractor.py
import re
import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Optional
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt

class MultimodalExtractor:
    def __init__(self):
        self.figure_patterns = [
            r'Figure\s+\d+',
            r'Fig\.\s*\d+',
            r'图\s*\d+',
        ]
        
        self.table_patterns = [
            r'Table\s+\d+',
            r'表\s*\d+',
        ]
        
        self.formula_patterns = [
            r'\$\$.*?\$\$',  # LaTeX display math
            r'\$.*?\$',      # LaTeX inline math
            r'\\begin\{equation\}.*?\\end\{equation\}',
            r'\\begin\{align\}.*?\\end\{align\}',
        ]
    
    def extract_multimodal_content(self, pdf_path: str) -> Dict:
        """提取PDF中的多模态内容"""
        try:
            doc = fitz.open(pdf_path)
            
            multimodal_content = {
                'figures': [],
                'tables': [],
                'formulas': [],
                'code_blocks': [],
                'text_content': ""
            }
            
            full_text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # 提取文本
                page_text = page.get_text()
                full_text += page_text + "\n"
                
                # 提取图像
                figures = self._extract_figures_from_page(page, page_num)
                multimodal_content['figures'].extend(figures)
                
                # 提取表格
                tables = self._extract_tables_from_page(page, page_num)
                multimodal_content['tables'].extend(tables)
            
            doc.close()
            
            # 从文本中提取公式和代码
            multimodal_content['formulas'] = self._extract_formulas(full_text)
            multimodal_content['code_blocks'] = self._extract_code_blocks(full_text)
            multimodal_content['text_content'] = full_text
            
            return multimodal_content
            
        except Exception as e:
            print(f"多模态内容提取错误: {e}")
            return {
                'figures': [],
                'tables': [],
                'formulas': [],
                'code_blocks': [],
                'text_content': ""
            }
    
    def _extract_figures_from_page(self, page, page_num: int) -> List[Dict]:
        """从页面提取图片"""
        figures = []
        
        # 获取页面中的图像
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                # 获取图像数据
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                if pix.n - pix.alpha < 4:  # 确保不是CMYK
                    # 转换为PNG格式
                    img_data = pix.tobytes("png")
                    
                    # 查找图像周围的描述文字
                    caption = self._find_figure_caption(page.get_text(), page_num, img_index)
                    
                    figure_info = {
                        'page': page_num,
                        'index': img_index,
                        'caption': caption,
                        'image_data': base64.b64encode(img_data).decode('utf-8'),
                        'format': 'png',
                        'width': pix.width,
                        'height': pix.height
                    }
                    
                    figures.append(figure_info)
                
                pix = None
                
            except Exception as e:
                print(f"图像提取错误: {e}")
                continue
        
        return figures
    
    def _extract_tables_from_page(self, page, page_num: int) -> List[Dict]:
        """从页面提取表格"""
        tables = []
        
        try:
            # 尝试提取表格结构
            tabs = page.find_tables()
            
            for tab_index, tab in enumerate(tabs):
                # 提取表格数据
                table_data = tab.extract()
                
                # 查找表格标题
                caption = self._find_table_caption(page.get_text(), page_num, tab_index)
                
                table_info = {
                    'page': page_num,
                    'index': tab_index,
                    'caption': caption,
                    'data': table_data,
                    'rows': len(table_data),
                    'cols': len(table_data[0]) if table_data else 0
                }
                
                tables.append(table_info)
                
        except Exception as e:
            print(f"表格提取错误: {e}")
        
        return tables
    
    def _extract_formulas(self, text: str) -> List[Dict]:
        """提取数学公式"""
        formulas = []
        
        for pattern in self.formula_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            
            for match in matches:
                formula_text = match.group(0)
                
                # 清理公式文本
                clean_formula = self._clean_formula(formula_text)
                
                if len(clean_formula) > 5:  # 过滤过短的匹配
                    formula_info = {
                        'raw_text': formula_text,
                        'clean_text': clean_formula,
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'type': self._classify_formula_type(clean_formula)
                    }
                    
                    formulas.append(formula_info)
        
        return formulas
    
    def _extract_code_blocks(self, text: str) -> List[Dict]:
        """提取代码块"""
        code_blocks = []
        
        # 匹配代码块模式
        code_patterns = [
            r'```(\w*)\n(.*?)```',  # Markdown风格
            r'```(.*?)```',         # 简单代码块
            r'\bdef\s+\w+\([^)]*\):.*?(?=\n\n|\Z)',  # Python函数
            r'\bclass\s+\w+.*?:.*?(?=\n\n|\Z)',     # Python类
            r'\b(?:import|from)\s+[\w.]+.*?(?=\n)',  # Python导入
        ]
        
        for pattern in code_patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
            
            for match in matches:
                if len(match.groups()) >= 2:
                    language = match.group(1)
                    code_text = match.group(2)
                else:
                    language = 'unknown'
                    code_text = match.group(0)
                
                if len(code_text.strip()) > 10:  # 过滤过短的代码
                    code_info = {
                        'language': language,
                        'code': code_text.strip(),
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'line_count': len(code_text.split('\n'))
                    }
                    
                    code_blocks.append(code_info)
        
        return code_blocks
    
    def _find_figure_caption(self, page_text: str, page_num: int, img_index: int) -> str:
        """查找图片标题"""
        for pattern in self.figure_patterns:
            matches = re.finditer(pattern + r'[:\.]?\s*([^.]*\.)', page_text, re.IGNORECASE)
            for match in matches:
                return match.group(0).strip()
        
        return f"Figure on page {page_num + 1}"
    
    def _find_table_caption(self, page_text: str, page_num: int, tab_index: int) -> str:
        """查找表格标题"""
        for pattern in self.table_patterns:
            matches = re.finditer(pattern + r'[:\.]?\s*([^.]*\.)', page_text, re.IGNORECASE)
            for match in matches:
                return match.group(0).strip()
        
        return f"Table on page {page_num + 1}"
    
    def _clean_formula(self, formula_text: str) -> str:
        """清理公式文本"""
        # 移除LaTeX标记
        clean_text = re.sub(r'\$+', '', formula_text)
        clean_text = re.sub(r'\\begin\{.*?\}|\\end\{.*?\}', '', clean_text)
        clean_text = clean_text.strip()
        
        return clean_text
    
    def _classify_formula_type(self, formula: str) -> str:
        """分类公式类型"""
        if any(op in formula for op in ['\\sum', '\\int', '\\prod']):
            return 'mathematical_operation'
        elif any(func in formula for func in ['\\frac', '\\sqrt', '^', '_']):
            return 'mathematical_expression'
        elif any(symbol in formula for func in ['\\alpha', '\\beta', '\\gamma']):
            return 'statistical_formula'
        else:
            return 'general_formula'


# 使用示例
def test_multimodal_extractor():
    extractor = MultimodalExtractor()
    
    # 假设你有一个PDF文件
    # content = extractor.extract_multimodal_content("sample_paper.pdf")
    
    # 测试公式提取
    sample_text = """
    The loss function is defined as:
    $$L = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2$$
    
    Where $N$ is the batch size and $\\alpha$ is the learning rate.
    """
    
    formulas = extractor._extract_formulas(sample_text)
    print("提取的公式:")
    for formula in formulas:
        print(f"- {formula['clean_text']} (类型: {formula['type']})")

if __name__ == "__main__":
    test_multimodal_extractor()