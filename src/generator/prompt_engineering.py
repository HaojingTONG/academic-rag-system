"""
提示词工程模块
实现高级的上下文增强与提示词构建功能
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re

class QueryType(Enum):
    """查询类型枚举"""
    GENERAL = "general"           # 一般性查询
    TECHNICAL = "technical"       # 技术性查询  
    COMPARISON = "comparison"     # 对比性查询
    DEFINITION = "definition"     # 定义性查询
    EXPLANATION = "explanation"   # 解释性查询
    SUMMARIZATION = "summarization" # 总结性查询
    APPLICATION = "application"   # 应用性查询

class ContextType(Enum):
    """上下文类型"""
    ACADEMIC_PAPER = "academic_paper"    # 学术论文
    TECHNICAL_DOC = "technical_doc"      # 技术文档
    OVERVIEW = "overview"                # 概述文档
    COMPARISON = "comparison"            # 对比文档

@dataclass
class ContextItem:
    """上下文项"""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    source_type: ContextType
    priority: int = 0  # 优先级，用于排序
    
@dataclass
class PromptTemplate:
    """提示词模板"""
    name: str
    template: str
    query_types: List[QueryType]
    context_limit: int = 5
    max_context_length: int = 2000
    language: str = "zh"  # 输出语言
    
class ContextProcessor:
    """上下文处理器"""
    
    def __init__(self):
        """初始化上下文处理器"""
        print("🔧 初始化上下文处理器")
        
    def analyze_query_type(self, query: str) -> QueryType:
        """分析查询类型"""
        query_lower = query.lower()
        
        # 定义性查询
        if any(keyword in query_lower for keyword in ['what is', 'define', 'definition', '什么是', '定义', '含义']):
            return QueryType.DEFINITION
        
        # 对比性查询    
        elif any(keyword in query_lower for keyword in ['compare', 'difference', 'vs', 'versus', '比较', '区别', '差异']):
            return QueryType.COMPARISON
        
        # 解释性查询
        elif any(keyword in query_lower for keyword in ['explain', 'how', 'why', 'describe', '解释', '如何', '为什么', '描述']):
            return QueryType.EXPLANATION
        
        # 总结性查询
        elif any(keyword in query_lower for keyword in ['summarize', 'summary', 'overview', '总结', '概述', '综述']):
            return QueryType.SUMMARIZATION
        
        # 应用性查询
        elif any(keyword in query_lower for keyword in ['application', 'use case', 'implement', '应用', '实现', '使用']):
            return QueryType.APPLICATION
        
        # 技术性查询
        elif any(keyword in query_lower for keyword in ['algorithm', 'method', 'technique', '算法', '方法', '技术', '实现']):
            return QueryType.TECHNICAL
        
        # 默认为一般性查询
        else:
            return QueryType.GENERAL
    
    def classify_context_type(self, metadata: Dict[str, Any]) -> ContextType:
        """分类上下文类型"""
        # 根据元数据判断文档类型
        section_type = metadata.get('section_type', '').lower()
        title = metadata.get('title', '').lower()
        
        if 'abstract' in section_type or 'introduction' in section_type:
            return ContextType.OVERVIEW
        elif 'method' in section_type or 'algorithm' in section_type:
            return ContextType.TECHNICAL_DOC
        elif 'comparison' in title or 'vs' in title:
            return ContextType.COMPARISON
        else:
            return ContextType.ACADEMIC_PAPER
    
    def calculate_context_priority(self, 
                                   context_item: ContextItem, 
                                   query_type: QueryType) -> int:
        """计算上下文优先级"""
        base_priority = int(context_item.similarity_score * 100)
        
        # 根据查询类型和上下文类型的匹配度调整优先级
        type_bonus = 0
        if query_type == QueryType.TECHNICAL and context_item.source_type == ContextType.TECHNICAL_DOC:
            type_bonus = 20
        elif query_type == QueryType.DEFINITION and context_item.source_type == ContextType.OVERVIEW:
            type_bonus = 15
        elif query_type == QueryType.COMPARISON and context_item.source_type == ContextType.COMPARISON:
            type_bonus = 25
        
        # 根据章节类型调整优先级
        section_type = context_item.metadata.get('section_type', '')
        section_bonus = 0
        if query_type == QueryType.DEFINITION and 'abstract' in section_type.lower():
            section_bonus = 10
        elif query_type == QueryType.TECHNICAL and 'method' in section_type.lower():
            section_bonus = 15
        
        return base_priority + type_bonus + section_bonus
    
    def process_contexts(self, 
                        retrieved_results: List[Any], 
                        query: str, 
                        max_contexts: int = 5) -> List[ContextItem]:
        """处理检索到的上下文"""
        query_type = self.analyze_query_type(query)
        print(f"🔍 查询类型分析: {query_type.value}")
        
        context_items = []
        
        for result in retrieved_results:
            # 处理不同类型的结果
            if hasattr(result, 'document'):  # EnhancedVectorRetrieval结果
                content = result.document
                metadata = result.metadata
                similarity = result.similarity_score
            elif isinstance(result, dict) and 'content' in result:  # 字典格式结果
                content = result['content']
                metadata = result.get('metadata', {})
                similarity = result.get('similarity_score', 0.0)
            else:  # 其他格式
                continue
            
            # 创建上下文项
            source_type = self.classify_context_type(metadata)
            context_item = ContextItem(
                content=content,
                metadata=metadata,
                similarity_score=similarity,
                source_type=source_type
            )
            
            # 计算优先级
            context_item.priority = self.calculate_context_priority(context_item, query_type)
            context_items.append(context_item)
        
        # 按优先级排序
        context_items.sort(key=lambda x: x.priority, reverse=True)
        
        # 限制数量
        context_items = context_items[:max_contexts]
        
        print(f"📊 上下文处理结果:")
        for i, item in enumerate(context_items, 1):
            title = item.metadata.get('title', 'Unknown')[:40] + "..."
            print(f"   {i}. 优先级: {item.priority} | 相似度: {item.similarity_score:.3f}")
            print(f"      类型: {item.source_type.value} | 论文: {title}")
        
        return context_items

class PromptTemplateManager:
    """提示词模板管理器"""
    
    def __init__(self):
        """初始化模板管理器"""
        self.templates = {}
        self._initialize_templates()
        print(f"📝 初始化提示词模板管理器，加载 {len(self.templates)} 个模板")
    
    def _initialize_templates(self):
        """初始化默认模板"""
        
        # 通用查询模板
        self.templates["general"] = PromptTemplate(
            name="通用查询模板",
            query_types=[QueryType.GENERAL],
            template="""请根据以下学术资料回答用户的问题。

问题：{query}

学术资料：
{context}

要求：
1. 基于提供的学术资料进行回答
2. 保持学术严谨性和准确性
3. 如果资料中信息不足，请明确说明
4. 使用中文回答

回答："""
        )
        
        # 技术性查询模板
        self.templates["technical"] = PromptTemplate(
            name="技术查询模板",
            query_types=[QueryType.TECHNICAL],
            template="""请基于以下学术资料回答技术问题。

技术问题：{query}

相关资料：
{context}

请提供详细的技术回答，包括：
1. 核心技术原理或方法
2. 具体实现细节（如果资料中有提及）
3. 技术优势和局限性
4. 相关的学术引用

如果资料中技术细节不够详细，请说明需要更多信息。

技术解答："""
        )
        
        # 定义性查询模板
        self.templates["definition"] = PromptTemplate(
            name="定义查询模板", 
            query_types=[QueryType.DEFINITION],
            template="""请根据以下学术资料为用户提供准确的定义。

需要定义的概念：{query}

参考资料：
{context}

请提供：
1. 清晰准确的定义
2. 关键特征或组成部分
3. 学术背景或起源（如果资料中有提及）
4. 相关概念的区别（如果适用）

如果资料中没有足够信息提供完整定义，请说明。

定义解答："""
        )
        
        # 对比性查询模板
        self.templates["comparison"] = PromptTemplate(
            name="对比查询模板",
            query_types=[QueryType.COMPARISON], 
            template="""请基于以下学术资料进行对比分析。

对比问题：{query}

参考资料：
{context}

请从以下角度进行对比分析：
1. 主要差异和相似点
2. 各自的优势和劣势
3. 适用场景和条件
4. 性能或效果对比（如果资料中有数据）

请保持客观中立，基于资料中的事实进行分析。

对比分析："""
        )
        
        # 解释性查询模板
        self.templates["explanation"] = PromptTemplate(
            name="解释查询模板",
            query_types=[QueryType.EXPLANATION],
            template="""请根据以下学术资料对用户的问题进行详细解释。

问题：{query}

学术资料：
{context}

请提供详细解释，包括：
1. 基本概念和原理
2. 工作机制或过程
3. 重要性和影响
4. 实际应用或例子（如果资料中有提及）

请确保解释清晰易懂，同时保持学术准确性。

详细解释："""
        )
        
        # 总结性查询模板  
        self.templates["summarization"] = PromptTemplate(
            name="总结查询模板",
            query_types=[QueryType.SUMMARIZATION],
            template="""请基于以下学术资料对用户询问的主题进行综合总结。

总结主题：{query}

相关资料：
{context}

请提供全面的总结，包括：
1. 主要概念和理论
2. 关键发现和结论
3. 重要方法和技术
4. 发展趋势和未来方向（如果资料中有讨论）

请确保总结全面、准确、逻辑清晰。

综合总结："""
        )
        
        # 应用性查询模板
        self.templates["application"] = PromptTemplate(
            name="应用查询模板",
            query_types=[QueryType.APPLICATION],
            template="""请根据以下学术资料回答关于实际应用的问题。

应用问题：{query}

参考资料：
{context}

请重点说明：
1. 实际应用场景和领域
2. 具体实现方案或步骤
3. 应用效果和案例
4. 实施中的注意事项和挑战

如果资料中应用信息有限，请基于理论推导可能的应用。

应用解答："""
        )
    
    def get_template(self, query_type: QueryType) -> PromptTemplate:
        """获取适合的模板"""
        # 查找匹配的模板
        for template in self.templates.values():
            if query_type in template.query_types:
                return template
        
        # 默认返回通用模板
        return self.templates["general"]
    
    def add_custom_template(self, template: PromptTemplate):
        """添加自定义模板"""
        self.templates[template.name] = template
        print(f"✅ 添加自定义模板: {template.name}")

class PromptBuilder:
    """提示词构建器"""
    
    def __init__(self):
        """初始化提示词构建器"""
        self.context_processor = ContextProcessor()
        self.template_manager = PromptTemplateManager()
        print("🏗️ 提示词构建器初始化完成")
    
    def build_context_section(self, 
                            context_items: List[ContextItem], 
                            max_length: int = 2000) -> str:
        """构建上下文部分"""
        context_parts = []
        current_length = 0
        
        for i, item in enumerate(context_items, 1):
            # 构建单个上下文条目
            title = item.metadata.get('title', f'资料{i}')
            section_type = item.metadata.get('section_type', '内容')
            authors = item.metadata.get('authors', '')
            
            # 格式化来源信息
            source_info = f"【资料{i}】{title}"
            if authors:
                source_info += f" (作者: {authors[:50]}{'...' if len(authors) > 50 else ''})"
            if section_type and section_type != 'content':
                source_info += f" [{section_type}]"
            
            # 清理和截断内容
            content = self._clean_content(item.content)
            
            # 构建条目
            entry = f"{source_info}\n{content}\n"
            
            # 检查长度限制
            if current_length + len(entry) > max_length and context_parts:
                break
            
            context_parts.append(entry)
            current_length += len(entry)
        
        return "\n".join(context_parts)
    
    def _clean_content(self, content: str) -> str:
        """清理内容文本"""
        # 移除多余的空白字符
        content = re.sub(r'\s+', ' ', content.strip())
        
        # 移除特殊字符
        content = re.sub(r'[^\w\s\u4e00-\u9fff.,;:!?()[\]{}-]', '', content)
        
        # 限制长度
        if len(content) > 800:
            content = content[:800] + "..."
        
        return content
    
    def build_prompt(self, 
                    query: str, 
                    retrieved_results: List[Any],
                    custom_template: Optional[str] = None,
                    max_context_length: int = 2000,
                    context_limit: int = 5) -> Dict[str, Any]:
        """构建完整的提示词"""
        
        print(f"🏗️ 构建提示词...")
        print(f"   - 查询: \"{query[:50]}{'...' if len(query) > 50 else ''}\"")
        print(f"   - 检索结果数: {len(retrieved_results)}")
        
        # 1. 处理上下文
        context_items = self.context_processor.process_contexts(
            retrieved_results, query, max_contexts=context_limit
        )
        
        # 2. 分析查询类型
        query_type = self.context_processor.analyze_query_type(query)
        
        # 3. 获取模板
        if custom_template:
            template_str = custom_template
            template_name = "自定义模板"
        else:
            template = self.template_manager.get_template(query_type)
            template_str = template.template
            template_name = template.name
        
        # 4. 构建上下文部分
        context_section = self.build_context_section(context_items, max_context_length)
        
        # 5. 填充模板
        try:
            final_prompt = template_str.format(
                query=query,
                context=context_section
            )
        except KeyError as e:
            print(f"⚠️ 模板格式错误: {e}")
            # 使用通用格式
            final_prompt = f"问题: {query}\n\n上下文:\n{context_section}\n\n请回答:"
        
        # 6. 构建返回结果
        result = {
            "prompt": final_prompt,
            "query": query,
            "query_type": query_type.value,
            "template_name": template_name,
            "context_items": len(context_items),
            "context_length": len(context_section),
            "prompt_length": len(final_prompt),
            "metadata": {
                "sources": [
                    {
                        "title": item.metadata.get('title', 'Unknown'),
                        "similarity": item.similarity_score,
                        "priority": item.priority,
                        "type": item.source_type.value
                    }
                    for item in context_items
                ]
            }
        }
        
        # 7. 输出构建统计
        print(f"✅ 提示词构建完成:")
        print(f"   - 模板: {template_name}")
        print(f"   - 查询类型: {query_type.value}")
        print(f"   - 上下文数量: {len(context_items)} 个")
        print(f"   - 上下文长度: {len(context_section)} 字符")
        print(f"   - 最终提示词长度: {len(final_prompt)} 字符")
        
        return result
    
    def get_template_info(self) -> Dict[str, Any]:
        """获取模板信息"""
        return {
            "available_templates": list(self.template_manager.templates.keys()),
            "query_types": [qt.value for qt in QueryType],
            "context_types": [ct.value for ct in ContextType]
        }

# 便捷函数
def create_prompt_builder() -> PromptBuilder:
    """创建提示词构建器的便捷函数"""
    return PromptBuilder()

def analyze_query_type(query: str) -> str:
    """分析查询类型的便捷函数"""
    processor = ContextProcessor()
    return processor.analyze_query_type(query).value