# src/generator/quality_enhancement.py
"""
生成质量提升模块
实现上下文优化、Prompt工程、事实核查、引用规范等功能
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import hashlib

@dataclass
class ContextWindow:
    """上下文窗口"""
    content: str
    source_papers: List[str]
    relevance_score: float
    confidence_level: float
    total_tokens: int

@dataclass
class Citation:
    """引用信息"""
    paper_id: str
    title: str
    authors: List[str]
    year: str
    content_snippet: str
    relevance_score: float

@dataclass
class FactCheckResult:
    """事实核查结果"""
    claim: str
    support_level: str  # strong, moderate, weak, contradicted
    supporting_sources: List[str]
    contradicting_sources: List[str]
    confidence_score: float

class ContextOptimizer:
    """上下文优化器 - 动态调整上下文长度和相关性"""
    
    def __init__(self, max_tokens=2048, min_relevance=0.3):
        self.max_tokens = max_tokens
        self.min_relevance = min_relevance
        self.token_weights = {
            'title': 2.0,
            'abstract': 1.5, 
            'methodology': 1.8,
            'results': 1.2,
            'conclusion': 1.3,
            'content': 1.0
        }
    
    def optimize_context(self, retrieved_results: List[Dict], query: str) -> ContextWindow:
        """优化上下文窗口"""
        
        # 1. 计算查询相关性分数
        scored_results = self._score_relevance(retrieved_results, query)
        
        # 2. 过滤低相关性内容
        filtered_results = [r for r in scored_results if r['relevance_score'] >= self.min_relevance]
        
        # 3. 动态调整上下文长度
        optimized_context = self._build_adaptive_context(filtered_results, query)
        
        return optimized_context
    
    def _score_relevance(self, results: List[Dict], query: str) -> List[Dict]:
        """计算相关性分数"""
        query_terms = set(query.lower().split())
        
        for result in results:
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            
            # 基础相似度分数
            base_score = result.get('combined_score', 0.5)
            
            # 查询词重叠度
            content_terms = set(content.lower().split())
            overlap_ratio = len(query_terms.intersection(content_terms)) / len(query_terms)
            
            # 章节类型权重
            section_type = metadata.get('section_type', 'content')
            section_weight = self.token_weights.get(section_type, 1.0)
            
            # 内容质量权重
            quality_bonus = 0
            if metadata.get('has_formulas'): quality_bonus += 0.1
            if metadata.get('has_citations'): quality_bonus += 0.1
            if metadata.get('has_code'): quality_bonus += 0.05
            
            # 综合相关性分数
            relevance_score = (base_score * 0.4 + 
                             overlap_ratio * 0.4 + 
                             section_weight * 0.1 + 
                             quality_bonus * 0.1) * section_weight
            
            result['relevance_score'] = min(relevance_score, 1.0)
        
        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)
    
    def _build_adaptive_context(self, results: List[Dict], query: str) -> ContextWindow:
        """构建自适应上下文"""
        context_parts = []
        total_tokens = 0
        source_papers = []
        relevance_scores = []
        
        # 添加查询信息
        context_parts.append(f"Query: {query}\n")
        total_tokens += len(query.split()) + 2
        
        for result in results:
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            relevance = result['relevance_score']
            
            # 估算token数量 (粗略估算: 1 token ≈ 0.75 words)
            content_tokens = int(len(content.split()) * 0.75)
            
            if total_tokens + content_tokens > self.max_tokens:
                # 如果超出限制，截断内容
                remaining_tokens = self.max_tokens - total_tokens
                words_to_keep = int(remaining_tokens / 0.75)
                content = ' '.join(content.split()[:words_to_keep]) + "..."
                total_tokens = self.max_tokens
            else:
                total_tokens += content_tokens
            
            # 构建上下文片段
            paper_id = metadata.get('paper_id', 'unknown')
            section_type = metadata.get('section_type', 'content')
            
            context_part = f"""
Source: {paper_id} ({section_type})
Relevance: {relevance:.3f}
Content: {content}
"""
            context_parts.append(context_part)
            source_papers.append(paper_id)
            relevance_scores.append(relevance)
            
            if total_tokens >= self.max_tokens:
                break
        
        # 计算整体置信度
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        confidence_level = min(avg_relevance * len(relevance_scores) / 5, 1.0)  # 归一化到[0,1]
        
        return ContextWindow(
            content="\n".join(context_parts),
            source_papers=list(set(source_papers)),
            relevance_score=avg_relevance,
            confidence_level=confidence_level,
            total_tokens=total_tokens
        )

class PromptEngineer:
    """Prompt工程师 - 专业领域提示词模板"""
    
    def __init__(self):
        self.templates = {
            'comparison': self._load_comparison_template(),
            'definition': self._load_definition_template(),
            'methodology': self._load_methodology_template(),
            'recent_work': self._load_recent_work_template(),
            'general': self._load_general_template()
        }
        
        # 专业术语词典
        self.domain_terminology = {
            'machine_learning': ['supervised learning', 'unsupervised learning', 'reinforcement learning'],
            'deep_learning': ['neural networks', 'backpropagation', 'gradient descent'],
            'nlp': ['tokenization', 'embedding', 'attention mechanism'],
            'computer_vision': ['convolution', 'feature extraction', 'object detection']
        }
    
    def generate_prompt(self, query: str, query_intent: str, context: ContextWindow) -> str:
        """生成专业提示词"""
        template = self.templates.get(query_intent, self.templates['general'])
        
        # 检测专业领域
        detected_domains = self._detect_domains(query + " " + context.content)
        
        # 构建专业提示词
        prompt = template.format(
            query=query,
            context=context.content,
            confidence_level=context.confidence_level,
            source_count=len(context.source_papers),
            domains=", ".join(detected_domains) if detected_domains else "AI/ML"
        )
        
        return prompt
    
    def _detect_domains(self, text: str) -> List[str]:
        """检测专业领域"""
        text_lower = text.lower()
        detected = []
        
        for domain, terms in self.domain_terminology.items():
            if any(term in text_lower for term in terms):
                detected.append(domain)
        
        return detected
    
    def _load_comparison_template(self) -> str:
        return """你是一位AI/机器学习领域的资深研究专家。请基于提供的学术论文内容回答比较性问题。

专业要求：
1. 使用准确的技术术语和概念
2. 提供客观、平衡的比较分析
3. 指出每种方法的优势和局限性
4. 引用具体的论文证据支持观点
5. 承认知识边界，不夸大结论

上下文信息置信度: {confidence_level:.2f}
参考文献数量: {source_count}
相关领域: {domains}

研究上下文:
{context}

用户问题: {query}

请提供专业的比较分析，包括：
- 核心差异点
- 各自优势
- 适用场景
- 性能对比
- 具体引用"""

    def _load_definition_template(self) -> str:
        return """你是一位AI/机器学习领域的资深专家和教育者。请基于学术论文内容提供准确的概念定义。

专业标准：
1. 提供精确、权威的定义
2. 包含技术细节和工作原理
3. 举例说明实际应用
4. 区分与相关概念的差异
5. 引用权威论文来源

上下文置信度: {confidence_level:.2f}
参考文献: {source_count}篇
领域范围: {domains}

学术资料:
{context}

定义请求: {query}

请提供专业定义，包括：
- 核心概念解释
- 技术原理
- 关键特征
- 应用实例
- 学术引用"""

    def _load_methodology_template(self) -> str:
        return """你是一位经验丰富的AI研究员和方法论专家。请基于学术论文详细说明技术方法。

方法论要求：
1. 清晰描述技术步骤和流程
2. 解释方法的理论基础
3. 分析实现难点和解决方案
4. 提供性能评估标准
5. 引用相关研究作为支撑

技术可信度: {confidence_level:.2f}
研究基础: {source_count}篇论文
专业领域: {domains}

技术文献:
{context}

方法询问: {query}

请详细说明方法论，包括：
- 算法步骤
- 理论依据
- 实现要点
- 评估方法
- 文献支持"""

    def _load_recent_work_template(self) -> str:
        return """你是一位跟踪AI前沿发展的研究分析师。请基于最新研究文献分析技术趋势。

分析框架：
1. 识别最新技术突破和进展
2. 分析发展趋势和方向
3. 评估技术成熟度和应用前景
4. 比较不同研究团队的贡献
5. 基于论文证据支持分析

信息时效性: {confidence_level:.2f}
文献覆盖: {source_count}篇
研究方向: {domains}

前沿文献:
{context}

趋势查询: {query}

请分析最新发展，包括：
- 关键突破点
- 技术趋势
- 研究热点
- 应用前景
- 重要论文"""

    def _load_general_template(self) -> str:
        return """你是一位AI/机器学习领域的专业研究顾问。请基于学术文献提供准确、专业的解答。

专业准则：
1. 基于学术证据进行客观分析
2. 使用准确的技术术语
3. 承认不确定性和知识局限
4. 提供平衡、全面的观点
5. 引用具体文献支持论点

文献质量: {confidence_level:.2f}
参考数量: {source_count}篇
相关领域: {domains}

学术资源:
{context}

研究问题: {query}

请提供专业解答，包括：
- 核心观点
- 支撑证据
- 不同视角
- 实际意义
- 文献引用"""

class FactChecker:
    """事实核查器 - 基于多源验证的可信度评估"""
    
    def __init__(self):
        self.confidence_thresholds = {
            'strong': 0.8,
            'moderate': 0.6,
            'weak': 0.4
        }
    
    def verify_claims(self, generated_answer: str, source_documents: List[Dict]) -> List[FactCheckResult]:
        """验证生成答案中的关键声明"""
        
        # 1. 提取关键声明
        claims = self._extract_claims(generated_answer)
        
        # 2. 对每个声明进行多源验证
        fact_check_results = []
        for claim in claims:
            verification = self._verify_single_claim(claim, source_documents)
            fact_check_results.append(verification)
        
        return fact_check_results
    
    def _extract_claims(self, text: str) -> List[str]:
        """提取文本中的关键声明"""
        # 识别声明性语句的模式
        claim_patterns = [
            r'[A-Z][^.!?]*(?:is|are|was|were|has|have|shows|demonstrates|proves|indicates)[^.!?]*[.!?]',
            r'[A-Z][^.!?]*(?:achieves|outperforms|improves|reduces|increases)[^.!?]*[.!?]',
            r'[A-Z][^.!?]*(?:method|approach|technique|algorithm)[^.!?]*[.!?]'
        ]
        
        claims = []
        for pattern in claim_patterns:
            matches = re.findall(pattern, text)
            claims.extend(matches)
        
        # 去重和清理
        claims = list(set([claim.strip() for claim in claims if len(claim.strip()) > 20]))
        return claims[:10]  # 限制数量
    
    def _verify_single_claim(self, claim: str, sources: List[Dict]) -> FactCheckResult:
        """验证单个声明"""
        supporting_sources = []
        contradicting_sources = []
        relevance_scores = []
        
        claim_terms = set(claim.lower().split())
        
        for source in sources:
            content = source.get('content', '')
            paper_id = source.get('metadata', {}).get('paper_id', 'unknown')
            
            # 计算声明与源文档的相关性
            content_terms = set(content.lower().split())
            overlap = len(claim_terms.intersection(content_terms))
            relevance = overlap / len(claim_terms) if claim_terms else 0
            
            if relevance > 0.3:  # 相关阈值
                # 简单的支持/反对检测
                support_indicators = ['shows', 'demonstrates', 'proves', 'confirms', 'validates']
                contradict_indicators = ['however', 'but', 'contradicts', 'disproves', 'challenges']
                
                support_score = sum(1 for indicator in support_indicators if indicator in content.lower())
                contradict_score = sum(1 for indicator in contradict_indicators if indicator in content.lower())
                
                if support_score > contradict_score:
                    supporting_sources.append(paper_id)
                elif contradict_score > support_score:
                    contradicting_sources.append(paper_id)
                
                relevance_scores.append(relevance)
        
        # 计算整体支持水平
        total_sources = len(supporting_sources) + len(contradicting_sources)
        if total_sources == 0:
            support_level = 'insufficient_data'
            confidence = 0.0
        else:
            support_ratio = len(supporting_sources) / total_sources
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
            confidence = support_ratio * avg_relevance
            
            if confidence >= self.confidence_thresholds['strong']:
                support_level = 'strong'
            elif confidence >= self.confidence_thresholds['moderate']:
                support_level = 'moderate'
            elif confidence >= self.confidence_thresholds['weak']:
                support_level = 'weak'
            else:
                support_level = 'contradicted' if contradicting_sources else 'weak'
        
        return FactCheckResult(
            claim=claim,
            support_level=support_level,
            supporting_sources=supporting_sources,
            contradicting_sources=contradicting_sources,
            confidence_score=confidence
        )

class CitationManager:
    """引用管理器 - 标准学术引用格式"""
    
    def __init__(self, citation_style='APA'):
        self.citation_style = citation_style
        self.citation_cache = {}
    
    def generate_citations(self, sources: List[Dict], content_snippets: Dict = None) -> List[Citation]:
        """生成标准引用"""
        citations = []
        
        for source in sources:
            metadata = source.get('metadata', {})
            paper_id = metadata.get('paper_id', 'unknown')
            
            # 避免重复引用
            if paper_id in self.citation_cache:
                citations.append(self.citation_cache[paper_id])
                continue
            
            # 提取引用信息
            title = metadata.get('title', 'Unknown Title')
            authors = self._parse_authors(metadata.get('authors', 'Unknown Author'))
            year = self._extract_year(metadata.get('published', '2024'))
            
            # 内容片段
            snippet = content_snippets.get(paper_id, source.get('content', ''))[:200] + "..." if content_snippets else ""
            
            citation = Citation(
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
                content_snippet=snippet,
                relevance_score=source.get('relevance_score', 0.5)
            )
            
            citations.append(citation)
            self.citation_cache[paper_id] = citation
        
        return citations
    
    def format_citation(self, citation: Citation) -> str:
        """格式化引用"""
        if self.citation_style == 'APA':
            return self._format_apa(citation)
        elif self.citation_style == 'IEEE':
            return self._format_ieee(citation)
        else:
            return self._format_apa(citation)  # 默认APA格式
    
    def _format_apa(self, citation: Citation) -> str:
        """APA格式引用"""
        authors_str = self._format_authors_apa(citation.authors)
        return f"{authors_str} ({citation.year}). {citation.title}. Retrieved from arXiv:{citation.paper_id}"
    
    def _format_ieee(self, citation: Citation) -> str:
        """IEEE格式引用"""
        authors_str = self._format_authors_ieee(citation.authors)
        return f'{authors_str}, "{citation.title}," arXiv preprint arXiv:{citation.paper_id}, {citation.year}.'
    
    def _parse_authors(self, authors_data) -> List[str]:
        """解析作者信息"""
        if isinstance(authors_data, list):
            return authors_data
        elif isinstance(authors_data, str):
            # 简单的作者名分割
            return [author.strip() for author in authors_data.split(',')]
        else:
            return ['Unknown Author']
    
    def _extract_year(self, date_str: str) -> str:
        """提取年份"""
        year_match = re.search(r'\d{4}', str(date_str))
        return year_match.group() if year_match else '2024'
    
    def _format_authors_apa(self, authors: List[str]) -> str:
        """APA格式作者名"""
        if not authors or authors == ['Unknown Author']:
            return "Unknown Author"
        
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} & {authors[1]}"
        else:
            return f"{authors[0]} et al."
    
    def _format_authors_ieee(self, authors: List[str]) -> str:
        """IEEE格式作者名"""
        if not authors or authors == ['Unknown Author']:
            return "Unknown Author"
        
        if len(authors) <= 3:
            return ', '.join(authors)
        else:
            return f"{authors[0]} et al."
    
    def generate_bibliography(self, citations: List[Citation]) -> str:
        """生成参考文献列表"""
        bibliography = "References:\n\n"
        
        # 按第一作者姓氏排序
        sorted_citations = sorted(citations, key=lambda c: c.authors[0] if c.authors else 'Unknown')
        
        for i, citation in enumerate(sorted_citations, 1):
            formatted_citation = self.format_citation(citation)
            bibliography += f"[{i}] {formatted_citation}\n\n"
        
        return bibliography

class QualityEnhancedGenerator:
    """质量增强生成器 - 整合所有生成质量提升功能"""
    
    def __init__(self, citation_style='APA'):
        self.context_optimizer = ContextOptimizer()
        self.prompt_engineer = PromptEngineer()
        self.fact_checker = FactChecker()
        self.citation_manager = CitationManager(citation_style)
        
        # 质量控制参数
        self.quality_config = {
            'min_confidence': 0.3,
            'max_context_tokens': 2048,
            'enable_fact_check': True,
            'enable_citations': True,
            'require_multi_source': True
        }
    
    def generate_enhanced_answer(self, query: str, query_intent: str, 
                               retrieved_results: List[Dict]) -> Dict:
        """生成增强质量的回答"""
        
        # 1. 上下文优化
        optimized_context = self.context_optimizer.optimize_context(retrieved_results, query)
        
        # 2. 质量检查
        if optimized_context.confidence_level < self.quality_config['min_confidence']:
            return {
                'answer': f"基于当前可用文献，无法提供高质量回答。建议扩展知识库或重新表述问题。\n\n置信度: {optimized_context.confidence_level:.2f}",
                'quality_warning': True,
                'confidence': optimized_context.confidence_level
            }
        
        # 3. 生成专业提示词
        enhanced_prompt = self.prompt_engineer.generate_prompt(query, query_intent, optimized_context)
        
        # 4. 生成引用
        citations = self.citation_manager.generate_citations(retrieved_results)
        
        # 5. 构建增强回答
        enhanced_answer = self._build_enhanced_response(
            query, optimized_context, citations, enhanced_prompt
        )
        
        # 6. 事实核查
        fact_check_results = []
        if self.quality_config['enable_fact_check']:
            fact_check_results = self.fact_checker.verify_claims(enhanced_answer, retrieved_results)
        
        return {
            'answer': enhanced_answer,
            'context_info': {
                'confidence_level': optimized_context.confidence_level,
                'source_count': len(optimized_context.source_papers),
                'total_tokens': optimized_context.total_tokens
            },
            'citations': citations,
            'fact_check': fact_check_results,
            'quality_metrics': self._calculate_quality_metrics(optimized_context, citations, fact_check_results)
        }
    
    def _build_enhanced_response(self, query: str, context: ContextWindow, 
                               citations: List[Citation], prompt: str) -> str:
        """构建增强回答"""
        
        # 注意：这里是构建回答的结构，实际的LLM生成需要外部API
        response_parts = []
        
        # 主要回答部分
        response_parts.append("基于学术文献的专业分析：\n")
        
        # 置信度声明
        confidence_text = self._get_confidence_text(context.confidence_level)
        response_parts.append(f"回答可信度: {confidence_text}\n")
        
        # 这里应该是LLM生成的内容，我们提供结构化框架
        response_parts.append("[此处为基于专业提示词生成的详细回答]\n")
        
        # 添加引用
        if self.quality_config['enable_citations'] and citations:
            response_parts.append("\n学术引用:")
            for i, citation in enumerate(citations[:5], 1):  # 限制引用数量
                formatted_citation = self.citation_manager.format_citation(citation)
                response_parts.append(f"[{i}] {formatted_citation}")
        
        return "\n".join(response_parts)
    
    def _get_confidence_text(self, confidence: float) -> str:
        """获取置信度描述"""
        if confidence >= 0.8:
            return "高 (基于多个权威源)"
        elif confidence >= 0.6:
            return "中等 (基于有限但相关的源)"
        elif confidence >= 0.4:
            return "较低 (基于少量相关源)"
        else:
            return "低 (证据不足)"
    
    def _calculate_quality_metrics(self, context: ContextWindow, 
                                 citations: List[Citation], 
                                 fact_checks: List[FactCheckResult]) -> Dict:
        """计算质量指标"""
        
        # 源质量分数
        source_quality = min(len(citations) / 3, 1.0)  # 理想情况下至少3个源
        
        # 事实核查分数
        if fact_checks:
            strong_claims = sum(1 for fc in fact_checks if fc.support_level == 'strong')
            fact_check_score = strong_claims / len(fact_checks)
        else:
            fact_check_score = 0.5  # 默认分数
        
        # 整体质量分数
        overall_quality = (context.confidence_level * 0.4 + 
                          source_quality * 0.3 + 
                          fact_check_score * 0.3)
        
        return {
            'overall_quality': overall_quality,
            'context_confidence': context.confidence_level,
            'source_quality': source_quality,
            'fact_check_score': fact_check_score,
            'recommendation': self._get_quality_recommendation(overall_quality)
        }
    
    def _get_quality_recommendation(self, quality_score: float) -> str:
        """获取质量建议"""
        if quality_score >= 0.8:
            return "高质量回答，可直接使用"
        elif quality_score >= 0.6:
            return "中等质量，建议验证关键信息"
        elif quality_score >= 0.4:
            return "质量一般，需要额外验证"
        else:
            return "质量较低，建议收集更多资料"


# 测试和使用示例
def test_quality_enhancement():
    """测试生成质量提升功能"""
    
    # 模拟检索结果
    mock_results = [
        {
            'content': 'Transformer architecture uses self-attention mechanism to process sequences in parallel.',
            'metadata': {
                'paper_id': '1706.03762',
                'title': 'Attention Is All You Need',
                'authors': ['Ashish Vaswani', 'Noam Shazeer'],
                'published': '2017-06-12',
                'section_type': 'abstract',
                'has_formulas': True,
                'has_citations': True
            },
            'combined_score': 0.9
        },
        {
            'content': 'BERT demonstrates that bidirectional training significantly improves language understanding.',
            'metadata': {
                'paper_id': '1810.04805',
                'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
                'authors': ['Jacob Devlin', 'Ming-Wei Chang'],
                'published': '2018-10-11',
                'section_type': 'results',
                'has_citations': True
            },
            'combined_score': 0.8
        }
    ]
    
    # 测试质量增强生成器
    generator = QualityEnhancedGenerator()
    
    result = generator.generate_enhanced_answer(
        query="What is the transformer architecture?",
        query_intent="definition",
        retrieved_results=mock_results
    )
    
    print("质量增强生成测试结果:")
    print(f"置信度: {result['context_info']['confidence_level']:.2f}")
    print(f"引用数量: {len(result['citations'])}")
    print(f"质量评分: {result['quality_metrics']['overall_quality']:.2f}")
    print(f"建议: {result['quality_metrics']['recommendation']}")

if __name__ == "__main__":
    test_quality_enhancement()