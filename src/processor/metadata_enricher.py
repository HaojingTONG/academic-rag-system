# src/processor/metadata_enricher.py
import re
import spacy
from typing import List, Dict, Set
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

class MetadataEnricher:
    def __init__(self):
        # 初始化NLP工具
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("请安装spaCy英文模型")
            self.nlp = None
        
        # 下载NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        
        # AI/ML领域的专业词汇
        self.ai_domains = {
            'machine_learning': ['learning', 'training', 'model', 'algorithm', 'supervised', 'unsupervised'],
            'deep_learning': ['neural', 'network', 'deep', 'layer', 'activation', 'backpropagation'],
            'nlp': ['language', 'text', 'linguistic', 'semantic', 'syntactic', 'parsing'],
            'computer_vision': ['image', 'visual', 'vision', 'pixel', 'convolution', 'detection'],
            'reinforcement_learning': ['reward', 'policy', 'agent', 'environment', 'action', 'value'],
            'ai_theory': ['artificial', 'intelligence', 'cognitive', 'reasoning', 'knowledge']
        }
        
        self.methodology_keywords = {
            'supervised_learning': ['classification', 'regression', 'labeled', 'training'],
            'unsupervised_learning': ['clustering', 'dimensionality', 'unsupervised', 'unlabeled'],
            'semi_supervised': ['semi-supervised', 'few-shot', 'active learning'],
            'transfer_learning': ['transfer', 'fine-tuning', 'pre-trained', 'domain adaptation'],
            'ensemble': ['ensemble', 'bagging', 'boosting', 'voting'],
            'optimization': ['optimization', 'gradient', 'descent', 'convergence']
        }
    
    def enrich_metadata(self, paper_data: Dict) -> Dict:
        """增强论文元数据"""
        
        # 基础信息
        title = paper_data.get('title', '')
        abstract = paper_data.get('abstract', '')
        full_text = paper_data.get('full_text', '')
        
        # 结合标题和摘要进行分析
        main_text = f"{title} {abstract}"
        
        enhanced_metadata = {
            # 原始元数据
            **paper_data,
            
            # 新增的增强元数据
            'keywords': self.extract_keywords(main_text, full_text),
            'research_domains': self.classify_research_domains(main_text),
            'methodologies': self.identify_methodologies(main_text),
            'contributions': self.extract_contributions(abstract),
            'technical_terms': self.extract_technical_terms(main_text),
            'difficulty_level': self.assess_difficulty_level(main_text),
            'paper_type': self.classify_paper_type(title, abstract),
            'novelty_indicators': self.detect_novelty_indicators(main_text),
            'experimental_setup': self.analyze_experimental_setup(full_text),
            'datasets_mentioned': self.extract_datasets(full_text),
            'metrics_mentioned': self.extract_evaluation_metrics(full_text)
        }
        
        return enhanced_metadata
    
    def extract_keywords(self, main_text: str, full_text: str, top_k: int = 10) -> List[str]:
        """提取关键词"""
        keywords = set()
        
        # 方法1: TF-IDF提取
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            if full_text and len(full_text) > 100:
                tfidf_matrix = vectorizer.fit_transform([full_text])
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]
                
                # 获取top-k关键词
                top_indices = tfidf_scores.argsort()[-top_k:][::-1]
                tfidf_keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
                keywords.update(tfidf_keywords)
        
        except Exception as e:
            print(f"TF-IDF关键词提取错误: {e}")
        
        # 方法2: 基于NER的实体提取
        if self.nlp:
            doc = self.nlp(main_text)
            entities = [ent.text.lower() for ent in doc.ents 
                       if ent.label_ in ['ORG', 'PRODUCT', 'EVENT', 'WORK_OF_ART']]
            keywords.update(entities)
        
        # 方法3: 领域特定关键词
        text_lower = main_text.lower()
        for domain, domain_keywords in self.ai_domains.items():
            for keyword in domain_keywords:
                if keyword in text_lower:
                    keywords.add(keyword)
        
        # 过滤和清理
        filtered_keywords = []
        for keyword in keywords:
            if (len(keyword) > 2 and 
                keyword not in self.stop_words and 
                not keyword.isdigit()):
                filtered_keywords.append(keyword)
        
        return filtered_keywords[:top_k]
    
    def classify_research_domains(self, text: str) -> List[Dict]:
        """分类研究领域"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.ai_domains.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                count = text_lower.count(keyword)
                if count > 0:
                    score += count
                    matched_keywords.append(keyword)
            
            if score > 0:
                domain_scores[domain] = {
                    'score': score,
                    'keywords': matched_keywords,
                    'confidence': min(score / len(keywords), 1.0)
                }
        
        # 按分数排序
        sorted_domains = sorted(domain_scores.items(), 
                               key=lambda x: x[1]['score'], 
                               reverse=True)
        
        return [{'domain': domain, **info} for domain, info in sorted_domains]
    
    def identify_methodologies(self, text: str) -> List[Dict]:
        """识别方法论"""
        text_lower = text.lower()
        methodology_scores = {}
        
        for method, keywords in self.methodology_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                methodology_scores[method] = {
                    'score': score,
                    'keywords': matched_keywords,
                    'confidence': score / len(keywords)
                }
        
        sorted_methods = sorted(methodology_scores.items(),
                               key=lambda x: x[1]['score'],
                               reverse=True)
        
        return [{'methodology': method, **info} for method, info in sorted_methods]
    
    def extract_contributions(self, abstract: str) -> List[str]:
        """提取论文贡献点"""
        contributions = []
        
        # 寻找贡献相关的句子
        contribution_patterns = [
            r'we propose.*?[.!]',
            r'we present.*?[.!]',
            r'we introduce.*?[.!]',
            r'we develop.*?[.!]',
            r'we show.*?[.!]',
            r'our contribution.*?[.!]',
            r'our main contribution.*?[.!]',
            r'novel.*?[.!]',
            r'new.*?approach.*?[.!]'
        ]
        
        for pattern in contribution_patterns:
            matches = re.finditer(pattern, abstract, re.IGNORECASE | re.DOTALL)
            for match in matches:
                contribution = match.group(0).strip()
                if len(contribution) > 20:  # 过滤过短的匹配
                    contributions.append(contribution)
        
        return contributions
    
    def extract_technical_terms(self, text: str) -> List[str]:
        """提取技术术语"""
        technical_terms = set()
        
        # 技术术语模式
        tech_patterns = [
            r'\b[A-Z][a-z]*[A-Z][A-Za-z]*\b',  # CamelCase
            r'\b[A-Z]{2,}(?:-[A-Z]{2,})*\b',   # 缩写
            r'\b\w+(?:-\w+)*(?:\s+\w+)*\s+(?:algorithm|model|method|approach|framework)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text)
            technical_terms.update(matches)
        
        # 过滤常见词汇
        filtered_terms = []
        for term in technical_terms:
            if (len(term) > 3 and 
                term.lower() not in self.stop_words and
                not term.isdigit()):
                filtered_terms.append(term)
        
        return list(set(filtered_terms))[:15]  # 返回前15个
    
    def assess_difficulty_level(self, text: str) -> Dict:
        """评估论文难度等级"""
        text_lower = text.lower()
        
        # 难度指标
        difficulty_indicators = {
            'beginner': ['introduction', 'basic', 'simple', 'overview', 'survey'],
            'intermediate': ['method', 'algorithm', 'approach', 'technique', 'implementation'],
            'advanced': ['novel', 'state-of-the-art', 'optimization', 'theoretical', 'complex'],
            'expert': ['breakthrough', 'revolutionary', 'paradigm', 'fundamental', 'cutting-edge']
        }
        
        level_scores = {}
        for level, indicators in difficulty_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            level_scores[level] = score
        
        # 计算总分和权重
        total_score = sum(level_scores.values())
        if total_score == 0:
            return {'level': 'intermediate', 'confidence': 0.5, 'scores': level_scores}
        
        # 加权计算最终难度
        weights = {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
        weighted_score = sum(weights[level] * score for level, score in level_scores.items())
        avg_difficulty = weighted_score / total_score
        
        if avg_difficulty <= 1.5:
            difficulty_level = 'beginner'
        elif avg_difficulty <= 2.5:
            difficulty_level = 'intermediate'
        elif avg_difficulty <= 3.5:
            difficulty_level = 'advanced'
        else:
            difficulty_level = 'expert'
        
        confidence = max(level_scores.values()) / total_score if total_score > 0 else 0.5
        
        return {
            'level': difficulty_level,
            'confidence': confidence,
            'scores': level_scores,
            'weighted_score': avg_difficulty
        }
    
    def classify_paper_type(self, title: str, abstract: str) -> Dict:
        """分类论文类型"""
        combined_text = f"{title} {abstract}".lower()
        
        paper_types = {
            'theoretical': ['theorem', 'proof', 'analysis', 'mathematical', 'formal'],
            'empirical': ['experiment', 'evaluation', 'benchmark', 'dataset', 'results'],
            'survey': ['survey', 'review', 'overview', 'comprehensive', 'state-of-the-art'],
            'methodology': ['method', 'approach', 'algorithm', 'framework', 'technique'],
            'application': ['application', 'system', 'implementation', 'practical', 'real-world'],
            'case_study': ['case study', 'analysis of', 'investigation', 'examination']
        }
        
        type_scores = {}
        for paper_type, keywords in paper_types.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                type_scores[paper_type] = score
        
        if not type_scores:
            return {'type': 'methodology', 'confidence': 0.5}
        
        primary_type = max(type_scores.items(), key=lambda x: x[1])
        total_score = sum(type_scores.values())
        confidence = primary_type[1] / total_score
        
        return {
            'type': primary_type[0],
            'confidence': confidence,
            'all_scores': type_scores
        }
    
    def detect_novelty_indicators(self, text: str) -> List[str]:
        """检测新颖性指标"""
        novelty_patterns = [
            r'first.*?to.*?[.!]',
            r'novel.*?[.!]',
            r'new.*?approach.*?[.!]',
            r'unprecedented.*?[.!]',
            r'breakthrough.*?[.!]',
            r'state-of-the-art.*?[.!]',
            r'superior.*?performance.*?[.!]',
            r'outperform.*?[.!]',
            r'significant.*?improvement.*?[.!]'
        ]
        
        novelty_indicators = []
        for pattern in novelty_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            novelty_indicators.extend(matches)
        
        return novelty_indicators[:5]  # 返回前5个
    
    def analyze_experimental_setup(self, full_text: str) -> Dict:
        """分析实验设置"""
        if not full_text:
            return {}
        
        text_lower = full_text.lower()
        
        # 实验相关关键词
        experiment_indicators = {
            'has_experiments': any(keyword in text_lower for keyword in 
                                 ['experiment', 'evaluation', 'test', 'benchmark']),
            'has_datasets': any(keyword in text_lower for keyword in 
                              ['dataset', 'data', 'corpus', 'benchmark']),
            'has_metrics': any(keyword in text_lower for keyword in 
                             ['accuracy', 'precision', 'recall', 'f1', 'metric']),
            'has_comparison': any(keyword in text_lower for keyword in 
                                ['compare', 'baseline', 'versus', 'outperform']),
            'has_ablation': any(keyword in text_lower for keyword in 
                              ['ablation', 'component', 'module', 'variant'])
        }
        
        return experiment_indicators
    
    def extract_datasets(self, full_text: str) -> List[str]:
        """提取数据集名称"""
        if not full_text:
            return []
        
        # 常见数据集名称模式
        dataset_patterns = [
            r'\b[A-Z][A-Za-z]*-?\d*\b(?:\s+dataset|\s+corpus)?',
            r'\b(?:CIFAR|MNIST|ImageNet|COCO|Wikipedia|Reuters|IMDb|Amazon)\b',
            r'\b\w+(?:-\w+)*\s+(?:dataset|corpus|benchmark)\b'
        ]
        
        datasets = set()
        for pattern in dataset_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            datasets.update(matches)
        
        # 过滤和清理
        filtered_datasets = []
        for dataset in datasets:
            if len(dataset) > 3 and dataset.lower() not in self.stop_words:
                filtered_datasets.append(dataset)
        
        return list(set(filtered_datasets))[:10]
    
    def extract_evaluation_metrics(self, full_text: str) -> List[str]:
        """提取评估指标"""
        if not full_text:
            return []
        
        # 评估指标模式
        metric_patterns = [
            r'\b(?:accuracy|precision|recall|f1|auc|map|bleu|rouge|meteor)\b',
            r'\b(?:mse|mae|rmse|r2|correlation)\b',
            r'\b(?:loss|error|score|rate)\b'
        ]
        
        metrics = set()
        for pattern in metric_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            metrics.update(matches)
        
        return list(metrics)


# 使用示例和测试
def test_metadata_enricher():
    enricher = MetadataEnricher()
    
    sample_paper = {
        'title': 'A Novel Deep Learning Approach for Natural Language Processing',
        'abstract': 'We propose a new neural network architecture for text classification. Our method outperforms existing approaches on benchmark datasets.',
        'full_text': 'We evaluate our method on CIFAR-10 and ImageNet datasets using accuracy and F1-score metrics...'
    }
    
    enriched = enricher.enrich_metadata(sample_paper)
    
    print("增强后的元数据:")
    for key, value in enriched.items():
        if key not in sample_paper:  # 只显示新增的元数据
            print(f"{key}: {value}")

if __name__ == "__main__":
    test_metadata_enricher()