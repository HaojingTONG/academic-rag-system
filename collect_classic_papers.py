#!/usr/bin/env python3
"""
收集经典高质量AI/ML论文
使用预定义的经典论文列表和arXiv API获取论文详情
包含Transformer、BERT、GPT等里程碑式论文
"""

import requests
import json
import time
import os
from datetime import datetime
from typing import List, Dict, Optional
import re

class ClassicPaperCollector:
    """经典论文收集器"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Academic-RAG-System/1.0'
        })
        
        # 经典高质量论文列表 - 包含arXiv ID和预估引用量
        self.classic_papers = [
            # Transformer系列 - 最重要的论文
            {"arxiv_id": "1706.03762", "title": "Attention Is All You Need", "estimated_citations": 50000, "category": "transformer"},
            {"arxiv_id": "1810.04805", "title": "BERT: Pre-training of Deep Bidirectional Transformers", "estimated_citations": 30000, "category": "bert"},
            {"arxiv_id": "1910.13461", "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach", "estimated_citations": 8000, "category": "bert"},
            {"arxiv_id": "2005.14165", "title": "GPT-3: Language Models are Few-Shot Learners", "estimated_citations": 15000, "category": "gpt"},
            {"arxiv_id": "1909.11942", "title": "ALBERT: A Lite BERT for Self-supervised Learning", "estimated_citations": 4000, "category": "bert"},
            
            # 经典深度学习论文
            {"arxiv_id": "1512.03385", "title": "Deep Residual Learning for Image Recognition", "estimated_citations": 45000, "category": "cnn"},
            {"arxiv_id": "1409.0473", "title": "Neural Machine Translation by Jointly Learning to Align and Translate", "estimated_citations": 15000, "category": "nmt"},
            {"arxiv_id": "1506.02025", "title": "Spatial Transformer Networks", "estimated_citations": 5000, "category": "cnn"},
            {"arxiv_id": "1502.01852", "title": "Delving Deep into Rectifiers", "estimated_citations": 8000, "category": "optimization"},
            {"arxiv_id": "1412.6980", "title": "Adam: A Method for Stochastic Optimization", "estimated_citations": 25000, "category": "optimization"},
            
            # 计算机视觉经典
            {"arxiv_id": "1409.1556", "title": "Very Deep Convolutional Networks for Large-Scale Image Recognition", "estimated_citations": 30000, "category": "cnn"},
            {"arxiv_id": "1312.4400", "title": "Visualizing and Understanding Convolutional Networks", "estimated_citations": 8000, "category": "cnn"},
            {"arxiv_id": "1506.01497", "title": "Faster R-CNN: Towards Real-Time Object Detection", "estimated_citations": 20000, "category": "detection"},
            {"arxiv_id": "1612.03144", "title": "You Only Look Once: Unified, Real-Time Object Detection", "estimated_citations": 15000, "category": "detection"},
            {"arxiv_id": "1505.04597", "title": "U-Net: Convolutional Networks for Biomedical Image Segmentation", "estimated_citations": 12000, "category": "segmentation"},
            
            # GAN系列
            {"arxiv_id": "1406.2661", "title": "Generative Adversarial Networks", "estimated_citations": 25000, "category": "gan"},
            {"arxiv_id": "1511.06434", "title": "Unsupervised Representation Learning with DCGANs", "estimated_citations": 8000, "category": "gan"},
            {"arxiv_id": "1703.10593", "title": "Unpaired Image-to-Image Translation using Cycle-GANs", "estimated_citations": 10000, "category": "gan"},
            {"arxiv_id": "1912.04958", "title": "Analyzing and Improving the Image Quality of StyleGAN", "estimated_citations": 3000, "category": "gan"},
            
            # 强化学习经典
            {"arxiv_id": "1312.5602", "title": "Playing Atari with Deep Reinforcement Learning", "estimated_citations": 8000, "category": "rl"},
            {"arxiv_id": "1509.02971", "title": "Deep Reinforcement Learning with Double Q-learning", "estimated_citations": 5000, "category": "rl"},
            {"arxiv_id": "1511.05952", "title": "Dueling Network Architectures for Deep Reinforcement Learning", "estimated_citations": 4000, "category": "rl"},
            {"arxiv_id": "1707.06347", "title": "Proximal Policy Optimization Algorithms", "estimated_citations": 6000, "category": "rl"},
            
            # NLP经典（除了Transformer系列）
            {"arxiv_id": "1301.3781", "title": "Efficient Estimation of Word Representations in Vector Space", "estimated_citations": 20000, "category": "nlp"},
            {"arxiv_id": "1405.0312", "title": "Distributed Representations of Sentences and Documents", "estimated_citations": 8000, "category": "nlp"},
            {"arxiv_id": "1508.04025", "title": "Effective Approaches to Attention-based Neural Machine Translation", "estimated_citations": 6000, "category": "nmt"},
            {"arxiv_id": "1409.3215", "title": "Sequence to Sequence Learning with Neural Networks", "estimated_citations": 12000, "category": "seq2seq"},
            {"arxiv_id": "1511.08198", "title": "Neural Conversational Model", "estimated_citations": 3000, "category": "dialogue"},
            
            # 图神经网络
            {"arxiv_id": "1609.02907", "title": "Semi-Supervised Classification with Graph Convolutional Networks", "estimated_citations": 8000, "category": "gnn"},
            {"arxiv_id": "1710.10903", "title": "Graph Attention Networks", "estimated_citations": 5000, "category": "gnn"},
            {"arxiv_id": "1905.02850", "title": "How Powerful are Graph Neural Networks?", "estimated_citations": 3000, "category": "gnn"},
            
            # 自监督学习
            {"arxiv_id": "2002.05709", "title": "A Simple Framework for Contrastive Learning of Visual Representations", "estimated_citations": 5000, "category": "ssl"},
            {"arxiv_id": "2003.04297", "title": "Momentum Contrast for Unsupervised Visual Representation Learning", "estimated_citations": 4000, "category": "ssl"},
            {"arxiv_id": "2006.07733", "title": "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning", "estimated_citations": 2000, "category": "ssl"},
            
            # 元学习与少样本学习
            {"arxiv_id": "1703.03400", "title": "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", "estimated_citations": 4000, "category": "meta"},
            {"arxiv_id": "1606.04080", "title": "Matching Networks for One Shot Learning", "estimated_citations": 3000, "category": "few_shot"},
            {"arxiv_id": "1703.05175", "title": "Prototypical Networks for Few-shot Learning", "estimated_citations": 2000, "category": "few_shot"},
            
            # 可解释AI
            {"arxiv_id": "1602.04938", "title": "Why Should I Trust You?: Explaining Predictions", "estimated_citations": 6000, "category": "explainable"},
            {"arxiv_id": "1311.2901", "title": "Deep Inside Convolutional Networks: Visualising Image Classification", "estimated_citations": 4000, "category": "explainable"},
            {"arxiv_id": "1703.01365", "title": "A Unified Approach to Interpreting Model Predictions", "estimated_citations": 5000, "category": "explainable"},
            
            # 优化与正则化
            {"arxiv_id": "1207.0580", "title": "Improving neural networks by preventing co-adaptation", "estimated_citations": 15000, "category": "regularization"},
            {"arxiv_id": "1506.02142", "title": "Batch Normalization: Accelerating Deep Network Training", "estimated_citations": 12000, "category": "normalization"},
            {"arxiv_id": "1607.06450", "title": "Layer Normalization", "estimated_citations": 4000, "category": "normalization"},
            {"arxiv_id": "1910.05446", "title": "Weight Standardization", "estimated_citations": 500, "category": "normalization"},
            
            # 近期重要论文
            {"arxiv_id": "2010.11929", "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition", "estimated_citations": 3000, "category": "vision_transformer"},
            {"arxiv_id": "2103.00020", "title": "Learning Transferable Visual Models From Natural Language Supervision", "estimated_citations": 2000, "category": "multimodal"},
            {"arxiv_id": "2005.12872", "title": "GPT-4 Technical Report", "estimated_citations": 1000, "category": "gpt"},
            {"arxiv_id": "2203.15556", "title": "PaLM: Scaling Language Modeling with Pathways", "estimated_citations": 1000, "category": "llm"},
            {"arxiv_id": "2204.02311", "title": "PaLM-2 Technical Report", "estimated_citations": 500, "category": "llm"},
            
            # 多模态学习
            {"arxiv_id": "1908.03557", "title": "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations", "estimated_citations": 1000, "category": "multimodal"},
            {"arxiv_id": "2102.03334", "title": "ALIGN: Scaling Up Visual and Vision-Language Representation Learning", "estimated_citations": 800, "category": "multimodal"},
            
            # 知识蒸馏与模型压缩
            {"arxiv_id": "1503.02531", "title": "Distilling the Knowledge in a Neural Network", "estimated_citations": 8000, "category": "distillation"},
            {"arxiv_id": "1910.01108", "title": "DistilBERT, a distilled version of BERT", "estimated_citations": 2000, "category": "distillation"},
            {"arxiv_id": "1609.07061", "title": "Pruning Filters for Efficient ConvNets", "estimated_citations": 3000, "category": "compression"},
        ]
        
        self.collected_papers = []
        
    def get_arxiv_paper(self, arxiv_id: str) -> Optional[Dict]:
        """从arXiv获取论文详情"""
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                return self._parse_arxiv_response(response.text, arxiv_id)
            else:
                print(f"  获取失败: HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"  请求异常: {e}")
            return None
    
    def _parse_arxiv_response(self, xml_content: str, arxiv_id: str) -> Optional[Dict]:
        """解析arXiv XML响应"""
        try:
            import xml.etree.ElementTree as ET
            
            root = ET.fromstring(xml_content)
            
            # 定义命名空间
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # 查找entry元素
            entry = root.find('atom:entry', ns)
            if entry is None:
                return None
            
            # 提取信息
            title = entry.find('atom:title', ns)
            title_text = title.text.strip() if title is not None else ""
            
            summary = entry.find('atom:summary', ns)
            abstract_text = summary.text.strip() if summary is not None else ""
            
            # 提取作者
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text.strip())
            
            # 提取发布日期
            published = entry.find('atom:published', ns)
            published_date = published.text[:10] if published is not None else ""
            
            # 提取类别
            categories = []
            for category in entry.findall('atom:category', ns):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # 构建论文信息
            paper = {
                'id': arxiv_id,
                'title': title_text,
                'abstract': abstract_text,
                'authors': authors,
                'published': published_date,
                'url': f"https://arxiv.org/abs/{arxiv_id}",
                'categories': categories,
                'source': 'arxiv'
            }
            
            return paper if paper['title'] and paper['abstract'] else None
            
        except Exception as e:
            print(f"  解析XML失败: {e}")
            return None
    
    def collect_papers(self, target_count: int = 100) -> List[Dict]:
        """收集经典论文"""
        print(f"开始收集 {min(target_count, len(self.classic_papers))} 篇经典AI/ML论文...")
        print("=" * 60)
        
        # 按预估引用量排序
        sorted_papers = sorted(
            self.classic_papers, 
            key=lambda x: x['estimated_citations'], 
            reverse=True
        )
        
        collected_count = 0
        failed_count = 0
        
        for i, paper_info in enumerate(sorted_papers):
            if collected_count >= target_count:
                break
            
            arxiv_id = paper_info['arxiv_id']
            estimated_citations = paper_info['estimated_citations']
            category = paper_info['category']
            
            print(f"\n进度: {collected_count + 1}/{target_count} - {arxiv_id} ({category})")
            print(f"预估引用: {estimated_citations:,}")
            
            # 获取论文详情
            paper = self.get_arxiv_paper(arxiv_id)
            
            if paper:
                # 添加元信息
                paper['estimated_citations'] = estimated_citations
                paper['category'] = category
                paper['quality_score'] = self._calculate_quality_score(paper, estimated_citations)
                paper['year'] = self._extract_year_from_date(paper.get('published', ''))
                
                self.collected_papers.append(paper)
                collected_count += 1
                
                print(f"  ✅ {paper['title'][:60]}...")
                print(f"     作者: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
                print(f"     年份: {paper.get('year', 'N/A')}")
            else:
                failed_count += 1
                print(f"  ❌ 获取失败")
            
            # 控制请求频率
            time.sleep(0.5)
        
        print(f"\n收集完成！")
        print(f"成功: {collected_count} 篇")
        print(f"失败: {failed_count} 篇")
        
        return self.collected_papers
    
    def _calculate_quality_score(self, paper: Dict, estimated_citations: int) -> float:
        """计算质量分数"""
        # 基于预估引用量的质量分数
        citation_score = min(estimated_citations / 10000, 2.0)  # 归一化
        
        # 基于内容质量的加成
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '')
        
        content_bonus = 0
        # 标题包含重要关键词
        important_keywords = ['transformer', 'attention', 'bert', 'gpt', 'neural', 'deep', 'learning']
        for keyword in important_keywords:
            if keyword in title:
                content_bonus += 0.1
        
        # 摘要长度和质量
        if len(abstract) > 1000:
            content_bonus += 0.2
        elif len(abstract) > 500:
            content_bonus += 0.1
        
        return citation_score + content_bonus
    
    def _extract_year_from_date(self, date_str: str) -> Optional[int]:
        """从日期字符串提取年份"""
        if not date_str:
            return None
        try:
            return int(date_str[:4])
        except:
            return None
    
    def save_papers(self, output_file: str = "data/high_quality_papers.json"):
        """保存收集的论文"""
        os.makedirs("data", exist_ok=True)
        
        # 按质量分数排序
        self.collected_papers.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # 生成统计信息
        stats = {
            'total_papers': len(self.collected_papers),
            'collection_date': datetime.now().isoformat(),
            'avg_estimated_citations': sum(p.get('estimated_citations', 0) for p in self.collected_papers) / len(self.collected_papers) if self.collected_papers else 0,
            'avg_quality_score': sum(p.get('quality_score', 0) for p in self.collected_papers) / len(self.collected_papers) if self.collected_papers else 0,
            'category_distribution': {},
            'year_distribution': {}
        }
        
        # 统计分布
        for paper in self.collected_papers:
            category = paper.get('category', 'unknown')
            year = paper.get('year')
            
            stats['category_distribution'][category] = stats['category_distribution'].get(category, 0) + 1
            if year:
                stats['year_distribution'][str(year)] = stats['year_distribution'].get(str(year), 0) + 1
        
        # 保存数据
        final_data = {
            'papers': self.collected_papers,
            'statistics': stats
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n论文数据已保存: {output_file}")
        print(f"统计信息:")
        print(f"  总论文数: {stats['total_papers']}")
        print(f"  平均预估引用: {stats['avg_estimated_citations']:,.0f}")
        print(f"  平均质量分数: {stats['avg_quality_score']:.2f}")
        
        # 显示类别分布
        print(f"\n类别分布:")
        for category, count in sorted(stats['category_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count} 篇")
        
        # 显示年份分布
        print(f"\n年份分布:")
        for year in sorted(stats['year_distribution'].keys(), reverse=True)[:8]:
            count = stats['year_distribution'][year]
            print(f"  {year}: {count} 篇")
        
        return output_file


def main():
    """主函数"""
    print("经典高质量AI/ML论文收集器")
    print("基于预定义的里程碑式论文列表")
    print("=" * 60)
    
    collector = ClassicPaperCollector()
    
    try:
        # 收集论文
        papers = collector.collect_papers(target_count=100)
        
        if papers:
            # 保存结果
            output_file = collector.save_papers()
            
            print(f"\n✅ 成功收集了 {len(papers)} 篇经典AI/ML论文！")
            print(f"\n包含了以下重要论文系列:")
            print(f"• Transformer系列 (Attention Is All You Need, BERT, GPT等)")
            print(f"• 深度学习基础 (ResNet, VGG, Dropout等)")
            print(f"• 计算机视觉 (CNN, 目标检测, 图像分割等)")
            print(f"• 自然语言处理 (Word2Vec, Seq2Seq, 注意力机制等)")
            print(f"• 生成模型 (GAN, VAE等)")
            print(f"• 强化学习 (DQN, Policy Gradient等)")
            print(f"• 图神经网络 (GCN, GAT等)")
            
            print(f"\n下一步:")
            print(f"1. 查看收集结果: {output_file}")
            print(f"2. 运行整合程序: python integrate_papers.py")
            print(f"3. 测试新的RAG系统: python main_rag_system.py")
        else:
            print("❌ 未收集到论文")
    
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        if collector.collected_papers:
            print(f"已收集 {len(collector.collected_papers)} 篇论文，正在保存...")
            collector.save_papers()
    
    except Exception as e:
        print(f"\n❌ 收集过程出现错误: {e}")
        if collector.collected_papers:
            print(f"尝试保存已收集的 {len(collector.collected_papers)} 篇论文...")
            collector.save_papers()


if __name__ == "__main__":
    main()