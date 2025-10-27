#!/usr/bin/env python3
"""
收集高质量AI/ML论文
- 使用Semantic Scholar API获取高引用量论文
- 重点关注深度学习、机器学习、自然语言处理等核心领域
- 优先选择近年来的高影响力论文
"""

import requests
import json
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import random

class HighQualityPaperCollector:
    """高质量论文收集器"""
    
    def __init__(self):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.session = requests.Session()
        
        # 设置请求头
        self.session.headers.update({
            'User-Agent': 'Academic-RAG-System/1.0 (haojingtong@example.com)'
        })
        
        # 核心AI/ML关键词 - 按重要性排序
        self.core_keywords = [
            # 深度学习核心
            "transformer attention mechanism",
            "BERT language model", 
            "GPT generative model",
            "neural machine translation",
            "convolutional neural network",
            "recurrent neural network",
            "deep reinforcement learning",
            
            # 机器学习基础
            "machine learning algorithm",
            "supervised learning",
            "unsupervised learning", 
            "transfer learning",
            "meta learning",
            "few shot learning",
            
            # 计算机视觉
            "computer vision deep learning",
            "object detection",
            "image classification",
            "semantic segmentation",
            
            # 自然语言处理
            "natural language processing",
            "text classification",
            "sentiment analysis",
            "question answering",
            
            # 前沿技术
            "graph neural network",
            "generative adversarial network",
            "variational autoencoder",
            "self supervised learning"
        ]
        
        # 知名会议和期刊（用于质量过滤）
        self.top_venues = [
            'NeurIPS', 'ICML', 'ICLR', 'AAAI', 'IJCAI', 'ACL', 'EMNLP', 
            'CVPR', 'ICCV', 'ECCV', 'Nature', 'Science', 'JMLR', 
            'IEEE TPAMI', 'TACL', 'CoRR'
        ]
        
        self.collected_papers = []
        self.paper_ids_seen = set()
        
    def search_papers_by_keyword(self, keyword: str, limit: int = 20, 
                                min_citations: int = 50) -> List[Dict]:
        """根据关键词搜索论文"""
        
        url = f"{self.base_url}/paper/search"
        
        # 计算时间范围 - 优先近5年的论文
        end_year = datetime.now().year
        start_year = end_year - 5
        
        params = {
            'query': keyword,
            'limit': limit,
            'fields': 'paperId,title,abstract,authors,year,citationCount,venue,publicationDate,url,references,citations',
            'publicationDateOrYear': f'{start_year}-{end_year}'
        }
        
        try:
            print(f"搜索关键词: {keyword}")
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                papers = data.get('data', [])
                
                # 过滤高引用量论文
                high_quality_papers = []
                for paper in papers:
                    citation_count = paper.get('citationCount', 0)
                    year = paper.get('year')
                    paper_id = paper.get('paperId')
                    
                    # 跳过已见过的论文
                    if paper_id in self.paper_ids_seen:
                        continue
                    
                    # 引用量过滤 - 根据年份动态调整阈值
                    if year and year >= 2020:
                        # 近期论文降低引用量要求
                        min_citations_adjusted = max(min_citations // 3, 10)
                    elif year and year >= 2018:
                        min_citations_adjusted = min_citations // 2
                    else:
                        min_citations_adjusted = min_citations
                    
                    if citation_count >= min_citations_adjusted:
                        # 检查是否来自知名会议/期刊
                        venue = paper.get('venue', '')
                        venue_bonus = any(top_venue.lower() in venue.lower() 
                                        for top_venue in self.top_venues)
                        
                        # 计算质量分数
                        quality_score = self._calculate_quality_score(paper, venue_bonus)
                        paper['quality_score'] = quality_score
                        
                        high_quality_papers.append(paper)
                        self.paper_ids_seen.add(paper_id)
                
                print(f"  找到 {len(high_quality_papers)} 篇高质量论文")
                return high_quality_papers
                
            elif response.status_code == 429:
                print("  API限流，等待...")
                time.sleep(5)
                return []
            else:
                print(f"  搜索失败: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"  搜索异常: {e}")
            return []
    
    def _calculate_quality_score(self, paper: Dict, venue_bonus: bool = False) -> float:
        """计算论文质量分数"""
        citation_count = paper.get('citationCount', 0)
        year = paper.get('year', 2020)
        current_year = datetime.now().year
        
        # 基础分数：引用量
        base_score = min(citation_count / 1000, 1.0)  # 归一化到[0,1]
        
        # 时间衰减：越新的论文给予更高权重
        years_ago = max(current_year - year, 1)
        time_factor = 1.0 / (1 + years_ago * 0.1)  # 轻微的时间衰减
        
        # 会议/期刊加成
        venue_factor = 1.2 if venue_bonus else 1.0
        
        # 标题和摘要质量（简单启发式）
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        
        content_factor = 1.0
        if len(abstract) > 500:  # 摘要详细
            content_factor += 0.1
        if any(keyword in title.lower() for keyword in ['deep', 'neural', 'learning', 'attention']):
            content_factor += 0.1
        
        # 综合质量分数
        quality_score = base_score * time_factor * venue_factor * content_factor
        
        return min(quality_score, 2.0)  # 限制最高分数
    
    def get_paper_details(self, paper_id: str) -> Optional[Dict]:
        """获取论文详细信息"""
        url = f"{self.base_url}/paper/{paper_id}"
        
        params = {
            'fields': 'paperId,title,abstract,authors,year,citationCount,venue,publicationDate,url,publicationVenue,fieldsOfStudy'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            print(f"获取论文详情失败: {e}")
            return None
    
    def collect_papers(self, target_count: int = 100) -> List[Dict]:
        """收集高质量论文"""
        print(f"开始收集 {target_count} 篇高质量AI/ML论文...")
        print("=" * 60)
        
        # 随机打乱关键词顺序，获得更多样性
        keywords = self.core_keywords.copy()
        random.shuffle(keywords)
        
        collected_count = 0
        
        for i, keyword in enumerate(keywords):
            if collected_count >= target_count:
                break
                
            print(f"\n进度: {collected_count}/{target_count} - 搜索 {keyword}")
            
            # 动态调整搜索参数
            remaining = target_count - collected_count
            limit = min(max(remaining // (len(keywords) - i), 10), 100)
            
            # 根据已收集数量调整引用量阈值
            if collected_count < target_count // 3:
                min_citations = 100  # 前1/3要求更高引用量
            elif collected_count < target_count * 2 // 3:
                min_citations = 50   # 中间1/3适中引用量
            else:
                min_citations = 20   # 后1/3降低要求以确保数量
            
            papers = self.search_papers_by_keyword(
                keyword, 
                limit=limit, 
                min_citations=min_citations
            )
            
            # 按质量分数排序
            papers.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
            # 添加到收集列表
            for paper in papers[:limit]:
                if collected_count >= target_count:
                    break
                
                # 进一步处理论文信息
                processed_paper = self._process_paper(paper)
                if processed_paper:
                    self.collected_papers.append(processed_paper)
                    collected_count += 1
                    
                    print(f"  ✅ {collected_count:3d}. {processed_paper['title'][:50]}... "
                          f"(引用: {processed_paper['citationCount']}, "
                          f"质量: {processed_paper['quality_score']:.2f})")
            
            # API限流控制
            time.sleep(1)
        
        print(f"\n收集完成！共获得 {len(self.collected_papers)} 篇高质量论文")
        return self.collected_papers
    
    def _process_paper(self, paper: Dict) -> Optional[Dict]:
        """处理论文信息"""
        try:
            # 提取作者信息
            authors = []
            for author in paper.get('authors', []):
                if isinstance(author, dict):
                    name = author.get('name', '')
                    if name:
                        authors.append(name)
                else:
                    authors.append(str(author))
            
            # 生成唯一ID
            paper_id = paper.get('paperId', '')
            if not paper_id:
                paper_id = f"semantic_{hash(paper.get('title', ''))}"
            
            # 获取发布日期
            pub_date = paper.get('publicationDate', '')
            if not pub_date and paper.get('year'):
                pub_date = f"{paper['year']}-01-01"
            
            processed = {
                'id': paper_id,
                'title': paper.get('title', '').strip(),
                'abstract': paper.get('abstract', '').strip(),
                'authors': authors[:5],  # 限制作者数量
                'published': pub_date,
                'citationCount': paper.get('citationCount', 0),
                'venue': paper.get('venue', ''),
                'year': paper.get('year'),
                'quality_score': paper.get('quality_score', 0),
                'url': paper.get('url', ''),
                'source': 'semantic_scholar'
            }
            
            # 验证必要字段
            if not processed['title'] or len(processed['title']) < 10:
                return None
            
            if not processed['abstract'] or len(processed['abstract']) < 50:
                return None
            
            return processed
            
        except Exception as e:
            print(f"处理论文失败: {e}")
            return None
    
    def save_papers(self, output_file: str = "data/high_quality_papers.json"):
        """保存收集的论文"""
        # 确保data目录存在
        os.makedirs("data", exist_ok=True)
        
        # 按引用量和质量分数排序
        self.collected_papers.sort(
            key=lambda x: (x.get('citationCount', 0) * 0.7 + 
                          x.get('quality_score', 0) * 100 * 0.3), 
            reverse=True
        )
        
        # 保存详细统计信息
        stats = {
            'total_papers': len(self.collected_papers),
            'collection_date': datetime.now().isoformat(),
            'avg_citations': sum(p.get('citationCount', 0) for p in self.collected_papers) / len(self.collected_papers) if self.collected_papers else 0,
            'avg_quality_score': sum(p.get('quality_score', 0) for p in self.collected_papers) / len(self.collected_papers) if self.collected_papers else 0,
            'year_distribution': {},
            'venue_distribution': {}
        }
        
        # 统计年份和会议分布
        for paper in self.collected_papers:
            year = paper.get('year')
            venue = paper.get('venue', 'Unknown')
            
            if year:
                stats['year_distribution'][str(year)] = stats['year_distribution'].get(str(year), 0) + 1
            
            stats['venue_distribution'][venue] = stats['venue_distribution'].get(venue, 0) + 1
        
        # 保存数据
        final_data = {
            'papers': self.collected_papers,
            'statistics': stats
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n论文数据已保存到: {output_file}")
        print(f"统计信息:")
        print(f"  总论文数: {stats['total_papers']}")
        print(f"  平均引用量: {stats['avg_citations']:.1f}")
        print(f"  平均质量分数: {stats['avg_quality_score']:.2f}")
        
        # 显示年份分布
        print(f"  年份分布:")
        for year in sorted(stats['year_distribution'].keys(), reverse=True):
            count = stats['year_distribution'][year]
            print(f"    {year}: {count} 篇")
        
        return output_file


def main():
    """主函数"""
    print("高质量AI/ML论文收集器")
    print("=" * 60)
    
    # 初始化收集器
    collector = HighQualityPaperCollector()
    
    try:
        # 收集论文
        papers = collector.collect_papers(target_count=100)
        
        if papers:
            # 保存结果
            output_file = collector.save_papers()
            
            print(f"\n✅ 成功收集了 {len(papers)} 篇高质量AI/ML论文！")
            print(f"\n下一步:")
            print(f"1. 查看收集结果: {output_file}")
            print(f"2. 运行整合程序: python integrate_papers.py")
            print(f"3. 测试新的RAG系统: python main_rag_system.py")
        else:
            print("❌ 未收集到论文，请检查网络连接或API限制")
    
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