#!/usr/bin/env python3
"""
论文数据整合器
将新收集的高质量论文与现有论文数据合并，去重并优化
"""

import json
import os
from typing import List, Dict, Set
from datetime import datetime

class PaperIntegrator:
    """论文数据整合器"""
    
    def __init__(self):
        self.existing_papers = []
        self.new_papers = []
        self.integrated_papers = []
        self.duplicate_count = 0
        
    def load_existing_papers(self, file_path: str = "data/papers_info.json") -> bool:
        """加载现有论文数据"""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.existing_papers = json.load(f)
                print(f"✅ 加载现有论文: {len(self.existing_papers)} 篇")
                return True
            except Exception as e:
                print(f"❌ 加载现有论文失败: {e}")
                return False
        else:
            print("📝 未找到现有论文数据，将创建新文件")
            return True
    
    def load_new_papers(self, file_path: str = "data/high_quality_papers.json") -> bool:
        """加载新收集的论文数据"""
        if not os.path.exists(file_path):
            print(f"❌ 未找到新论文数据文件: {file_path}")
            print("请先运行: python collect_high_quality_papers.py")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 支持两种格式
            if 'papers' in data:
                self.new_papers = data['papers']
                stats = data.get('statistics', {})
                print(f"✅ 加载新收集论文: {len(self.new_papers)} 篇")
                print(f"   平均引用量: {stats.get('avg_citations', 0):.1f}")
                print(f"   平均质量分数: {stats.get('avg_quality_score', 0):.2f}")
            else:
                self.new_papers = data
                print(f"✅ 加载新收集论文: {len(self.new_papers)} 篇")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载新论文失败: {e}")
            return False
    
    def _normalize_title(self, title: str) -> str:
        """标准化标题用于去重"""
        if not title:
            return ""
        
        # 转小写，移除标点符号和多余空格
        import re
        normalized = re.sub(r'[^\w\s]', ' ', title.lower())
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _are_papers_duplicate(self, paper1: Dict, paper2: Dict) -> bool:
        """检查两篇论文是否重复"""
        
        # 1. 检查ID
        id1 = paper1.get('id', '')
        id2 = paper2.get('id', '')
        if id1 and id2 and id1 == id2:
            return True
        
        # 2. 检查标题相似度
        title1 = self._normalize_title(paper1.get('title', ''))
        title2 = self._normalize_title(paper2.get('title', ''))
        
        if not title1 or not title2:
            return False
        
        # 简单的字符串相似度检查
        if title1 == title2:
            return True
        
        # 检查标题包含关系（处理标题长短不一的情况）
        if len(title1) > len(title2):
            longer, shorter = title1, title2
        else:
            longer, shorter = title2, title1
        
        # 如果短标题是长标题的子集且相似度很高
        if shorter in longer and len(shorter) / len(longer) > 0.8:
            return True
        
        return False
    
    def integrate_papers(self) -> List[Dict]:
        """整合论文数据"""
        print("\n开始整合论文数据...")
        
        # 创建现有论文的标题索引
        existing_titles = set()
        for paper in self.existing_papers:
            title = self._normalize_title(paper.get('title', ''))
            if title:
                existing_titles.add(title)
        
        self.integrated_papers = self.existing_papers.copy()
        
        # 处理新论文
        for new_paper in self.new_papers:
            is_duplicate = False
            
            # 检查是否与现有论文重复
            for existing_paper in self.existing_papers:
                if self._are_papers_duplicate(new_paper, existing_paper):
                    is_duplicate = True
                    self.duplicate_count += 1
                    break
            
            if not is_duplicate:
                # 标准化新论文格式
                standardized_paper = self._standardize_paper_format(new_paper)
                if standardized_paper:
                    self.integrated_papers.append(standardized_paper)
        
        print(f"✅ 整合完成:")
        print(f"   原有论文: {len(self.existing_papers)} 篇")
        print(f"   新增论文: {len(self.integrated_papers) - len(self.existing_papers)} 篇")
        print(f"   重复论文: {self.duplicate_count} 篇")
        print(f"   总计论文: {len(self.integrated_papers)} 篇")
        
        return self.integrated_papers
    
    def _standardize_paper_format(self, paper: Dict) -> Dict:
        """标准化论文格式"""
        try:
            # 确保必需字段存在
            standardized = {
                'id': paper.get('id', f"paper_{hash(paper.get('title', ''))}"[:16]),
                'title': paper.get('title', '').strip(),
                'abstract': paper.get('abstract', '').strip(),
                'authors': paper.get('authors', []),
                'published': paper.get('published', paper.get('publicationDate', '')),
                'venue': paper.get('venue', ''),
                'url': paper.get('url', ''),
                'source': paper.get('source', 'collected')
            }
            
            # 添加质量指标（如果有的话）
            if 'citationCount' in paper:
                standardized['citationCount'] = paper['citationCount']
            if 'quality_score' in paper:
                standardized['quality_score'] = paper['quality_score']
            if 'year' in paper:
                standardized['year'] = paper['year']
            
            # 验证必要字段
            if not standardized['title'] or len(standardized['title']) < 5:
                return None
            
            if not standardized['abstract'] or len(standardized['abstract']) < 20:
                return None
            
            return standardized
            
        except Exception as e:
            print(f"标准化论文格式失败: {e}")
            return None
    
    def save_integrated_papers(self, output_file: str = "data/papers_info.json"):
        """保存整合后的论文数据"""
        
        # 按质量排序（如果有质量指标的话）
        def sort_key(paper):
            citation_count = paper.get('citationCount', 0)
            quality_score = paper.get('quality_score', 0)
            year = paper.get('year', 2020)
            
            # 综合排序：引用量 * 0.6 + 质量分数 * 0.3 + 年份权重 * 0.1
            year_weight = (year - 2015) / 10 if year else 0  # 年份权重
            total_score = citation_count * 0.6 + quality_score * 100 * 0.3 + year_weight * 10
            return total_score
        
        self.integrated_papers.sort(key=sort_key, reverse=True)
        
        # 创建备份
        if os.path.exists(output_file):
            backup_file = f"{output_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(output_file, backup_file)
            print(f"📋 创建备份: {backup_file}")
        
        # 保存整合数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.integrated_papers, f, ensure_ascii=False, indent=2)
        
        print(f"💾 整合数据已保存: {output_file}")
        
        # 生成统计报告
        self._generate_statistics_report()
        
        return output_file
    
    def _generate_statistics_report(self):
        """生成统计报告"""
        if not self.integrated_papers:
            return
        
        print(f"\n📊 整合数据统计报告:")
        print("=" * 50)
        
        # 基础统计
        total_papers = len(self.integrated_papers)
        papers_with_citations = sum(1 for p in self.integrated_papers if p.get('citationCount', 0) > 0)
        
        print(f"论文总数: {total_papers}")
        print(f"有引用数据: {papers_with_citations} 篇")
        
        # 引用量统计
        if papers_with_citations > 0:
            citations = [p.get('citationCount', 0) for p in self.integrated_papers if p.get('citationCount', 0) > 0]
            avg_citations = sum(citations) / len(citations)
            max_citations = max(citations)
            print(f"平均引用量: {avg_citations:.1f}")
            print(f"最高引用量: {max_citations}")
            
            # 高引用论文统计
            high_citation_papers = [p for p in self.integrated_papers if p.get('citationCount', 0) >= 100]
            print(f"高引用论文 (≥100): {len(high_citation_papers)} 篇")
        
        # 年份分布
        year_dist = {}
        for paper in self.integrated_papers:
            year = paper.get('year')
            if year:
                year_dist[year] = year_dist.get(year, 0) + 1
        
        if year_dist:
            print(f"\n年份分布:")
            for year in sorted(year_dist.keys(), reverse=True)[:5]:
                print(f"  {year}: {year_dist[year]} 篇")
        
        # 来源统计
        source_dist = {}
        for paper in self.integrated_papers:
            source = paper.get('source', 'unknown')
            source_dist[source] = source_dist.get(source, 0) + 1
        
        if source_dist:
            print(f"\n数据来源:")
            for source, count in sorted(source_dist.items(), key=lambda x: x[1], reverse=True):
                print(f"  {source}: {count} 篇")
        
        # 质量分布
        quality_papers = [p for p in self.integrated_papers if 'quality_score' in p]
        if quality_papers:
            avg_quality = sum(p['quality_score'] for p in quality_papers) / len(quality_papers)
            print(f"\n质量评分:")
            print(f"  平均质量分数: {avg_quality:.2f}")
            print(f"  有质量评分: {len(quality_papers)} 篇")


def main():
    """主函数"""
    print("论文数据整合器")
    print("=" * 60)
    
    integrator = PaperIntegrator()
    
    try:
        # 1. 加载现有数据
        if not integrator.load_existing_papers():
            print("加载现有数据失败")
            return
        
        # 2. 加载新收集的数据
        if not integrator.load_new_papers():
            print("加载新数据失败，请先运行收集程序")
            return
        
        # 3. 整合数据
        integrated_papers = integrator.integrate_papers()
        
        if integrated_papers:
            # 4. 保存整合结果
            output_file = integrator.save_integrated_papers()
            
            print(f"\n✅ 数据整合完成！")
            print(f"\n下一步:")
            print(f"1. 查看整合结果: {output_file}")
            print(f"2. 测试新系统: python main_rag_system.py")
            print(f"3. 或运行测试: python test_direct_answer.py")
        else:
            print("❌ 整合失败，没有有效的论文数据")
    
    except Exception as e:
        print(f"❌ 整合过程出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()