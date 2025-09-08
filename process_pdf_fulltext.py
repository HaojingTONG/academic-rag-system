#!/usr/bin/env python3
"""
PDF全文处理脚本
批量处理raw_papers中的PDF文件，提取完整内容并更新论文数据
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import time

# 添加src到路径  
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'src'))

from src.processor.pdf_processor import AcademicPDFProcessor
from src.processor.document_chunker import DocumentChunker, ChunkingConfig

def process_pdf_fulltext():
    """处理所有PDF的全文内容"""
    print("🚀 开始PDF全文处理")
    print("=" * 80)
    
    # 初始化处理器
    pdf_processor = AcademicPDFProcessor()
    
    # 配置智能分块器
    chunking_config = ChunkingConfig(
        strategy='hybrid',
        chunk_size=600,
        chunk_overlap=100,
        min_chunk_size=100,
        preserve_sentences=True,
        section_aware=True
    )
    chunker = DocumentChunker(chunking_config)
    
    # 加载现有论文数据
    papers_file = Path("data/main_system_papers.json")
    if not papers_file.exists():
        print("❌ 论文数据文件不存在: data/main_system_papers.json")
        return False
    
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers_data = json.load(f)
    
    print(f"📚 加载论文数据: {len(papers_data)} 篇")
    
    # PDF文件目录
    pdf_dir = Path("data/raw_papers")
    if not pdf_dir.exists():
        print("❌ PDF目录不存在: data/raw_papers")
        return False
    
    pdf_files = {f.stem: f for f in pdf_dir.glob("*.pdf")}
    print(f"📄 找到PDF文件: {len(pdf_files)} 个")
    
    # 统计信息
    processed_count = 0
    enhanced_count = 0
    failed_count = 0
    
    # 处理每篇论文
    for i, paper in enumerate(papers_data):
        paper_id = paper.get('id', '')
        title = paper.get('title', 'Unknown')[:50]
        
        print(f"\n📖 [{i+1}/{len(papers_data)}] {title}...")
        print(f"   ID: {paper_id}")
        
        # 检查是否有对应的PDF文件
        if paper_id not in pdf_files:
            print(f"   ⚠️ 未找到PDF文件: {paper_id}.pdf")
            continue
        
        pdf_path = pdf_files[paper_id]
        
        # 检查是否已经处理过全文
        existing_chunks = paper.get('processed_chunks', [])
        if len(existing_chunks) > 1:
            section_types = [chunk['metadata']['section_type'] for chunk in existing_chunks]
            if any(stype != 'abstract' for stype in section_types):
                print(f"   ✅ 已处理全文 ({len(existing_chunks)} 块)")
                processed_count += 1
                continue
        
        # 提取PDF全文内容
        print(f"   🔄 提取PDF全文...")
        pdf_content = pdf_processor.extract_pdf_content(str(pdf_path))
        
        if not pdf_content:
            print(f"   ❌ PDF提取失败")
            failed_count += 1
            continue
        
        # 准备文档用于分块
        documents_for_chunking = []
        
        # 添加摘要（如果存在）
        if pdf_content.abstract:
            documents_for_chunking.append({
                'content': f"Title: {pdf_content.title}\n\nAbstract: {pdf_content.abstract}",
                'metadata': {
                    'section_type': 'abstract',
                    'title': pdf_content.title
                }
            })
        
        # 添加各个章节
        for section in pdf_content.sections:
            if section.content.strip() and len(section.content) > 200:  # 过滤太短的章节
                documents_for_chunking.append({
                    'content': f"Section: {section.title}\n\n{section.content}",
                    'metadata': {
                        'section_type': section.section_type,
                        'section_title': section.title,
                        'page_range': f"{section.page_range[0]}-{section.page_range[1]}",
                        'confidence': section.confidence
                    }
                })
        
        if not documents_for_chunking:
            print(f"   ❌ 无有效内容可处理")
            failed_count += 1
            continue
        
        print(f"   📑 准备分块: {len(documents_for_chunking)} 个章节")
        
        # 智能分块处理
        all_chunks = []
        for doc in documents_for_chunking:
            try:
                chunks = chunker.chunk_document(
                    text=doc['content'],
                    paper_id=paper_id,
                    metadata=doc['metadata']
                )
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"   ⚠️ 分块失败: {e}")
                continue
        
        if not all_chunks:
            print(f"   ❌ 分块处理失败")
            failed_count += 1
            continue
        
        # 转换为论文数据格式
        processed_chunks = []
        for chunk in all_chunks:
            chunk_data = {
                'text': chunk.text,
                'metadata': {
                    'chunk_id': chunk.chunk_id,
                    'paper_id': paper_id,
                    'title': pdf_content.title,
                    'section_type': chunk.metadata.get('section_type', 'content'),
                    'word_count': chunk.word_count,
                    'char_count': chunk.char_count,
                    'has_formulas': pdf_content.has_formulas,
                    'has_code': False,  # 可以后续扩展
                    'has_citations': 'references' in chunk.text.lower(),
                    'section_title': chunk.metadata.get('section_title', ''),
                    'page_range': chunk.metadata.get('page_range', ''),
                    'language': pdf_content.language,
                    'total_pages': pdf_content.total_pages,
                    'processing_version': '2.0'  # 标记为全文处理版本
                }
            }
            processed_chunks.append(chunk_data)
        
        # 更新论文数据
        paper['processed_chunks'] = processed_chunks
        paper['pdf_processed'] = True
        paper['pdf_stats'] = {
            'total_pages': pdf_content.total_pages,
            'total_words': pdf_content.total_words,
            'sections_count': len(pdf_content.sections),
            'chunks_count': len(processed_chunks),
            'has_formulas': pdf_content.has_formulas,
            'has_tables': pdf_content.has_tables,
            'has_figures': pdf_content.has_figures,
            'language': pdf_content.language
        }
        
        enhanced_count += 1
        print(f"   ✅ 全文处理完成: {len(processed_chunks)} 个文档块")
        
        # 每处理10篇论文保存一次
        if enhanced_count % 10 == 0:
            print(f"\n💾 中间保存... (已处理 {enhanced_count} 篇)")
            with open(papers_file, 'w', encoding='utf-8') as f:
                json.dump(papers_data, f, ensure_ascii=False, indent=2)
    
    # 最终保存
    print(f"\n💾 保存最终结果...")
    with open(papers_file, 'w', encoding='utf-8') as f:
        json.dump(papers_data, f, ensure_ascii=False, indent=2)
    
    # 生成统计报告
    print(f"\n" + "=" * 80)
    print(f"📊 PDF全文处理完成!")
    print(f"=" * 80)
    print(f"✅ 新处理论文: {enhanced_count} 篇")
    print(f"✅ 已处理论文: {processed_count} 篇") 
    print(f"❌ 处理失败: {failed_count} 篇")
    print(f"📚 总论文数: {len(papers_data)} 篇")
    
    if enhanced_count > 0:
        print(f"\n🎯 处理效果:")
        
        # 统计平均文档块数
        total_chunks = 0
        enhanced_papers = 0
        
        for paper in papers_data:
            chunks = paper.get('processed_chunks', [])
            if len(chunks) > 1 and paper.get('pdf_processed', False):
                total_chunks += len(chunks)
                enhanced_papers += 1
        
        if enhanced_papers > 0:
            avg_chunks = total_chunks / enhanced_papers
            print(f"   📑 平均文档块数: {avg_chunks:.1f} 个/论文")
            print(f"   🚀 检索质量将显著提升!")
    
    return enhanced_count > 0

def main():
    """主函数"""
    try:
        success = process_pdf_fulltext()
        return 0 if success else 1
    except KeyboardInterrupt:
        print(f"\n\n⏹️ 用户中断处理")
        return 1
    except Exception as e:
        print(f"\n❌ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())