#!/usr/bin/env python3
"""
测试短章节处理功能
验证短章节不会被误删，重要信息得到保留
"""

import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'src'))

from src.processor.pdf_processor import AcademicPDFProcessor
from short_section_handler import ShortSectionHandler


def test_short_section_detection():
    """测试短章节检测功能"""

    print("🧪 测试短章节处理功能")
    print("=" * 60)

    # 1. 测试PDF处理
    pdf_processor = AcademicPDFProcessor()
    short_handler = ShortSectionHandler()

    # 选择一个测试文件
    import glob
    pdf_files = glob.glob("data/raw_papers/*.pdf")

    if not pdf_files:
        print("❌ 未找到测试PDF文件")
        return

    test_file = pdf_files[0]
    filename = test_file.split('/')[-1]

    print(f"📄 测试文件: {filename}")

    # 2. 提取PDF内容
    pdf_content = pdf_processor.extract_pdf_content(test_file)

    if not pdf_content:
        print("❌ PDF内容提取失败")
        return

    print(f"📚 原始章节数: {len(pdf_content.sections)}")

    # 3. 分析原始章节长度分布
    print("\n📊 原始章节长度分布:")
    short_sections = 0
    important_short_sections = 0

    for section in pdf_content.sections:
        length = len(section.content.strip())
        is_short = length <= 200
        is_important = short_handler._is_important_section(section)

        if is_short:
            short_sections += 1
            if is_important:
                important_short_sections += 1

            print(f"   📏 {section.section_type}: '{section.title[:40]}...' "
                  f"({length} 字符) {'⭐重要' if is_important else '普通'}")

    print(f"\n📈 统计:")
    print(f"   总章节数: {len(pdf_content.sections)}")
    print(f"   短章节数: {short_sections}")
    print(f"   重要短章节: {important_short_sections}")

    # 4. 使用短章节处理器处理
    print(f"\n🔧 应用短章节处理策略:")
    processed_docs = short_handler.process_short_sections(pdf_content.sections, min_length=200)

    # 5. 分析处理结果
    print(f"\n📊 处理后结果分析:")
    print(f"   处理前: {len(pdf_content.sections)} 章节")
    print(f"   处理后: {len(processed_docs)} 文档块")

    # 统计短章节处理情况
    preserved_short = 0
    merged_sections = 0
    extended_sections = 0

    for doc in processed_docs:
        metadata = doc['metadata']
        if metadata.get('is_short_section', False):
            reason = metadata.get('short_section_reason', '')
            if 'merge' in reason:
                merged_sections += 1
            elif 'extend' in reason:
                extended_sections += 1
            else:
                preserved_short += 1

    print(f"\n🎯 短章节处理统计:")
    print(f"   独立保留: {preserved_short}")
    print(f"   合并处理: {merged_sections}")
    print(f"   扩展上下文: {extended_sections}")

    # 6. 展示重要短章节的处理结果
    print(f"\n📋 重要短章节处理详情:")
    for i, doc in enumerate(processed_docs):
        metadata = doc['metadata']
        if metadata.get('is_short_section', False):
            section_type = metadata.get('section_type', 'unknown')
            reason = metadata.get('short_section_reason', '')
            length = len(doc['content'])

            print(f"   {i+1}. {section_type} -> {reason} ({length} 字符)")

            # 显示内容预览
            content_preview = doc['content'][:150].replace('\n', ' ')
            print(f"      预览: {content_preview}...")

    print(f"\n✅ 短章节处理测试完成")

    return {
        'original_sections': len(pdf_content.sections),
        'processed_docs': len(processed_docs),
        'short_sections': short_sections,
        'important_short_sections': important_short_sections,
        'preserved_short': preserved_short,
        'merged_sections': merged_sections,
        'extended_sections': extended_sections
    }

def compare_old_vs_new_processing():
    """对比旧版本和新版本的处理效果"""

    print(f"\n🔄 对比新旧处理方式")
    print("=" * 60)

    pdf_processor = AcademicPDFProcessor()

    import glob
    pdf_files = glob.glob("data/raw_papers/*.pdf")[:3]  # 测试前3个文件

    for pdf_file in pdf_files:
        filename = pdf_file.split('/')[-1]
        print(f"\n📄 测试: {filename}")

        pdf_content = pdf_processor.extract_pdf_content(pdf_file)
        if not pdf_content:
            continue

        # 旧版本逻辑：简单过滤 ≤200 字符
        old_way_sections = []
        for section in pdf_content.sections:
            if len(section.content.strip()) > 200:
                old_way_sections.append(section)

        # 新版本逻辑：智能处理
        short_handler = ShortSectionHandler()
        new_way_docs = short_handler.process_short_sections(pdf_content.sections, min_length=200)

        print(f"   旧方式: {len(pdf_content.sections)} → {len(old_way_sections)} 章节 "
              f"(丢失 {len(pdf_content.sections) - len(old_way_sections)} 个)")
        print(f"   新方式: {len(pdf_content.sections)} → {len(new_way_docs)} 文档块 "
              f"(处理 {len(pdf_content.sections) - len([d for d in new_way_docs if not d['metadata'].get('is_short_section', False)])} 个短章节)")

        # 检查是否有重要信息被旧方式丢失
        lost_important = 0
        for section in pdf_content.sections:
            if (len(section.content.strip()) <= 200 and
                short_handler._is_important_section(section)):
                lost_important += 1
                print(f"      ⚠️ 旧方式会丢失: {section.section_type} '{section.title[:30]}...'")

        if lost_important == 0:
            print(f"      ✅ 本文档无重要短章节丢失风险")

    print(f"\n✅ 对比测试完成")

if __name__ == "__main__":
    # 运行测试
    result = test_short_section_detection()

    if result:
        compare_old_vs_new_processing()

        print(f"\n📊 测试总结:")
        print(f"   原始章节: {result['original_sections']}")
        print(f"   处理后文档: {result['processed_docs']}")
        print(f"   短章节检测: {result['short_sections']} (其中重要: {result['important_short_sections']})")
        print(f"   处理策略: 保留 {result['preserved_short']}, 合并 {result['merged_sections']}, 扩展 {result['extended_sections']}")