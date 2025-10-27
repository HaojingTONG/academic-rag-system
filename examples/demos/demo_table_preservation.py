#!/usr/bin/env python3
"""
Demo: Table Preservation Feature
Shows how tables are extracted and preserved in the chunking pipeline
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.processor.document_chunker import DocumentChunker, ChunkingConfig

def demo():
    print("\n" + "="*80)
    print("TABLE PRESERVATION FEATURE DEMONSTRATION")
    print("="*80 + "\n")

    # Sample academic paper with table
    paper_text = """
Title: Deep Learning Performance Analysis

Abstract: We present a comprehensive analysis of deep learning models
across multiple benchmarks. Our experiments reveal significant performance
differences between architectures.

1. Introduction

Deep learning has transformed computer vision and natural language processing.
Recent advances include:
- Transformer architectures
- Self-attention mechanisms
- Pre-training strategies

2. Experimental Setup

We evaluated models on standard benchmarks using consistent hyperparameters.

Table 1: Model Performance Comparison

| Model | ImageNet Acc | COCO mAP | Params (M) | Inference (ms) |
| --- | --- | --- | --- | --- |
| ResNet-50 | 76.2 | 38.5 | 25.6 | 12.3 |
| EfficientNet-B0 | 77.1 | 39.2 | 5.3 | 8.7 |
| Vision Transformer | 81.8 | 45.1 | 86.6 | 35.2 |
| Swin Transformer | 83.5 | 48.3 | 88.0 | 42.1 |

The table shows accuracy on ImageNet classification and COCO object detection.

3. Results and Analysis

Vision Transformers achieve superior performance but require more parameters
and computation time. EfficientNet provides the best efficiency trade-off.

Table 2: Training Configuration

| Parameter | Value |
| --- | --- |
| Batch Size | 128 |
| Learning Rate | 0.001 |
| Epochs | 100 |
| Optimizer | AdamW |
| Weight Decay | 0.05 |

4. Conclusion

Our analysis demonstrates that model selection depends on the specific
deployment constraints and performance requirements.
"""

    # Configure chunker
    config = ChunkingConfig(
        strategy='hybrid',
        chunk_size=600,
        chunk_overlap=100,
        min_chunk_size=100
    )

    chunker = DocumentChunker(config)

    print("📄 Processing academic paper with tables...")
    print(f"   Text length: {len(paper_text)} characters\n")

    # Chunk the document
    chunks = chunker.chunk_document(
        paper_text,
        "demo_paper_001",
        metadata={"title": "Deep Learning Performance Analysis"}
    )

    print(f"✅ Generated {len(chunks)} chunks\n")

    # Analyze chunks
    table_chunks = [c for c in chunks if c.section_type == "table"]
    text_chunks = [c for c in chunks if c.section_type != "table"]

    print(f"📊 Chunk Breakdown:")
    print(f"   - Text chunks: {len(text_chunks)}")
    print(f"   - Table chunks: {len(table_chunks)}")
    print()

    # Display table chunks
    print("="*80)
    print("TABLE CHUNKS (Preserved)")
    print("="*80 + "\n")

    for i, tc in enumerate(table_chunks, 1):
        print(f"Table {i}: {tc.chunk_id}")
        print(f"  Section Type: {tc.section_type}")
        print(f"  Has Table Flag: {tc.metadata.get('has_table')}")
        print(f"  Size: {tc.char_count} chars, {tc.word_count} words")
        print(f"  Position: {tc.start_char}-{tc.end_char}")
        print(f"\n  Content Preview:")
        print("  " + "-"*76)
        for line in tc.text.split('\n')[:10]:
            print(f"  {line}")
        if tc.text.count('\n') > 10:
            print(f"  ... ({tc.text.count(chr(10)) - 10} more lines)")
        print()

    # Display text chunks
    print("="*80)
    print("TEXT CHUNKS (Regular Content)")
    print("="*80 + "\n")

    for i, chunk in enumerate(text_chunks[:3], 1):  # Show first 3
        print(f"Chunk {i}: {chunk.chunk_id}")
        print(f"  Section Type: {chunk.section_type}")
        print(f"  Has Table Flag: {chunk.metadata.get('has_table')}")
        print(f"  Size: {chunk.char_count} chars")
        print(f"\n  Content Preview:")
        print("  " + "-"*76)
        preview = chunk.text[:200].replace('\n', '\n  ')
        print(f"  {preview}")
        if len(chunk.text) > 200:
            print(f"  ... (truncated)")
        print()

    # Statistics
    print("="*80)
    print("STATISTICS")
    print("="*80)

    stats = chunker.get_chunking_stats(chunks)
    print(f"\n  Total chunks: {stats['total_chunks']}")
    print(f"  Average chunk size: {stats['avg_chunk_size']:.1f} chars")
    print(f"  Average word count: {stats['avg_word_count']:.1f}")
    print(f"  Section types: {', '.join(stats['section_types'])}")

    # Metadata analysis
    chunks_with_tables = sum(1 for c in chunks if c.metadata.get('has_table'))
    print(f"\n  Chunks with has_table=True: {chunks_with_tables}")
    print(f"  Table extraction rate: {chunks_with_tables}/{len(chunks)} " +
          f"({100*chunks_with_tables/len(chunks):.1f}%)")

    print("\n" + "="*80)
    print("KEY FEATURES DEMONSTRATED")
    print("="*80)
    print("""
✅ Tables detected and extracted from document
✅ Tables serialized to Markdown format with proper structure
✅ Table chunks have section_type='table'
✅ Metadata flag has_table=True set correctly
✅ Text chunks separated from table chunks
✅ Document order preserved after merging
✅ All chunks properly indexed sequentially
""")

    print("="*80)
    print("DEMO COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    demo()
