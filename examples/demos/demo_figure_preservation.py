#!/usr/bin/env python3
"""
Demo: Figure Caption Preservation and Image Extraction
Shows how figure captions are extracted and optionally enhanced with OCR/image captioning
"""

import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.append(str(Path(__file__).parent))

from src.processor.document_chunker import DocumentChunker, ChunkingConfig
from src.processor.image_caption_extractor import (
    StubImageCaptionExtractor,
    ImageCaptionExtractor,
    create_extractor
)

def demo():
    print("\n" + "="*80)
    print("FIGURE CAPTION PRESERVATION & IMAGE EXTRACTION DEMO")
    print("="*80 + "\n")

    # Sample academic paper with figures
    paper_text = """
Title: Deep Neural Networks for Image Classification

Abstract: We present a comprehensive study of deep neural network architectures
for image classification tasks. Our experiments span multiple datasets and
demonstrate significant performance improvements.

1. Introduction

Deep learning has revolutionized computer vision tasks. We focus on three
key architectural innovations that improve classification accuracy.

2. Methodology

Figure 1: Proposed network architecture.
The architecture consists of five convolutional layers followed by three
fully-connected layers. Each convolutional layer uses batch normalization
and ReLU activation.

Our implementation follows standard practices with data augmentation including
random cropping, horizontal flipping, and color jittering.

3. Experimental Results

We evaluated our method on three benchmark datasets.

Figure 2: Training curves showing loss and accuracy over 200 epochs.
The solid lines represent training metrics while dashed lines show validation
performance. Convergence occurs around epoch 150.

Table 1: Performance Comparison

| Method | CIFAR-10 | ImageNet | Places365 |
| --- | --- | --- | --- |
| ResNet-50 | 94.2 | 76.1 | 54.8 |
| Our Method | 95.7 | 78.3 | 56.9 |

Figure 3: Confusion matrix for CIFAR-10 classification.

As shown in Figure 3, our model achieves high accuracy across all classes
with minimal confusion between similar categories.

4. Conclusion

The proposed architecture demonstrates consistent improvements over baseline
methods across multiple benchmarks.
"""

    print("📄 Processing academic paper with figures and tables...")
    print(f"   Text length: {len(paper_text)} characters\n")

    # Configure chunker
    config = ChunkingConfig(
        strategy='hybrid',
        chunk_size=500,
        chunk_overlap=80,
        min_chunk_size=100
    )

    chunker = DocumentChunker(config)

    # Demo 1: Basic figure extraction (no image processing)
    print("="*80)
    print("DEMO 1: Basic Figure Caption Extraction")
    print("="*80 + "\n")

    chunks = chunker.chunk_document(paper_text, "demo_paper_001")

    figure_chunks = [c for c in chunks if c.section_type == "figure" or
                     c.metadata.get('has_figure')]
    table_chunks = [c for c in chunks if c.section_type == "table"]

    print(f"📊 Extraction Results:")
    print(f"   - Total chunks: {len(chunks)}")
    print(f"   - Figure chunks: {len(figure_chunks)}")
    print(f"   - Table chunks: {len(table_chunks)}\n")

    for i, fc in enumerate(figure_chunks, 1):
        print(f"Figure {i}:")
        print(f"  ID: {fc.chunk_id}")
        print(f"  Figure ID: {fc.metadata.get('figure_id')}")
        print(f"  Section Type: {fc.section_type}")
        print(f"  Caption: {fc.metadata.get('figure_caption', fc.text[:80])}...")
        print(f"  Merged: {fc.metadata.get('merged_from_small_caption', False)}")
        print()

    # Demo 2: With mock image extraction
    print("="*80)
    print("DEMO 2: Figure Extraction with Mock Image Captioning")
    print("="*80 + "\n")

    # Create mock image extractor
    mock_extractor = Mock(spec=ImageCaptionExtractor)

    def mock_extract_all(image_path):
        # Simulate different outputs for different figures
        if 'fig_1' in image_path:
            return {
                'ocr_text': 'Conv1->BN->ReLU Conv2->BN->ReLU ... FC1->FC2->Softmax',
                'image_caption': 'A diagram showing a neural network architecture with five convolutional layers',
                'success': True,
                'error': None
            }
        elif 'fig_2' in image_path:
            return {
                'ocr_text': 'Epoch 0-200, Loss 2.5->0.3, Accuracy 0.1->0.96',
                'image_caption': 'Training curves showing decreasing loss and increasing accuracy over epochs',
                'success': True,
                'error': None
            }
        elif 'fig_3' in image_path:
            return {
                'ocr_text': '',  # Confusion matrix might not have text
                'image_caption': 'A 10x10 confusion matrix with high values on the diagonal',
                'success': True,
                'error': None
            }
        return {
            'ocr_text': '',
            'image_caption': '',
            'success': False,
            'error': 'Image not found'
        }

    mock_extractor.extract_all = mock_extract_all

    # Provide image paths for figures
    metadata = {
        'title': 'Deep Neural Networks for Image Classification',
        'figure_image_paths': {
            'fig_1': '/path/to/figures/architecture.png',
            'fig_2': '/path/to/figures/training_curves.png',
            'fig_3': '/path/to/figures/confusion_matrix.png'
        }
    }

    chunks_with_images = chunker.chunk_document(
        paper_text,
        "demo_paper_002",
        metadata,
        mock_extractor
    )

    enhanced_figure_chunks = [c for c in chunks_with_images
                             if c.metadata.get('has_figure') and
                             (c.metadata.get('ocr_text') or c.metadata.get('image_description'))]

    print(f"📊 Enhanced Extraction Results:")
    print(f"   - Total chunks: {len(chunks_with_images)}")
    print(f"   - Figures with image descriptions: {len(enhanced_figure_chunks)}\n")

    for i, fc in enumerate(enhanced_figure_chunks, 1):
        print(f"Enhanced Figure {i}:")
        print(f"  Figure ID: {fc.metadata.get('figure_id')}")
        print(f"  Original Caption: {fc.metadata.get('figure_caption', 'N/A')[:70]}...")

        if fc.metadata.get('ocr_text'):
            print(f"  OCR Text: {fc.metadata['ocr_text'][:70]}...")

        if fc.metadata.get('image_description'):
            print(f"  AI Caption: {fc.metadata['image_description'][:70]}...")

        print(f"  Chunk size: {fc.char_count} chars, {fc.word_count} words")
        print()

    # Demo 3: Comparison of chunk content
    print("="*80)
    print("DEMO 3: Content Comparison (With vs Without Image Extraction)")
    print("="*80 + "\n")

    # Find matching figure chunks
    for basic_chunk in figure_chunks:
        fig_id = basic_chunk.metadata.get('figure_id')
        if fig_id:
            # Find enhanced version
            enhanced = next((c for c in enhanced_figure_chunks
                           if c.metadata.get('figure_id') == fig_id), None)

            if enhanced:
                print(f"Figure {fig_id}:")
                print(f"\n  Without image extraction:")
                print(f"    Size: {basic_chunk.char_count} chars")
                print(f"    Content: {basic_chunk.text[:100]}...")

                print(f"\n  With image extraction:")
                print(f"    Size: {enhanced.char_count} chars")
                print(f"    Content: {enhanced.text[:100]}...")

                improvement = enhanced.char_count - basic_chunk.char_count
                print(f"\n  Enhancement: +{improvement} chars " +
                      f"({improvement/basic_chunk.char_count*100:.1f}% increase)")
                print()

    # Demo 4: Stub extractor (graceful degradation)
    print("="*80)
    print("DEMO 4: Graceful Degradation (Missing Dependencies)")
    print("="*80 + "\n")

    stub_extractor = StubImageCaptionExtractor()

    print(f"Stub Extractor Configuration:")
    print(f"  OCR available: {stub_extractor.ocr_available}")
    print(f"  OCR backend: {stub_extractor.ocr_backend}")
    print(f"  Captioning available: {stub_extractor.caption_available}")
    print(f"  Caption backend: {stub_extractor.caption_backend}\n")

    chunks_stub = chunker.chunk_document(
        paper_text,
        "demo_paper_003",
        metadata,
        stub_extractor
    )

    print(f"✅ Ingestion succeeded even without OCR/captioning dependencies")
    print(f"   Total chunks: {len(chunks_stub)}")
    print(f"   Figure chunks preserved: {len([c for c in chunks_stub if c.metadata.get('has_figure')])}\n")

    # Summary
    print("="*80)
    print("KEY FEATURES DEMONSTRATED")
    print("="*80)
    print("""
✅ Figure captions detected and extracted from document
✅ Small captions (<min_chunk_size) merged with narrative text
✅ Large captions become dedicated chunks with section_type='figure'
✅ Figure metadata includes: has_figure, figure_id, figure_caption
✅ OCR text extracted from figure images (when available)
✅ AI-generated captions added to figure chunks
✅ Image descriptions appended to chunk text for embedding
✅ Graceful degradation when OCR/captioning unavailable
✅ Figures and tables handled separately
✅ Original metadata preserved through processing
""")

    print("="*80)
    print("DEMO COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    demo()
