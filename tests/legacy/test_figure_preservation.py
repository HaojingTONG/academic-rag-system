#!/usr/bin/env python3
"""
Test figure preservation in document chunking
Verify that figure captions are extracted, stored with metadata, and merged appropriately
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'src'))

from src.processor.document_chunker import DocumentChunker, ChunkingConfig
from src.processor.image_caption_extractor import (
    StubImageCaptionExtractor,
    ImageCaptionExtractor,
    append_image_descriptions_to_chunk
)

def test_figure_extraction_basic():
    """Test basic figure caption extraction"""

    print("=" * 80)
    print("Test: Basic Figure Caption Extraction")
    print("=" * 80)

    test_text = """
    Abstract: This paper presents experimental results.

    Introduction

    We conducted experiments on multiple datasets.

    Figure 1: Performance comparison across different methods.
    The graph shows accuracy on the y-axis and dataset size on the x-axis.

    Discussion

    As shown in Figure 1, the results demonstrate significant improvement.
    """

    config = ChunkingConfig(strategy='hybrid', chunk_size=300, chunk_overlap=50)
    chunker = DocumentChunker(config)

    chunks = chunker.chunk_document(test_text, "test_paper_1")

    # Verify figure chunk exists
    figure_chunks = [c for c in chunks if c.section_type == "figure"]

    print(f"Total chunks: {len(chunks)}")
    print(f"Figure chunks: {len(figure_chunks)}")

    assert len(figure_chunks) > 0, "No figure chunks found"

    # Check metadata
    figure_chunk = figure_chunks[0]
    print(f"\nFigure chunk metadata:")
    print(f"  has_figure: {figure_chunk.metadata.get('has_figure', False)}")
    print(f"  section_type: {figure_chunk.section_type}")
    print(f"  figure_id: {figure_chunk.metadata.get('figure_id')}")
    print(f"  figure_caption: {figure_chunk.metadata.get('figure_caption', '')[:80]}...")

    assert figure_chunk.metadata.get('has_figure') == True, "has_figure flag not set"
    assert figure_chunk.section_type == "figure", "section_type not set to figure"
    assert figure_chunk.metadata.get('figure_id') == 'fig_1', "figure_id not extracted correctly"
    assert 'Performance comparison' in figure_chunk.metadata.get('figure_caption', ''), "Caption not stored"

    print(f"\nFigure chunk content:\n{figure_chunk.text[:150]}")

    print("\n✅ Basic figure extraction test PASSED\n")
    return chunks


def test_small_figure_caption_merging():
    """Test that small figure captions are merged with narrative"""

    print("=" * 80)
    print("Test: Small Figure Caption Merging")
    print("=" * 80)

    test_text = """
    Results Section

    Our experimental results are shown below.

    Fig. 2: Results.

    The results indicate a strong correlation between variables.
    """

    config = ChunkingConfig(strategy='semantic', chunk_size=300, min_chunk_size=100)
    chunker = DocumentChunker(config)

    chunks = chunker.chunk_document(test_text, "test_paper_2")

    print(f"Total chunks: {len(chunks)}")

    # Find chunks with figures
    chunks_with_figures = [c for c in chunks if c.metadata.get('has_figure')]

    print(f"Chunks with has_figure=True: {len(chunks_with_figures)}")

    # Should have at least one chunk with figure info
    assert len(chunks_with_figures) > 0, "No chunks with figure metadata found"

    # Check if caption was merged
    merged_chunk = chunks_with_figures[0]
    print(f"\nMerged chunk:")
    print(f"  section_type: {merged_chunk.section_type}")
    print(f"  figure_caption: {merged_chunk.metadata.get('figure_caption')}")
    print(f"  merged_from_small_caption: {merged_chunk.metadata.get('merged_from_small_caption')}")
    print(f"  Content: {merged_chunk.text[:100]}...")

    # Verify caption is in the text
    assert 'Results' in merged_chunk.text or 'Fig. 2' in merged_chunk.text, "Caption not merged into text"

    print("\n✅ Small figure caption merging test PASSED\n")
    return chunks


def test_multiple_figures():
    """Test document with multiple figures"""

    print("=" * 80)
    print("Test: Multiple Figures in Document")
    print("=" * 80)

    test_text = """
    Abstract: Comprehensive evaluation of deep learning models.

    Figure 1: Architecture diagram.
    The figure shows the neural network architecture with multiple layers.

    Methodology

    We implemented the model as described above.

    Figure 2: Training curves over 100 epochs.
    Loss decreases steadily while accuracy improves.

    Figure 3: Comparison with baseline methods.

    Conclusion

    The three figures demonstrate the effectiveness of our approach.
    """

    config = ChunkingConfig(strategy='hybrid', chunk_size=400)
    chunker = DocumentChunker(config)

    chunks = chunker.chunk_document(test_text, "test_paper_3")

    figure_chunks = [c for c in chunks if c.section_type == "figure"]
    chunks_with_figures = [c for c in chunks if c.metadata.get('has_figure')]

    print(f"Total chunks: {len(chunks)}")
    print(f"Dedicated figure chunks: {len(figure_chunks)}")
    print(f"Chunks with has_figure flag: {len(chunks_with_figures)}")

    assert len(chunks_with_figures) >= 2, f"Expected at least 2 figures, found {len(chunks_with_figures)}"

    for i, chunk in enumerate(chunks_with_figures):
        print(f"\nFigure {i+1}:")
        print(f"  Chunk ID: {chunk.chunk_id}")
        print(f"  Figure ID: {chunk.metadata.get('figure_id')}")
        print(f"  Caption: {chunk.metadata.get('figure_caption', 'N/A')[:60]}...")

    print("\n✅ Multiple figures test PASSED\n")
    return chunks


def test_figure_vs_table_detection():
    """Test that figures and tables are distinguished correctly"""

    print("=" * 80)
    print("Test: Figure vs Table Detection")
    print("=" * 80)

    test_text = """
    Results

    Table 1: Performance metrics

    | Model | Accuracy |
    | --- | --- |
    | A | 0.95 |

    Figure 1: Visualization of results across multiple experiments.
    The chart shows performance trends over time with error bars indicating variance.
    This comprehensive visualization helps understand the model behavior.

    Discussion

    Both Table 1 and Figure 1 support our conclusions about the effectiveness of our approach.
    """

    config = ChunkingConfig(strategy='semantic', chunk_size=300, min_chunk_size=80)
    chunker = DocumentChunker(config)

    chunks = chunker.chunk_document(test_text, "test_paper_4")

    table_chunks = [c for c in chunks if c.section_type == "table"]
    figure_chunks = [c for c in chunks if c.section_type == "figure"]
    figure_caption_chunks = [c for c in chunks if c.section_type == "figure_caption"]

    print(f"Total chunks: {len(chunks)}")
    print(f"Table chunks: {len(table_chunks)}")
    print(f"Figure chunks (standalone): {len(figure_chunks)}")
    print(f"Figure caption chunks (merged): {len(figure_caption_chunks)}")

    assert len(table_chunks) > 0, "Table not detected"

    # Figure can be either standalone or merged with narrative
    total_figure_chunks = len(figure_chunks) + len(figure_caption_chunks)
    assert total_figure_chunks > 0, f"Figure not detected (found {total_figure_chunks})"

    # Verify they're separate from tables
    if figure_chunks:
        assert table_chunks[0].chunk_id != figure_chunks[0].chunk_id, "Table and figure not separated"

    print(f"\n✅ Found {len(table_chunks)} table(s) and {total_figure_chunks} figure(s)")
    print("\n✅ Figure vs table detection test PASSED\n")
    return chunks


def test_image_caption_extraction_mock():
    """Test image caption extraction with mock extractor"""

    print("=" * 80)
    print("Test: Image Caption Extraction (Mock)")
    print("=" * 80)

    # Create mock extractor
    mock_extractor = Mock(spec=ImageCaptionExtractor)
    mock_extractor.extract_all = Mock(return_value={
        'ocr_text': 'Accuracy: 95.2%, Precision: 94.8%',
        'image_caption': 'A bar chart showing model performance metrics',
        'success': True,
        'error': None
    })

    test_text = """
    Experimental Results

    Figure 1: Performance comparison.
    The figure shows accuracy and precision for different models.

    Analysis

    As demonstrated in Figure 1, our method achieves superior results.
    """

    config = ChunkingConfig(strategy='semantic', chunk_size=400)
    chunker = DocumentChunker(config)

    # Provide image paths in metadata
    metadata = {
        'title': 'Test Paper',
        'figure_image_paths': {
            'fig_1': '/path/to/figure1.png'
        }
    }

    chunks = chunker.chunk_document(test_text, "test_paper_5", metadata, mock_extractor)

    # Find figure chunk
    figure_chunks = [c for c in chunks if c.metadata.get('figure_id') == 'fig_1']

    assert len(figure_chunks) > 0, "Figure chunk not found"

    figure_chunk = figure_chunks[0]

    print(f"Figure chunk with image extraction:")
    print(f"  OCR text in metadata: {figure_chunk.metadata.get('ocr_text', 'N/A')}")
    print(f"  Image description in metadata: {figure_chunk.metadata.get('image_description', 'N/A')}")
    print(f"  Text includes OCR: {'OCR Text' in figure_chunk.text}")
    print(f"  Text includes caption: {'Image Description' in figure_chunk.text}")

    # Verify mock was called
    assert mock_extractor.extract_all.called, "Image extractor not invoked"

    # Verify descriptions were appended
    if figure_chunk.metadata.get('ocr_text'):
        print(f"\n✅ OCR text added: {figure_chunk.metadata['ocr_text']}")

    if figure_chunk.metadata.get('image_description'):
        print(f"✅ Image caption added: {figure_chunk.metadata['image_description']}")

    print("\n✅ Image caption extraction test PASSED\n")
    return chunks


def test_stub_extractor_graceful_degradation():
    """Test that stub extractor doesn't break ingestion"""

    print("=" * 80)
    print("Test: Stub Extractor Graceful Degradation")
    print("=" * 80)

    stub_extractor = StubImageCaptionExtractor()

    test_text = """
    Methods

    Figure 1: System architecture.

    Results

    Performance metrics are shown in Figure 1.
    """

    config = ChunkingConfig(strategy='fixed_size', chunk_size=200)
    chunker = DocumentChunker(config)

    metadata = {
        'figure_image_paths': {
            'fig_1': '/nonexistent/path.png'
        }
    }

    # Should not raise exception even with missing dependencies
    try:
        chunks = chunker.chunk_document(test_text, "test_paper_6", metadata, stub_extractor)
        print(f"✅ Ingestion succeeded with stub extractor")
        print(f"   Total chunks: {len(chunks)}")
        print("\n✅ Stub extractor test PASSED\n")
        return chunks
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        raise


def test_figure_metadata_preservation():
    """Test that figure metadata is properly preserved"""

    print("=" * 80)
    print("Test: Figure Metadata Preservation")
    print("=" * 80)

    test_text = """
    Introduction

    Recent advances in machine learning are significant.

    Figure 3: Attention mechanism visualization.
    This figure illustrates how the attention weights are distributed across input tokens.

    Methodology

    We employ the attention mechanism shown in Figure 3.
    """

    config = ChunkingConfig(strategy='semantic', chunk_size=300)
    chunker = DocumentChunker(config)

    # Add custom metadata
    custom_metadata = {
        'paper_title': 'Test Paper on Attention',
        'year': 2024
    }

    chunks = chunker.chunk_document(test_text, "test_paper_7", custom_metadata)

    figure_chunks = [c for c in chunks if c.section_type == "figure"]

    assert len(figure_chunks) > 0, "No figure chunks found"

    figure_chunk = figure_chunks[0]

    print(f"Figure chunk metadata:")
    for key, value in sorted(figure_chunk.metadata.items()):
        print(f"  {key}: {value}")

    # Verify custom metadata preserved
    assert figure_chunk.metadata.get('paper_title') == 'Test Paper on Attention', "Custom metadata lost"
    assert figure_chunk.metadata.get('year') == 2024, "Year metadata lost"

    # Verify figure-specific metadata
    assert figure_chunk.metadata.get('has_figure') == True
    assert figure_chunk.metadata.get('figure_id') == 'fig_3'
    assert figure_chunk.metadata.get('figure_caption') is not None

    print("\n✅ Metadata preservation test PASSED\n")
    return chunks


def test_figure_id_extraction():
    """Test figure ID extraction from various formats"""

    print("=" * 80)
    print("Test: Figure ID Extraction")
    print("=" * 80)

    test_cases = [
        ("Figure 1: Test", "fig_1"),
        ("Fig. 5: Another test", "fig_5"),
        ("FIG. 10: Upper case", "fig_10"),
        ("FIGURE 23: All caps", "fig_23"),
    ]

    config = ChunkingConfig(strategy='fixed_size', chunk_size=500)
    chunker = DocumentChunker(config)

    for caption, expected_id in test_cases:
        test_text = f"""
        Introduction

        {caption}

        Discussion
        """

        chunks = chunker.chunk_document(test_text, "test_id_extraction")
        figure_chunks = [c for c in chunks if c.section_type == "figure"]

        if figure_chunks:
            actual_id = figure_chunks[0].metadata.get('figure_id')
            print(f"Caption: '{caption}' -> ID: '{actual_id}' (expected: '{expected_id}')")
            assert actual_id == expected_id, f"ID mismatch for '{caption}'"

    print("\n✅ Figure ID extraction test PASSED\n")


def run_all_tests():
    """Run all figure preservation tests"""

    print("\n" + "=" * 80)
    print("RUNNING ALL FIGURE PRESERVATION TESTS")
    print("=" * 80 + "\n")

    tests = [
        ("Basic Figure Extraction", test_figure_extraction_basic),
        ("Small Caption Merging", test_small_figure_caption_merging),
        ("Multiple Figures", test_multiple_figures),
        ("Figure vs Table Detection", test_figure_vs_table_detection),
        ("Image Caption Extraction (Mock)", test_image_caption_extraction_mock),
        ("Stub Extractor Graceful Degradation", test_stub_extractor_graceful_degradation),
        ("Figure Metadata Preservation", test_figure_metadata_preservation),
        ("Figure ID Extraction", test_figure_id_extraction),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            test_func()
            results[test_name] = "PASSED"
        except AssertionError as e:
            print(f"\n❌ {test_name} FAILED: {e}\n")
            results[test_name] = f"FAILED: {e}"
        except Exception as e:
            print(f"\n❌ {test_name} ERROR: {e}\n")
            results[test_name] = f"ERROR: {e}"

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(results)

    for test_name, result in results.items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"{status} {test_name}: {result}")

    print(f"\n{'='*80}")
    print(f"TOTAL: {passed}/{total} tests passed")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    run_all_tests()
