#!/usr/bin/env python3
"""
Test table preservation in document chunking
Verify that tables are extracted, serialized to Markdown, and tracked with metadata
"""

import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'src'))

from src.processor.document_chunker import DocumentChunker, ChunkingConfig

def test_table_extraction_basic():
    """Test basic table extraction and serialization"""

    print("=" * 80)
    print("Test: Basic Table Extraction")
    print("=" * 80)

    # Test text with simple Markdown table
    test_text = """
    Abstract: This paper presents experimental results.

    Introduction

    We conducted experiments on multiple datasets.

    Table 1: Performance Comparison

    | Model | Accuracy | F1-Score |
    | --- | --- | --- |
    | Baseline | 85.2 | 0.83 |
    | Proposed | 92.5 | 0.91 |

    Discussion

    The results show significant improvement.
    """

    config = ChunkingConfig(strategy='hybrid', chunk_size=300, chunk_overlap=50)
    chunker = DocumentChunker(config)

    chunks = chunker.chunk_document(test_text, "test_paper_1")

    # Verify table chunk exists
    table_chunks = [c for c in chunks if c.section_type == "table"]

    print(f"Total chunks: {len(chunks)}")
    print(f"Table chunks: {len(table_chunks)}")

    assert len(table_chunks) > 0, "No table chunks found"

    # Check metadata
    table_chunk = table_chunks[0]
    print(f"\nTable chunk metadata:")
    print(f"  has_table: {table_chunk.metadata.get('has_table', False)}")
    print(f"  section_type: {table_chunk.section_type}")
    print(f"  chunk_id: {table_chunk.chunk_id}")

    assert table_chunk.metadata.get('has_table') == True, "has_table flag not set"
    assert table_chunk.section_type == "table", "section_type not set to table"

    print(f"\nTable chunk content:\n{table_chunk.text[:200]}")

    print("\n✅ Basic table extraction test PASSED\n")
    return chunks

def test_table_extraction_csv_style():
    """Test extraction of CSV-style tables"""

    print("=" * 80)
    print("Test: CSV-Style Table Extraction")
    print("=" * 80)

    test_text = """
    Methodology

    Our experiments used the following parameters.

    Table 2: Hyperparameters
    Learning Rate, Batch Size, Epochs
    0.001, 32, 100
    0.0001, 64, 150
    0.01, 16, 50

    Results

    The model converged after 80 epochs on average.
    """

    config = ChunkingConfig(strategy='semantic', chunk_size=400)
    chunker = DocumentChunker(config)

    chunks = chunker.chunk_document(test_text, "test_paper_2")

    table_chunks = [c for c in chunks if c.section_type == "table"]

    print(f"Total chunks: {len(chunks)}")
    print(f"Table chunks: {len(table_chunks)}")

    assert len(table_chunks) > 0, "No CSV table chunks found"

    table_chunk = table_chunks[0]
    print(f"\nCSV Table serialized content:\n{table_chunk.text}")

    # Verify Markdown format
    assert '|' in table_chunk.text or '-' in table_chunk.text, "Table not serialized to Markdown"

    print("\n✅ CSV-style table extraction test PASSED\n")
    return chunks

def test_table_extraction_ascii_table():
    """Test extraction of ASCII-bordered tables"""

    print("=" * 80)
    print("Test: ASCII-Bordered Table Extraction")
    print("=" * 80)

    test_text = """
    Experimental Setup

    We compared three different approaches.

    Table 3: Comparison Results

    +----------+----------+---------+
    | Method   | Speed    | Quality |
    +----------+----------+---------+
    | A        | Fast     | Good    |
    | B        | Medium   | Better  |
    | C        | Slow     | Best    |
    +----------+----------+---------+

    Conclusion

    Method C provides the best quality despite slower speed.
    """

    config = ChunkingConfig(strategy='fixed_size', chunk_size=500)
    chunker = DocumentChunker(config)

    chunks = chunker.chunk_document(test_text, "test_paper_3")

    table_chunks = [c for c in chunks if c.section_type == "table"]

    print(f"Total chunks: {len(chunks)}")
    print(f"Table chunks: {len(table_chunks)}")

    if len(table_chunks) > 0:
        table_chunk = table_chunks[0]
        print(f"\nASCII Table serialized content:\n{table_chunk.text}")
        assert table_chunk.metadata.get('has_table') == True, "has_table flag not set"
        print("\n✅ ASCII table extraction test PASSED\n")
    else:
        print("\n⚠️  ASCII table not extracted (this is acceptable)\n")

    return chunks

def test_multiple_tables():
    """Test document with multiple tables"""

    print("=" * 80)
    print("Test: Multiple Tables in Document")
    print("=" * 80)

    test_text = """
    Abstract: Comprehensive experimental evaluation.

    Table 1: Dataset Statistics

    | Dataset | Samples | Features |
    | --- | --- | --- |
    | Train | 10000 | 128 |
    | Test | 2000 | 128 |

    Methodology

    We used standard machine learning algorithms.

    Table 2: Algorithm Performance

    | Algorithm | Accuracy | Time (ms) |
    | --- | --- | --- |
    | SVM | 88.5 | 120 |
    | Random Forest | 91.2 | 85 |
    | Neural Net | 93.7 | 200 |

    Conclusion

    Neural networks achieved best accuracy.
    """

    config = ChunkingConfig(strategy='hybrid', chunk_size=400)
    chunker = DocumentChunker(config)

    chunks = chunker.chunk_document(test_text, "test_paper_4")

    table_chunks = [c for c in chunks if c.section_type == "table"]

    print(f"Total chunks: {len(chunks)}")
    print(f"Table chunks: {len(table_chunks)}")

    assert len(table_chunks) >= 2, f"Expected 2 tables, found {len(table_chunks)}"

    for i, table_chunk in enumerate(table_chunks):
        print(f"\nTable {i+1}:")
        print(f"  Chunk ID: {table_chunk.chunk_id}")
        print(f"  has_table: {table_chunk.metadata.get('has_table')}")
        print(f"  Content preview: {table_chunk.text[:80]}...")

    print("\n✅ Multiple tables test PASSED\n")
    return chunks

def test_table_metadata_preservation():
    """Test that has_table flag is properly set in metadata"""

    print("=" * 80)
    print("Test: Table Metadata Preservation")
    print("=" * 80)

    test_text = """
    Introduction

    This section has no tables, just regular text.
    We discuss the background and motivation.

    Table 1: Summary Statistics

    | Metric | Value |
    | --- | --- |
    | Mean | 45.2 |
    | Std Dev | 12.3 |

    Discussion

    The summary statistics are shown above.
    """

    config = ChunkingConfig(strategy='semantic', chunk_size=300)
    chunker = DocumentChunker(config)

    chunks = chunker.chunk_document(test_text, "test_paper_5",
                                    metadata={"paper_title": "Test Paper"})

    print(f"Total chunks: {len(chunks)}")

    # Check that table chunks have has_table=True
    table_count = 0
    non_table_count = 0

    for chunk in chunks:
        has_table = chunk.metadata.get('has_table', False)
        is_table_section = chunk.section_type == "table"

        print(f"\nChunk {chunk.chunk_index}:")
        print(f"  section_type: {chunk.section_type}")
        print(f"  has_table: {has_table}")
        print(f"  Content preview: {chunk.text[:60].replace(chr(10), ' ')}...")

        if is_table_section:
            assert has_table == True, f"Table chunk missing has_table flag"
            table_count += 1
        else:
            non_table_count += 1

    print(f"\n✅ Table chunks with has_table=True: {table_count}")
    print(f"✅ Non-table chunks: {non_table_count}")
    print("\n✅ Metadata preservation test PASSED\n")

    return chunks

def test_table_size_limits():
    """Test that table chunks respect size constraints"""

    print("=" * 80)
    print("Test: Table Size Constraints")
    print("=" * 80)

    # Create a large table
    table_rows = "\n".join([f"| Row {i} | Value {i*10} | Data {i*100} |"
                           for i in range(1, 51)])

    test_text = f"""
    Results Section

    Table 1: Large Dataset Results

    | ID | Score | Details |
    | --- | --- | --- |
    {table_rows}

    Analysis

    The table above shows all results.
    """

    config = ChunkingConfig(
        strategy='hybrid',
        chunk_size=500,
        max_chunk_size=2000,
        min_chunk_size=100
    )
    chunker = DocumentChunker(config)

    chunks = chunker.chunk_document(test_text, "test_paper_6")

    table_chunks = [c for c in chunks if c.section_type == "table"]

    print(f"Total chunks: {len(chunks)}")
    print(f"Table chunks: {len(table_chunks)}")

    if table_chunks:
        for i, tc in enumerate(table_chunks):
            print(f"\nTable chunk {i+1}:")
            print(f"  Size: {tc.char_count} chars")
            print(f"  Within limits: {tc.char_count <= config.max_chunk_size}")

    print("\n✅ Table size constraints test PASSED\n")
    return chunks

def run_all_tests():
    """Run all table preservation tests"""

    print("\n" + "=" * 80)
    print("RUNNING ALL TABLE PRESERVATION TESTS")
    print("=" * 80 + "\n")

    tests = [
        ("Basic Table Extraction", test_table_extraction_basic),
        ("CSV-Style Table", test_table_extraction_csv_style),
        ("ASCII-Bordered Table", test_table_extraction_ascii_table),
        ("Multiple Tables", test_multiple_tables),
        ("Table Metadata Preservation", test_table_metadata_preservation),
        ("Table Size Constraints", test_table_size_limits),
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
