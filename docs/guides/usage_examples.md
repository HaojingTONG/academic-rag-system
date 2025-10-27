# Usage Examples: Figure and Table Preservation

## Quick Start

### Basic Document Chunking with Figure/Table Extraction

```python
from src.processor.document_chunker import DocumentChunker, ChunkingConfig

# Configure chunker
config = ChunkingConfig(
    strategy='hybrid',
    chunk_size=500,
    chunk_overlap=100,
    min_chunk_size=100
)

chunker = DocumentChunker(config)

# Sample text with figures and tables
text = """
Abstract: Novel approach to deep learning.

Figure 1: System architecture diagram.
Shows the complete pipeline with preprocessing, model, and postprocessing.

Table 1: Performance comparison

| Model | Accuracy | Speed |
| ResNet | 94.2 | 15ms |
| Our Model | 96.1 | 12ms |

Results show significant improvement.
"""

# Chunk the document
chunks = chunker.chunk_document(text, paper_id="paper_001")

# Access different chunk types
tables = [c for c in chunks if c.section_type == "table"]
figures = [c for c in chunks if c.section_type == "figure"]
text_chunks = [c for c in chunks if c.section_type not in ["table", "figure", "figure_caption"]]

print(f"Found {len(tables)} tables, {len(figures)} figures, {len(text_chunks)} text chunks")
```

## Advanced: Image Caption Extraction

### With Auto-Detection of Available Libraries

```python
from src.processor.document_chunker import DocumentChunker, ChunkingConfig
from src.processor.image_caption_extractor import create_extractor

# Create extractor (auto-detects PaddleOCR, BLIP, etc.)
extractor = create_extractor()

# Prepare metadata with figure image paths
metadata = {
    'title': 'My Research Paper',
    'authors': 'Smith et al.',
    'year': 2024,
    'figure_image_paths': {
        'fig_1': '/path/to/extracted/figure1.png',
        'fig_2': '/path/to/extracted/figure2.png'
    }
}

# Chunk with image extraction
chunks = chunker.chunk_document(
    text,
    paper_id="paper_002",
    metadata=metadata,
    image_extractor=extractor
)

# Check enhanced figure chunks
for chunk in chunks:
    if chunk.metadata.get('has_figure'):
        print(f"\nFigure {chunk.metadata['figure_id']}:")
        print(f"  Caption: {chunk.metadata['figure_caption']}")

        if chunk.metadata.get('ocr_text'):
            print(f"  OCR Text: {chunk.metadata['ocr_text']}")

        if chunk.metadata.get('image_description'):
            print(f"  AI Caption: {chunk.metadata['image_description']}")
```

### With Specific Backends

```python
from src.processor.image_caption_extractor import (
    create_extractor,
    PaddleOCRExtractor,
    BLIPCaptionExtractor,
    HybridImageCaptionExtractor
)

# Option 1: Specify backends
extractor = create_extractor(
    ocr_backend='paddleocr',      # or 'tesseract', 'none', 'auto'
    caption_backend='blip'        # or 'clip', 'none', 'auto'
)

# Option 2: Create specific extractors
ocr = PaddleOCRExtractor(use_gpu=True)
caption = BLIPCaptionExtractor(model_name="Salesforce/blip-image-captioning-base")
extractor = HybridImageCaptionExtractor(ocr, caption)

# Option 3: Create custom extractor
from src.processor.image_caption_extractor import ImageCaptionExtractor

class MyCustomExtractor(ImageCaptionExtractor):
    def extract_text(self, image_path: str) -> str:
        # Your OCR implementation
        return "extracted text"

    def generate_caption(self, image_path: str) -> str:
        # Your captioning implementation
        return "generated caption"

extractor = MyCustomExtractor()
```

## Vector Store Integration

### Search by Content Type

```python
from src.retriever.vector_store import VectorStore

vs = VectorStore()

# Search only in tables
table_results = vs.search_with_tables("performance metrics", top_k=5)

# Search only in figures
figure_results = vs.search_with_figures("architecture diagram", top_k=5)

# Search with custom filters
results = vs.search(
    "deep learning results",
    top_k=10,
    filter_metadata={
        "has_figure": True,
        "has_formulas": True
    }
)

# Get statistics
stats = vs.get_statistics()
print(f"Tables: {stats['content_features']['has_table']}")
print(f"Figures: {stats['content_features']['has_figure']}")
```

## Working with Chunks

### Accessing Figure Metadata

```python
for chunk in chunks:
    if chunk.section_type == "figure":
        # Standalone figure chunk
        print(f"Figure ID: {chunk.metadata['figure_id']}")
        print(f"Caption: {chunk.metadata['figure_caption']}")
        print(f"Standalone: True")

    elif chunk.section_type == "figure_caption":
        # Figure caption merged with narrative
        print(f"Figure ID: {chunk.metadata['figure_id']}")
        print(f"Caption: {chunk.metadata['figure_caption']}")
        print(f"Merged: {chunk.metadata.get('merged_from_small_caption', False)}")
```

### Filtering Chunks

```python
# Get all chunks with figures (standalone or merged)
figure_chunks = [c for c in chunks if c.metadata.get('has_figure')]

# Get only standalone figure chunks
standalone_figures = [c for c in chunks if c.section_type == "figure"]

# Get figure captions merged with text
merged_figures = [
    c for c in chunks
    if c.metadata.get('merged_from_small_caption')
]

# Get chunks with both figures and formulas
enriched_chunks = [
    c for c in chunks
    if c.metadata.get('has_figure') and c.metadata.get('has_formulas')
]
```

## Handling Edge Cases

### Graceful Degradation Without OCR/Captioning

```python
from src.processor.image_caption_extractor import StubImageCaptionExtractor

# Use stub extractor when dependencies unavailable
stub = StubImageCaptionExtractor()

# This won't fail even if PaddleOCR/BLIP not installed
chunks = chunker.chunk_document(
    text,
    paper_id="paper_003",
    metadata={'figure_image_paths': {'fig_1': '/path/to/img.png'}},
    image_extractor=stub
)

# Figure captions still extracted, just without OCR/AI captions
```

### Custom Metadata Preservation

```python
# Your custom metadata
custom_meta = {
    'conference': 'NeurIPS 2024',
    'topic': 'Computer Vision',
    'keywords': ['deep learning', 'CNN', 'attention']
}

chunks = chunker.chunk_document(text, "paper_004", custom_meta)

# Custom metadata preserved in all chunks
for chunk in chunks:
    assert chunk.metadata['conference'] == 'NeurIPS 2024'
    assert chunk.metadata['topic'] == 'Computer Vision'
```

### Mixed Content Documents

```python
text = """
Abstract: Comprehensive study.

Table 1: Baseline results
| Method | Score |
| A | 0.90 |

Figure 1: Improved architecture.
Our novel design includes attention mechanisms.

Figure 2: Small caption.

Table 2: Ablation study
| Component | Impact |
| Attention | +2.1% |

Discussion
The results demonstrate effectiveness.
"""

chunks = chunker.chunk_document(text, "mixed_doc")

# Analyze chunk distribution
for chunk in chunks:
    print(f"Chunk {chunk.chunk_index}:")
    print(f"  Type: {chunk.section_type}")
    print(f"  Has table: {chunk.metadata.get('has_table', False)}")
    print(f"  Has figure: {chunk.metadata.get('has_figure', False)}")
    print(f"  Size: {chunk.char_count} chars")
```

## PDF Processing Integration

### Example: Extending PDF Processor

```python
# In your PDF processing code
from src.processor.pdf_processor import AcademicPDFProcessor
from src.processor.document_chunker import DocumentChunker, ChunkingConfig
from src.processor.image_caption_extractor import create_extractor

# Extract PDF content
processor = AcademicPDFProcessor()
pdf_content = processor.extract_pdf_content("paper.pdf")

# Build metadata including figure image paths
# (Assuming PDF processor extracted figure images to temp directory)
metadata = {
    'title': pdf_content.title,
    'total_pages': pdf_content.total_pages,
    'figure_image_paths': {
        'fig_1': '/tmp/paper_figures/fig1.png',
        'fig_2': '/tmp/paper_figures/fig2.png',
        # ... more figures
    }
}

# Convert sections to text
full_text = ""
for section in pdf_content.sections:
    full_text += f"{section.title}\n\n{section.content}\n\n"

# Chunk with figure extraction
config = ChunkingConfig(strategy='hybrid', chunk_size=600)
chunker = DocumentChunker(config)
extractor = create_extractor()

chunks = chunker.chunk_document(
    full_text,
    paper_id="arxiv_2024_001",
    metadata=metadata,
    image_extractor=extractor
)
```

## Best Practices

### 1. Choose Appropriate Chunk Size

```python
# For short papers (< 10 pages)
config = ChunkingConfig(chunk_size=400, min_chunk_size=80)

# For long papers (> 20 pages)
config = ChunkingConfig(chunk_size=600, min_chunk_size=120)

# For very technical papers with many formulas
config = ChunkingConfig(chunk_size=500, min_chunk_size=100, preserve_sentences=True)
```

### 2. Handle Missing Image Files

```python
# Validate image paths before processing
import os

if 'figure_image_paths' in metadata:
    valid_paths = {
        fig_id: path
        for fig_id, path in metadata['figure_image_paths'].items()
        if os.path.exists(path)
    }
    metadata['figure_image_paths'] = valid_paths
```

### 3. Monitor Processing

```python
import logging

logging.basicConfig(level=logging.INFO)

# Will log warnings for missing dependencies, failed extractions, etc.
chunks = chunker.chunk_document(text, "paper_005", metadata, extractor)
```

### 4. Batch Processing

```python
papers = [
    {'id': 'p1', 'text': text1, 'metadata': meta1},
    {'id': 'p2', 'text': text2, 'metadata': meta2},
    # ...
]

all_chunks = []
extractor = create_extractor()  # Create once, reuse

for paper in papers:
    try:
        chunks = chunker.chunk_document(
            paper['text'],
            paper['id'],
            paper['metadata'],
            extractor
        )
        all_chunks.extend(chunks)
    except Exception as e:
        logging.error(f"Failed to process {paper['id']}: {e}")
        continue
```

## Testing Your Integration

```python
def test_my_integration():
    """Test figure/table extraction in your pipeline"""
    chunker = DocumentChunker(ChunkingConfig(chunk_size=300))

    test_text = """
    Figure 1: Test figure.
    Table 1: Test table
    | A | B |
    | 1 | 2 |
    """

    chunks = chunker.chunk_document(test_text, "test")

    # Verify extraction
    assert any(c.section_type == "figure" for c in chunks), "Figure not extracted"
    assert any(c.section_type == "table" for c in chunks), "Table not extracted"
    assert any(c.metadata.get('has_figure') for c in chunks), "has_figure not set"
    assert any(c.metadata.get('has_table') for c in chunks), "has_table not set"

    print("✅ Integration test passed")

test_my_integration()
```

## Troubleshooting

### Issue: Figures Not Detected

**Solution**: Check caption format matches patterns:
- ✅ "Figure 1:", "Fig. 2:", "FIG. 3:"
- ❌ "Fig 1" (missing period/colon), "Diagram 1"

### Issue: Small Figures Not Merged

**Cause**: Caption exceeds `min_chunk_size`

**Solution**: Adjust configuration:
```python
config = ChunkingConfig(min_chunk_size=50)  # Lower threshold
```

### Issue: OCR/Captioning Not Working

**Diagnosis**:
```python
from src.processor.image_caption_extractor import StubImageCaptionExtractor

stub = StubImageCaptionExtractor()
print(f"OCR available: {stub.ocr_available}")
print(f"Captioning available: {stub.caption_available}")
```

**Solution**: Install dependencies:
```bash
pip install paddleocr transformers pillow torch
```

### Issue: Image Descriptions Not Appended

**Cause**: Missing `figure_image_paths` in metadata

**Solution**:
```python
# Ensure metadata includes paths
metadata = {
    'figure_image_paths': {
        'fig_1': '/correct/path/to/image.png'
    }
}
```
