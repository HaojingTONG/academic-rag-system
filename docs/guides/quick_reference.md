# Quick Reference: Figure & Table Extraction

## At a Glance

### What's New
✅ **Figure captions** automatically detected and extracted
✅ **OCR text** from figure images (optional)
✅ **AI-generated captions** for figures (optional)
✅ **Smart merging** of small captions with narrative
✅ **Metadata tracking** with `has_figure` flag
✅ **Vector search** by figures using `search_with_figures()`

### Detected Patterns
- **Figures**: `Figure 1:`, `Fig. 2:`, `FIG. 3:`, `FIGURE 4:`
- **Tables**: `Table 1:`, `Tab. 2:`, `TABLE 3:`

## Basic Usage

```python
from src.processor.document_chunker import DocumentChunker, ChunkingConfig

chunker = DocumentChunker(ChunkingConfig(chunk_size=500))
chunks = chunker.chunk_document(text, paper_id="p001")

# Filter by type
tables = [c for c in chunks if c.section_type == "table"]
figures = [c for c in chunks if c.section_type == "figure"]
```

## With Image Extraction

```python
from src.processor.image_caption_extractor import create_extractor

extractor = create_extractor()  # Auto-detects available backends

metadata = {
    'figure_image_paths': {
        'fig_1': '/path/to/figure1.png',
        'fig_2': '/path/to/figure2.png'
    }
}

chunks = chunker.chunk_document(text, "p002", metadata, extractor)

# Access enhanced metadata
for chunk in chunks:
    if chunk.metadata.get('ocr_text'):
        print(chunk.metadata['ocr_text'])
```

## Search Operations

```python
from src.retriever.vector_store import VectorStore

vs = VectorStore()

# Search tables only
vs.search_with_tables("performance metrics", top_k=5)

# Search figures only
vs.search_with_figures("architecture diagram", top_k=5)

# Combined filter
vs.search("results", top_k=10, filter_metadata={
    "has_figure": True,
    "has_formulas": True
})
```

## Chunk Metadata Schema

```python
# Figure chunk
{
    'has_figure': True,
    'figure_id': 'fig_3',
    'figure_caption': 'Architecture diagram showing...',
    'section_type': 'figure',  # or 'figure_caption' if merged

    # Optional (when image extractor used):
    'ocr_text': 'Text extracted from image',
    'image_description': 'AI-generated caption',

    # Other standard metadata
    'paper_id': 'p001',
    'chunk_index': 5,
    'has_formulas': False,
    # ...
}
```

## Configuration Options

```python
config = ChunkingConfig(
    strategy='hybrid',           # 'fixed_size', 'semantic', 'hybrid'
    chunk_size=500,             # Target size in characters
    chunk_overlap=100,          # Overlap between chunks
    min_chunk_size=100,         # Minimum size (affects merging)
    max_chunk_size=1000,        # Maximum size
    preserve_sentences=True,    # Break at sentence boundaries
    preserve_paragraphs=True,   # Preserve paragraph structure
    section_aware=True          # Respect section boundaries
)
```

## Image Extractor Backends

```python
# Auto-detect (recommended)
extractor = create_extractor()

# Specific backends
extractor = create_extractor(
    ocr_backend='paddleocr',    # 'paddleocr', 'tesseract', 'auto', 'none'
    caption_backend='blip'       # 'blip', 'clip', 'auto', 'none'
)

# Stub (no dependencies)
from src.processor.image_caption_extractor import StubImageCaptionExtractor
extractor = StubImageCaptionExtractor()
```

## Common Patterns

### Filter chunks with figures
```python
figure_chunks = [c for c in chunks if c.metadata.get('has_figure')]
```

### Get figure ID
```python
fig_id = chunk.metadata.get('figure_id')  # e.g., 'fig_1', 'fig_3'
```

### Check if caption was merged
```python
is_merged = chunk.metadata.get('merged_from_small_caption', False)
```

### Access full caption
```python
caption = chunk.metadata.get('figure_caption', '')
```

### Get OCR/AI descriptions
```python
ocr = chunk.metadata.get('ocr_text', '')
ai_caption = chunk.metadata.get('image_description', '')
```

## Tests

```bash
# Run table tests
python test_table_preservation.py

# Run figure tests
python test_figure_preservation.py

# Run demo
python demo_figure_preservation.py
```

## Dependencies

### Required (Core)
- Python 3.7+
- PyMuPDF (fitz)
- Standard library

### Optional (Image Extraction)
```bash
# OCR
pip install paddleocr

# Image Captioning
pip install transformers pillow torch

# All at once
pip install paddleocr transformers pillow torch
```

## Key Files

| File | Purpose |
|------|---------|
| `src/processor/document_chunker.py` | Main chunking logic |
| `src/processor/image_caption_extractor.py` | OCR/captioning module |
| `src/retriever/vector_store.py` | Vector search with filters |
| `test_figure_preservation.py` | Figure extraction tests |
| `test_table_preservation.py` | Table extraction tests |
| `demo_figure_preservation.py` | Interactive demo |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Figures not detected | Check caption format: `Figure N:` or `Fig. N:` |
| Tables not detected | Check format: `Table N:` or pipe-delimited rows |
| Small captions not merged | Lower `min_chunk_size` in config |
| OCR not working | Install `paddleocr`: `pip install paddleocr` |
| Captioning not working | Install `transformers pillow torch` |
| Image not found error | Check paths in `figure_image_paths` metadata |

## Best Practices

✅ Use `create_extractor()` with auto-detection
✅ Validate image paths before processing
✅ Use appropriate `chunk_size` for your documents
✅ Enable logging to monitor processing
✅ Handle extraction failures gracefully
✅ Batch process with shared extractor instance

❌ Don't hardcode image paths in code
❌ Don't fail ingestion on OCR errors
❌ Don't set `min_chunk_size` too high (prevents merging)
❌ Don't reprocess images unnecessarily

## Performance Tips

- Create extractor once, reuse for multiple documents
- Use GPU for OCR/captioning if available
- Validate image paths before extraction
- Set appropriate batch sizes for vector store
- Cache image descriptions to avoid reprocessing

## Version Info

- **Feature**: Figure & Table Preservation
- **Version**: 1.0
- **Status**: Production Ready
- **Tests**: 8/8 passing (figures), 6/6 passing (tables)
- **Breaking Changes**: None
- **Backward Compatible**: Yes
