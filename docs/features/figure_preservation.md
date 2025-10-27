# Figure Preservation and Image Extraction Feature - Implementation Summary

## Overview
Extended the document ingestion pipeline to preserve figure captions and optionally enhance them with OCR-extracted text and AI-generated image descriptions. Figures are now detected, extracted, and stored as dedicated chunks or merged with narrative text based on caption size.

## Core Components Implemented

### 1. Figure Detection and Extraction (document_chunker.py)

#### Figure Block Detection (lines 685-701)
- Detects figure captions starting with: `Figure \d+`, `Fig.\d+`, `FIG.\d+`, `FIGURE \d+`
- Identifies multi-line captions (up to 5 lines)
- Stops at empty lines, section headers, or other figures/tables

#### Figure Caption Extraction (lines 703-743)
- Extracts complete captions with surrounding context
- Parses figure IDs (e.g., "Figure 3" -> "fig_3")
- Creates dedicated chunks for large captions (>= min_chunk_size)
- Marks small captions for merging with narrative text

#### Smart Caption Handling
**Large Captions** (>= min_chunk_size):
- Create dedicated `Chunk` with `section_type="figure"`
- Store complete caption as chunk text
- Set `has_figure=True`, `figure_id`, and `figure_caption` in metadata

**Small Captions** (< min_chunk_size):
- Merge with surrounding narrative using placeholder: `[FIGURE_n_SMALL:caption]`
- During merge, replace placeholder with actual caption text
- Update chunk metadata: `has_figure=True`, `merged_from_small_caption=True`
- Change `section_type` to `"figure_caption"`

### 2. Image Caption Extraction Module (image_caption_extractor.py)

#### Abstract Base Class: ImageCaptionExtractor
```python
class ImageCaptionExtractor(ABC):
    @abstractmethod
    def extract_text(image_path: str) -> str:
        """Extract text from image using OCR"""

    @abstractmethod
    def generate_caption(image_path: str) -> str:
        """Generate descriptive caption for image"""

    def extract_all(image_path: str) -> Dict:
        """Extract both OCR text and caption"""
```

#### Concrete Implementations

**StubImageCaptionExtractor**
- Graceful degradation when dependencies missing
- Checks for PaddleOCR, Tesseract, BLIP, CLIP availability
- Logs warnings but doesn't break ingestion
- Returns empty strings if libraries unavailable

**PaddleOCRExtractor**
- Uses PaddleOCR for text extraction from figure images
- Configurable GPU/CPU execution
- Extracts text lines and concatenates into single string

**BLIPCaptionExtractor**
- Uses Salesforce BLIP model for image captioning
- Generates natural language descriptions of figures
- Runs on CUDA if available, falls back to CPU

**HybridImageCaptionExtractor**
- Combines OCR and captioning capabilities
- Auto-initializes available extractors
- Provides complete image understanding

#### Factory Function
```python
def create_extractor(ocr_backend='auto', caption_backend='auto'):
    """Create appropriate extractor based on available dependencies"""
```

### 3. Integration with Document Chunker

#### Updated chunk_document Signature
```python
def chunk_document(text: str, paper_id: str, metadata: Dict = None,
                  image_extractor = None) -> List[Chunk]:
```

#### Processing Pipeline
1. Extract tables from text
2. Extract figures from text (without tables)
3. Chunk remaining text normally
4. Merge text, table, and figure chunks
5. **NEW**: Append image descriptions if `image_extractor` provided

#### Image Description Appending (lines 858-889)
```python
def _append_image_descriptions(chunks, figure_image_paths, image_extractor):
    """
    For each figure chunk with matching image path:
    1. Call extractor.extract_all(image_path)
    2. Append OCR text and caption to chunk.text
    3. Store in metadata: ocr_text, image_description
    4. Update char_count and word_count
    """
```

### 4. Metadata Schema

#### Figure Chunk Metadata
```python
{
    'has_figure': True,                    # Flag for filtering
    'figure_caption': str,                 # Raw caption text
    'figure_id': str,                      # e.g., "fig_3"
    'section_type': 'figure',              # or 'figure_caption' if merged
    'merged_from_small_caption': bool,     # True if merged with narrative

    # Optional (when image extraction used):
    'ocr_text': str,                       # Text extracted from image
    'image_description': str,              # AI-generated caption
}
```

### 5. Vector Store Integration (vector_store.py)

#### New Search Method
```python
def search_with_figures(query: str, top_k: int = 5) -> Dict:
    """Search only chunks containing figures"""
    filter_condition = {"has_figure": True}
    return self.search(query, top_k, filter_condition)
```

#### Updated Statistics
- Added `has_figure` to `content_features` tracking (line 178)
- Statistics now report figure chunk counts alongside tables, formulas, etc.

### 6. Content Analysis
Updated `_analyze_content_features` to detect figures:
```python
'has_figure': bool(re.search(r'^\s*(?:Figure|Fig\.)\s+\d+', text, re.MULTILINE | re.IGNORECASE))
```

## Usage Examples

### Basic Usage (No Image Processing)
```python
from src.processor.document_chunker import DocumentChunker, ChunkingConfig

config = ChunkingConfig(strategy='hybrid', chunk_size=500)
chunker = DocumentChunker(config)

text = """
Introduction

Figure 1: System architecture.
The diagram shows the main components.

Methods

Our implementation follows standard practices.
"""

chunks = chunker.chunk_document(text, "paper_123")

# Find figure chunks
figure_chunks = [c for c in chunks if c.metadata.get('has_figure')]
for chunk in figure_chunks:
    print(f"Figure: {chunk.metadata['figure_id']}")
    print(f"Caption: {chunk.metadata['figure_caption']}")
```

### With Image Extraction
```python
from src.processor.image_caption_extractor import create_extractor

# Create extractor (auto-detects available backends)
extractor = create_extractor()

# Provide image paths in metadata
metadata = {
    'title': 'My Paper',
    'figure_image_paths': {
        'fig_1': '/path/to/figure1.png',
        'fig_2': '/path/to/figure2.png'
    }
}

chunks = chunker.chunk_document(text, "paper_123", metadata, extractor)

# Figure chunks now include OCR and AI captions
for chunk in figure_chunks:
    if chunk.metadata.get('ocr_text'):
        print(f"OCR: {chunk.metadata['ocr_text']}")
    if chunk.metadata.get('image_description'):
        print(f"AI Caption: {chunk.metadata['image_description']}")
```

### Vector Store Search
```python
from src.retriever.vector_store import VectorStore

vs = VectorStore()

# Search only in figure content
results = vs.search_with_figures("architecture diagram", top_k=5)
```

## Test Coverage

Created comprehensive test suite in `test_figure_preservation.py`:

1. **Basic Figure Extraction** - Standalone figure captions
2. **Small Caption Merging** - Captions merged with narrative
3. **Multiple Figures** - Documents with 2+ figures
4. **Figure vs Table Detection** - Correct differentiation
5. **Image Caption Extraction (Mock)** - OCR/captioning integration
6. **Stub Extractor** - Graceful degradation without dependencies
7. **Metadata Preservation** - Custom metadata survives processing
8. **Figure ID Extraction** - Various caption formats

**All 8 tests passed (100% success rate)**

## Key Design Decisions

### 1. Small Caption Merging
**Rationale**: Very short captions (e.g., "Fig. 2: Results.") provide little value as standalone chunks. Merging with surrounding narrative maintains context and improves embedding quality.

**Implementation**: Use placeholder `[FIGURE_n_SMALL:caption]` during extraction, then replace during merge phase. This preserves document order while allowing metadata updates.

### 2. Graceful Degradation
**Rationale**: OCR/captioning libraries are optional dependencies. System should work without them.

**Implementation**:
- `StubImageCaptionExtractor` checks for available libraries
- Logs warnings but returns empty strings
- Never fails ingestion due to missing dependencies
- Extension points for future custom extractors

### 3. Separation of Concerns
**Rationale**: Image processing is computationally expensive and optional.

**Implementation**:
- Image extraction is separate module (`image_caption_extractor.py`)
- Chunker accepts optional `image_extractor` parameter
- Can run chunking without image processing
- Image paths passed via metadata, not hardcoded

### 4. Metadata-Driven Image Paths
**Rationale**: PDF processor already extracts images to files. Don't re-extract.

**Implementation**:
- Expect `figure_image_paths` dict in metadata: `{figure_id: image_path}`
- Chunker matches `figure_id` from caption with paths
- Only processes figures with available images

## Files Modified/Created

### Modified
1. **src/processor/document_chunker.py**
   - Added figure detection methods (185 lines)
   - Updated chunk_document signature
   - Added image description appending
   - Updated metadata analysis

2. **src/retriever/vector_store.py**
   - Added `search_with_figures()` method
   - Updated statistics tracking

### Created
1. **src/processor/image_caption_extractor.py** (370 lines)
   - Abstract base class
   - Stub, PaddleOCR, BLIP, and Hybrid extractors
   - Factory function and utilities

2. **test_figure_preservation.py** (434 lines)
   - 8 comprehensive test cases
   - Mock-based image extraction testing

3. **demo_figure_preservation.py** (272 lines)
   - Interactive demonstration
   - Shows all major features

4. **FIGURE_PRESERVATION_SUMMARY.md** (this document)

## Backward Compatibility

✅ **No Breaking Changes**
- Existing table extraction logic unchanged
- Optional `image_extractor` parameter (default: None)
- Metadata structure extended, not modified
- All existing tests pass
- Documents without figures process normally

## Performance Considerations

- **Figure extraction**: O(n) single pass through lines
- **Caption merging**: O(m) where m = number of small captions
- **Image processing**: Optional, only when extractor provided
- **No impact** on documents without figures
- **Graceful degradation**: Missing dependencies don't slow down system

## Future Enhancements

Potential improvements (not implemented):
1. Detect figures without captions (image-only detection)
2. Extract subfigure labels (a), (b), (c)
3. Support LaTeX figure environments
4. Handle multi-page figure continuations
5. Cache image descriptions to avoid re-processing
6. Support video/animation descriptions
7. Extract figure URLs from papers with embedded images

## Dependencies

### Required (Already Installed)
- Python 3.7+
- Standard library modules

### Optional (For Image Extraction)
- `paddleocr` - OCR text extraction
- `pytesseract` - Alternative OCR backend
- `transformers` + `pillow` - BLIP image captioning
- `clip` - Alternative captioning backend
- `torch` - Deep learning framework

**Installation**:
```bash
# For OCR
pip install paddleocr

# For image captioning
pip install transformers pillow torch

# Or install all at once
pip install paddleocr transformers pillow torch
```

## Error Handling

- **Missing image files**: Logged as warnings, ingestion continues
- **OCR failures**: Return empty string, don't fail
- **Captioning failures**: Return empty string, don't fail
- **Missing dependencies**: Use stub extractor, log warning
- **Invalid figure captions**: Treated as regular text
- **Corrupted images**: Caught by extractor, logged, continue

All errors are non-fatal to ensure robust ingestion pipeline.
