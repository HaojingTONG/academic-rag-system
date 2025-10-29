"""
Document Chunking
=================

Chunk documents with:
- Token-based chunking (target 700 tokens)
- Overlap (12% default)
- Title/section context preservation
- Metadata propagation

Usage:
    chunker = DocumentChunker(chunk_size=700, overlap=0.12)
    chunks = chunker.chunk_document(parsed_doc)
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("⚠️ tiktoken not available, using approximate tokenization")


@dataclass
class Chunk:
    """Document chunk"""
    chunk_id: str
    doc_id: str
    text: str
    tokens: int
    title: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    chunk_index: int = 0
    total_chunks: int = 0
    metadata: Optional[Dict] = None


class DocumentChunker:
    """Chunk documents with overlap and context"""

    def __init__(
        self,
        chunk_size: int = 700,
        overlap: float = 0.12,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize chunker

        Args:
            chunk_size: Target chunk size in tokens
            overlap: Overlap ratio (0.0-1.0)
            encoding_name: tiktoken encoding name
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.overlap_tokens = int(chunk_size * overlap)

        # Initialize tokenizer
        if HAS_TIKTOKEN:
            self.encoding = tiktoken.get_encoding(encoding_name)
        else:
            self.encoding = None

    def chunk_document(self, parsed_doc, add_context: bool = True) -> List[Chunk]:
        """
        Chunk parsed document

        Args:
            parsed_doc: ParsedDocument object
            add_context: Add title/section context to chunks

        Returns:
            List of Chunk objects
        """
        chunks = []

        # Split text into paragraphs
        paragraphs = self._split_paragraphs(parsed_doc.text)

        # Track current chunk
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        # Track section context
        current_section = None
        sections_map = self._build_sections_map(parsed_doc.sections, parsed_doc.text)

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            # Update section context
            section_name = sections_map.get(para, current_section)
            if section_name:
                current_section = section_name

            # Check if adding this paragraph exceeds chunk size
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = '\n\n'.join(current_chunk)

                # Add context prefix
                if add_context:
                    chunk_text = self._add_context_prefix(
                        chunk_text,
                        title=parsed_doc.metadata.get('title'),
                        section=current_section
                    )

                chunk_id = f"{parsed_doc.doc_id}_chunk_{chunk_index}"

                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    doc_id=parsed_doc.doc_id,
                    text=chunk_text,
                    tokens=self._count_tokens(chunk_text),
                    title=parsed_doc.metadata.get('title'),
                    section=current_section,
                    chunk_index=chunk_index,
                    metadata=parsed_doc.metadata
                ))

                chunk_index += 1

                # Start new chunk with overlap
                current_chunk = self._get_overlap_paragraphs(
                    current_chunk,
                    self.overlap_tokens
                )
                current_tokens = sum(self._count_tokens(p) for p in current_chunk)

            # Add paragraph
            current_chunk.append(para)
            current_tokens += para_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)

            if add_context:
                chunk_text = self._add_context_prefix(
                    chunk_text,
                    title=parsed_doc.metadata.get('title'),
                    section=current_section
                )

            chunk_id = f"{parsed_doc.doc_id}_chunk_{chunk_index}"

            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=parsed_doc.doc_id,
                text=chunk_text,
                tokens=self._count_tokens(chunk_text),
                title=parsed_doc.metadata.get('title'),
                section=current_section,
                chunk_index=chunk_index,
                metadata=parsed_doc.metadata
            ))

        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines
        paragraphs = re.split(r'\n\n+', text)

        # Filter empty and very short paragraphs
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 20]

        return paragraphs

    def _build_sections_map(self, sections: List[Dict], text: str) -> Dict[str, str]:
        """Build mapping from paragraph to section"""
        sections_map = {}

        # Simple approach: map paragraphs near section headers to that section
        paragraphs = self._split_paragraphs(text)

        for section in sections:
            section_name = section['name']

            # Find paragraphs mentioning this section
            for para in paragraphs:
                if section_name.lower() in para.lower()[:100]:
                    sections_map[para] = section_name

        return sections_map

    def _get_overlap_paragraphs(self, paragraphs: List[str], target_tokens: int) -> List[str]:
        """Get last N paragraphs that fit in target tokens"""
        overlap_paras = []
        tokens = 0

        # Work backwards
        for para in reversed(paragraphs):
            para_tokens = self._count_tokens(para)
            if tokens + para_tokens <= target_tokens:
                overlap_paras.insert(0, para)
                tokens += para_tokens
            else:
                break

        return overlap_paras

    def _add_context_prefix(
        self,
        text: str,
        title: Optional[str] = None,
        section: Optional[str] = None
    ) -> str:
        """Add context prefix to chunk"""
        prefix_parts = []

        if title:
            prefix_parts.append(f"Title: {title}")

        if section:
            prefix_parts.append(f"Section: {section}")

        if prefix_parts:
            prefix = ' | '.join(prefix_parts)
            return f"{prefix}\n\n{text}"

        return text

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Approximate: ~4 chars per token
            return len(text) // 4


# Example usage
if __name__ == "__main__":
    from indexer.parse import DocumentParser

    # Parse document
    parser = DocumentParser()
    # parsed = parser.parse_file('data/raw_papers/attention_is_all_you_need.pdf')

    # Chunk it
    chunker = DocumentChunker(chunk_size=700, overlap=0.12)
    # chunks = chunker.chunk_document(parsed)

    # print(f"Created {len(chunks)} chunks")
    # for i, chunk in enumerate(chunks[:3]):
    #     print(f"\nChunk {i}:")
    #     print(f"  ID: {chunk.chunk_id}")
    #     print(f"  Tokens: {chunk.tokens}")
    #     print(f"  Section: {chunk.section}")
    #     print(f"  Text preview: {chunk.text[:100]}...")

    print("Chunker ready")
