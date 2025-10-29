"""
Document Parser
===============

Extracts text and metadata from PDF/TXT files:
- Clean text extraction (handle hyphenation, ligatures)
- Metadata extraction (title, authors, year, venue, DOI)
- Section detection and structure preservation
- Persist to JSON format

Usage:
    parser = DocumentParser()
    result = parser.parse_file('paper.pdf')
    # result = {'doc_id', 'text', 'metadata', 'sections', 'sha256'}
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("⚠️ PyMuPDF not available, PDF parsing limited")


@dataclass
class ParsedDocument:
    """Parsed document result"""
    doc_id: str
    text: str
    metadata: Dict
    sections: List[Dict]
    sha256: str
    source_path: str


class DocumentParser:
    """Parse PDF and TXT documents"""

    def __init__(self):
        """Initialize parser"""
        self.has_pymupdf = HAS_PYMUPDF

    def parse_file(self, file_path: str) -> ParsedDocument:
        """
        Parse document file

        Args:
            file_path: Path to PDF or TXT file

        Returns:
            ParsedDocument object
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate doc_id from filename
        doc_id = self._generate_doc_id(file_path)

        # Parse based on extension
        if file_path.suffix.lower() == '.pdf':
            if not self.has_pymupdf:
                raise RuntimeError("PyMuPDF required for PDF parsing. Install with: pip install pymupdf")
            result = self._parse_pdf(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            result = self._parse_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Compute SHA256 of cleaned text
        sha256 = hashlib.sha256(result['text'].encode('utf-8')).hexdigest()

        return ParsedDocument(
            doc_id=doc_id,
            text=result['text'],
            metadata=result['metadata'],
            sections=result['sections'],
            sha256=sha256,
            source_path=str(file_path)
        )

    def _generate_doc_id(self, file_path: Path) -> str:
        """
        Generate document ID from filename

        Args:
            file_path: Path to file

        Returns:
            Document ID (sanitized filename)
        """
        # Remove extension and sanitize
        doc_id = file_path.stem
        doc_id = re.sub(r'[^a-z0-9_-]', '_', doc_id.lower())
        doc_id = re.sub(r'_+', '_', doc_id)
        return doc_id.strip('_')

    def _parse_pdf(self, file_path: Path) -> Dict:
        """Parse PDF file"""
        doc = fitz.open(file_path)

        # Extract metadata
        metadata = self._extract_pdf_metadata(doc)

        # Extract text with structure
        sections = []
        full_text_parts = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()

            # Clean text
            page_text = self._clean_text(page_text)

            # Detect sections (simple approach)
            page_sections = self._detect_sections(page_text, page_num)
            sections.extend(page_sections)

            full_text_parts.append(page_text)

        # Combine full text
        full_text = '\n\n'.join(full_text_parts)

        doc.close()

        return {
            'text': full_text,
            'metadata': metadata,
            'sections': sections
        }

    def _parse_text(self, file_path: Path) -> Dict:
        """Parse plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Clean text
        text = self._clean_text(text)

        # Extract basic metadata from content
        metadata = self._extract_text_metadata(text)

        # Detect sections
        sections = self._detect_sections(text, page_num=0)

        return {
            'text': text,
            'metadata': metadata,
            'sections': sections
        }

    def _extract_pdf_metadata(self, doc) -> Dict:
        """Extract metadata from PDF"""
        metadata = {}

        # Get PDF metadata
        pdf_meta = doc.metadata

        # Try to extract from PDF metadata
        if pdf_meta.get('title'):
            metadata['title'] = pdf_meta['title']
        if pdf_meta.get('author'):
            metadata['authors'] = pdf_meta['author']

        # Try to extract from first page
        if len(doc) > 0:
            first_page = doc[0].get_text()

            # Extract title (usually largest text on first page)
            if 'title' not in metadata:
                title = self._extract_title_from_text(first_page)
                if title:
                    metadata['title'] = title

            # Extract year
            year = self._extract_year(first_page)
            if year:
                metadata['year'] = year

            # Extract DOI
            doi = self._extract_doi(first_page)
            if doi:
                metadata['doi'] = doi

        return metadata

    def _extract_text_metadata(self, text: str) -> Dict:
        """Extract metadata from plain text"""
        metadata = {}

        # Extract from first few lines
        lines = text.split('\n')[:10]
        first_text = '\n'.join(lines)

        # Title (first non-empty line)
        title = self._extract_title_from_text(first_text)
        if title:
            metadata['title'] = title

        # Year
        year = self._extract_year(text)
        if year:
            metadata['year'] = year

        # DOI
        doi = self._extract_doi(text)
        if doi:
            metadata['doi'] = doi

        return metadata

    def _extract_title_from_text(self, text: str) -> Optional[str]:
        """Extract title from text"""
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        if not lines:
            return None

        # First substantial line is usually title
        for line in lines:
            if len(line) > 10 and not line.isupper():
                return line

        return lines[0] if lines else None

    def _extract_year(self, text: str) -> Optional[int]:
        """Extract publication year"""
        # Look for 4-digit year (1900-2099)
        matches = re.findall(r'\b(19|20)\d{2}\b', text)

        if matches:
            # Return first year found
            return int(matches[0])

        return None

    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI"""
        # DOI pattern: 10.xxxx/yyyy
        match = re.search(r'10\.\d{4,}/[^\s]+', text)

        if match:
            doi = match.group(0)
            # Clean trailing punctuation
            doi = doi.rstrip('.,;:')
            return doi

        return None

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove hyphenation at line breaks
        text = re.sub(r'-\n', '', text)

        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        # Fix common ligatures
        ligatures = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            'ﬃ': 'ffi',
            'ﬄ': 'ffl'
        }
        for lig, replacement in ligatures.items():
            text = text.replace(lig, replacement)

        # Remove form feed characters
        text = text.replace('\f', '\n')

        return text.strip()

    def _detect_sections(self, text: str, page_num: int) -> List[Dict]:
        """
        Detect sections in text

        Args:
            text: Page text
            page_num: Page number

        Returns:
            List of section dictionaries
        """
        sections = []

        # Simple section detection based on common headings
        section_patterns = [
            r'^(Abstract|ABSTRACT)\s*$',
            r'^(\d+\.?\s+)?Introduction\s*$',
            r'^(\d+\.?\s+)?Related Work\s*$',
            r'^(\d+\.?\s+)?Background\s*$',
            r'^(\d+\.?\s+)?Method(s|ology)?\s*$',
            r'^(\d+\.?\s+)?Experiment(s)?\s*$',
            r'^(\d+\.?\s+)?Results?\s*$',
            r'^(\d+\.?\s+)?Discussion\s*$',
            r'^(\d+\.?\s+)?Conclusion(s)?\s*$',
            r'^(\d+\.?\s+)?References?\s*$'
        ]

        lines = text.split('\n')

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            for pattern in section_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    # Extract section name
                    section_name = re.sub(r'^\d+\.?\s+', '', line_stripped)

                    sections.append({
                        'name': section_name,
                        'page': page_num,
                        'line': i
                    })
                    break

        return sections

    def save_parsed(self, parsed: ParsedDocument, output_dir: str = "data/parsed"):
        """
        Save parsed document to JSON

        Args:
            parsed: ParsedDocument object
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{parsed.doc_id}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(parsed), f, indent=2, ensure_ascii=False)

        print(f"✅ Saved parsed document: {output_file}")

    def load_parsed(self, doc_id: str, input_dir: str = "data/parsed") -> Optional[ParsedDocument]:
        """
        Load parsed document from JSON

        Args:
            doc_id: Document ID
            input_dir: Input directory

        Returns:
            ParsedDocument or None if not found
        """
        input_file = Path(input_dir) / f"{doc_id}.json"

        if not input_file.exists():
            return None

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return ParsedDocument(**data)


# Example usage
if __name__ == "__main__":
    parser = DocumentParser()

    # Parse a PDF
    # result = parser.parse_file('data/raw_papers/attention_is_all_you_need.pdf')
    # print(f"Doc ID: {result.doc_id}")
    # print(f"Title: {result.metadata.get('title')}")
    # print(f"Year: {result.metadata.get('year')}")
    # print(f"SHA256: {result.sha256}")
    # print(f"Text length: {len(result.text)}")
    # print(f"Sections: {len(result.sections)}")

    # parser.save_parsed(result)
    print("Parser ready")
