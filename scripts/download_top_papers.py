#!/usr/bin/env python3
"""
Download Top AI Papers by Citation Count
=========================================

This script downloads the most cited AI papers from various sources:
1. Semantic Scholar API (primary source)
2. ArXiv (for full-text PDFs)
3. Manual curated list (backup)

Usage:
    python scripts/download_top_papers.py --limit 200
    python scripts/download_top_papers.py --limit 50 --field "machine learning"
    python scripts/download_top_papers.py --limit 100 --start-year 2015
"""

import argparse
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Paper metadata"""
    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    citation_count: int
    arxiv_id: Optional[str] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None


class SemanticScholarAPI:
    """Semantic Scholar API client"""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"x-api-key": api_key})

    def search_papers(
        self,
        query: str,
        limit: int = 100,
        fields: List[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None
    ) -> List[Dict]:
        """Search papers with filters"""

        if fields is None:
            fields = [
                "paperId", "title", "authors", "year", "citationCount",
                "externalIds", "venue", "url", "abstract"
            ]

        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 100),  # API max is 100 per request
            "fields": ",".join(fields),
            "sort": "citationCount:desc"  # Sort by citations
        }

        if year_min:
            params["year"] = f"{year_min}-"
        if year_max and year_min:
            params["year"] = f"{year_min}-{year_max}"

        all_papers = []
        offset = 0

        with tqdm(total=limit, desc="Fetching papers") as pbar:
            while len(all_papers) < limit:
                params["offset"] = offset

                try:
                    response = self.session.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()

                    papers = data.get("data", [])
                    if not papers:
                        break

                    all_papers.extend(papers)
                    pbar.update(len(papers))

                    offset += len(papers)

                    # Rate limiting
                    time.sleep(1)

                except requests.exceptions.RequestException as e:
                    logger.error(f"API request failed: {e}")
                    break

        return all_papers[:limit]

    def get_paper_details(self, paper_id: str) -> Optional[Dict]:
        """Get detailed paper information"""
        url = f"{self.BASE_URL}/paper/{paper_id}"
        params = {
            "fields": "paperId,title,authors,year,citationCount,externalIds,venue,url,abstract"
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get paper {paper_id}: {e}")
            return None


class ArXivDownloader:
    """ArXiv PDF downloader"""

    BASE_URL = "https://arxiv.org"

    @staticmethod
    def download_pdf(arxiv_id: str, output_path: Path) -> bool:
        """Download PDF from ArXiv"""

        # Clean arxiv_id (remove version if present)
        arxiv_id = arxiv_id.split('v')[0]

        pdf_url = f"{ArXivDownloader.BASE_URL}/pdf/{arxiv_id}.pdf"

        try:
            logger.info(f"Downloading {arxiv_id}...")
            response = requests.get(pdf_url, stream=True, timeout=60)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"✓ Downloaded: {output_path.name}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Failed to download {arxiv_id}: {e}")
            return False

    @staticmethod
    def search_arxiv(query: str, max_results: int = 100) -> List[Dict]:
        """Search ArXiv API"""
        import urllib.parse

        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        url = f"{base_url}?{urllib.parse.urlencode(params)}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)

            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()

                papers.append({
                    'arxiv_id': arxiv_id,
                    'title': title
                })

            return papers

        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []


class PaperDownloader:
    """Main paper downloader orchestrator"""

    def __init__(self, output_dir: Path, metadata_file: Path):
        self.output_dir = output_dir
        self.metadata_file = metadata_file
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ss_api = SemanticScholarAPI()
        self.downloaded = []
        self.failed = []

    def download_top_papers(
        self,
        limit: int = 200,
        field: str = "artificial intelligence",
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ):
        """Download top papers by citation count"""

        logger.info(f"🔍 Searching for top {limit} papers in '{field}'...")

        # Step 1: Get papers from Semantic Scholar
        papers_data = self.ss_api.search_papers(
            query=field,
            limit=limit * 2,  # Get more to filter
            year_min=start_year,
            year_max=end_year
        )

        if not papers_data:
            logger.error("No papers found!")
            return

        logger.info(f"✓ Found {len(papers_data)} papers")

        # Step 2: Parse and filter papers
        papers = []
        for p in papers_data:
            arxiv_id = None
            external_ids = p.get('externalIds', {})
            if external_ids:
                arxiv_id = external_ids.get('ArXiv')

            paper = Paper(
                paper_id=p.get('paperId', ''),
                title=p.get('title', 'Unknown'),
                authors=[a.get('name', '') for a in p.get('authors', [])],
                year=p.get('year'),
                citation_count=p.get('citationCount', 0),
                arxiv_id=arxiv_id,
                venue=p.get('venue'),
                url=p.get('url'),
                abstract=p.get('abstract', '')
            )

            papers.append(paper)

        # Sort by citations and take top N
        papers.sort(key=lambda x: x.citation_count, reverse=True)
        papers = papers[:limit]

        logger.info(f"📊 Top paper: '{papers[0].title}' ({papers[0].citation_count} citations)")

        # Step 3: Download PDFs
        logger.info(f"\n📥 Downloading {len(papers)} PDFs...")

        with tqdm(total=len(papers), desc="Downloading PDFs") as pbar:
            for paper in papers:
                pbar.set_description(f"Downloading: {paper.title[:50]}...")

                if paper.arxiv_id:
                    # Try ArXiv
                    filename = f"{paper.arxiv_id.replace('/', '_')}.pdf"
                    output_path = self.output_dir / filename

                    if output_path.exists():
                        logger.info(f"⊙ Skipped (exists): {filename}")
                        self.downloaded.append(paper)
                    elif ArXivDownloader.download_pdf(paper.arxiv_id, output_path):
                        self.downloaded.append(paper)
                        time.sleep(2)  # Rate limiting
                    else:
                        self.failed.append(paper)
                else:
                    # No ArXiv ID
                    logger.warning(f"⊘ No ArXiv ID: {paper.title[:50]}")
                    self.failed.append(paper)

                pbar.update(1)

        # Step 4: Save metadata
        self.save_metadata()

        # Summary
        logger.info("\n" + "="*60)
        logger.info("📊 Download Summary:")
        logger.info(f"  ✓ Downloaded: {len(self.downloaded)}")
        logger.info(f"  ✗ Failed: {len(self.failed)}")
        logger.info(f"  📁 Location: {self.output_dir}")
        logger.info("="*60)

        if self.failed:
            logger.info(f"\n⚠️  {len(self.failed)} papers failed to download:")
            for paper in self.failed[:10]:  # Show first 10
                logger.info(f"  • {paper.title[:60]} ({paper.citation_count} citations)")
            if len(self.failed) > 10:
                logger.info(f"  ... and {len(self.failed) - 10} more")

    def save_metadata(self):
        """Save downloaded papers metadata"""
        metadata = {
            p.paper_id: asdict(p) for p in self.downloaded
        }

        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Metadata saved: {self.metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download top AI papers by citation count"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of papers to download (default: 200)"
    )
    parser.add_argument(
        "--field",
        type=str,
        default="artificial intelligence machine learning",
        help="Research field (default: 'artificial intelligence machine learning')"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Minimum publication year (default: no limit)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Maximum publication year (default: no limit)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw_papers",
        help="Output directory (default: data/raw_papers)"
    )

    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output_dir
    metadata_file = project_root / "data" / "downloaded_papers_metadata.json"

    # Print configuration
    print("="*60)
    print("🤖 AI Paper Downloader")
    print("="*60)
    print(f"Field: {args.field}")
    print(f"Limit: {args.limit}")
    print(f"Year range: {args.start_year or 'any'} - {args.end_year or 'any'}")
    print(f"Output: {output_dir}")
    print("="*60)
    print()

    # Download
    downloader = PaperDownloader(output_dir, metadata_file)
    downloader.download_top_papers(
        limit=args.limit,
        field=args.field,
        start_year=args.start_year,
        end_year=args.end_year
    )

    # Next steps
    print("\n" + "="*60)
    print("✅ Download complete!")
    print("\n📋 Next steps:")
    print("  1. Index the papers: make index")
    print("  2. Check stats: make stats")
    print("  3. Start the system: python app/main.py")
    print("="*60)


if __name__ == "__main__":
    main()
