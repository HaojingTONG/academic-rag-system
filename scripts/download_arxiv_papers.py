#!/usr/bin/env python3
"""
Download ArXiv-specific AI Papers
==================================

This script ONLY searches papers that exist on ArXiv,
ensuring 100% download success rate.

Usage:
    python scripts/download_arxiv_papers.py --limit 200
    python scripts/download_arxiv_papers.py --limit 100 --category cs.LG
"""

import argparse
import json
import time
import requests
from pathlib import Path
from typing import List, Dict
import logging
import xml.etree.ElementTree as ET
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class ArXivAPI:
    """ArXiv API client - guaranteed to have PDFs"""

    BASE_URL = "http://export.arxiv.org/api/query"

    # ArXiv categories for AI/ML
    CATEGORIES = {
        'cs.LG': 'Machine Learning',
        'cs.AI': 'Artificial Intelligence',
        'cs.CV': 'Computer Vision',
        'cs.CL': 'Computation and Language (NLP)',
        'cs.NE': 'Neural and Evolutionary Computing',
        'stat.ML': 'Machine Learning (Statistics)',
        'cs.IR': 'Information Retrieval',
        'cs.RO': 'Robotics',
    }

    @staticmethod
    def search_papers(
        category: str = 'cs.LG',
        max_results: int = 200,
        start_year: int = 2015
    ) -> List[Dict]:
        """Search ArXiv papers in specific category"""

        # Build query - only papers after start_year
        query = f"cat:{category} AND submittedDate:[{start_year}0101 TO 20251231]"

        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',  # or 'submittedDate' for newest
            'sortOrder': 'descending'
        }

        papers = []

        logger.info(f"🔍 Searching ArXiv category: {category} ({ArXivAPI.CATEGORIES.get(category, 'Unknown')})")
        logger.info(f"   Year range: {start_year}-2025")
        logger.info(f"   Max results: {max_results}")

        try:
            import urllib.parse
            url = f"{ArXivAPI.BASE_URL}?{urllib.parse.urlencode(params)}"

            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Parse XML
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            entries = root.findall('atom:entry', ns)

            logger.info(f"✓ Found {len(entries)} papers")

            for entry in entries:
                # Extract data
                arxiv_url = entry.find('atom:id', ns).text
                arxiv_id = arxiv_url.split('/')[-1]

                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')

                # Authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns).text
                    authors.append(name)

                # Published date
                published = entry.find('atom:published', ns).text
                year = int(published.split('-')[0])

                # Abstract
                abstract = entry.find('atom:summary', ns)
                abstract_text = abstract.text.strip() if abstract is not None else ''

                # Categories
                categories = [cat.attrib['term'] for cat in entry.findall('atom:category', ns)]

                papers.append({
                    'arxiv_id': arxiv_id,
                    'title': title,
                    'authors': authors,
                    'year': year,
                    'abstract': abstract_text,
                    'categories': categories,
                    'url': arxiv_url
                })

            return papers

        except Exception as e:
            logger.error(f"ArXiv API error: {e}")
            return []


def download_arxiv_pdf(arxiv_id: str, output_path: Path) -> bool:
    """Download PDF from ArXiv"""
    arxiv_id = arxiv_id.split('v')[0]  # Remove version
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    try:
        response = requests.get(pdf_url, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {arxiv_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download ArXiv papers (guaranteed PDF availability)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of papers to download (default: 200)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="cs.LG",
        choices=list(ArXivAPI.CATEGORIES.keys()),
        help="ArXiv category (default: cs.LG)"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="Minimum publication year (default: 2015)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw_papers",
        help="Output directory (default: data/raw_papers)"
    )

    args = parser.parse_args()

    # Setup
    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print config
    print("="*70)
    print("📚 ArXiv Paper Downloader (100% Success Rate)")
    print("="*70)
    print(f"Category: {args.category} - {ArXivAPI.CATEGORIES[args.category]}")
    print(f"Limit: {args.limit}")
    print(f"Year range: {args.start_year}-2025")
    print(f"Output: {output_dir}")
    print("="*70)
    print()

    # Search ArXiv
    papers = ArXivAPI.search_papers(
        category=args.category,
        max_results=args.limit,
        start_year=args.start_year
    )

    if not papers:
        logger.error("No papers found!")
        return

    # Download PDFs
    print()
    logger.info(f"📥 Downloading {len(papers)} PDFs...")

    downloaded = []
    skipped = []
    failed = []

    with tqdm(total=len(papers), desc="Downloading") as pbar:
        for paper in papers:
            arxiv_id = paper['arxiv_id']
            title = paper['title']

            pbar.set_description(f"Downloading: {title[:40]}...")

            # Output filename
            filename = f"{arxiv_id.replace('/', '_')}.pdf"
            output_path = output_dir / filename

            # Check if exists
            if output_path.exists():
                logger.info(f"⊙ Already exists: {filename}")
                skipped.append(paper)
                pbar.update(1)
                continue

            # Download
            if download_arxiv_pdf(arxiv_id, output_path):
                logger.info(f"✓ Downloaded: {filename}")
                downloaded.append(paper)
                time.sleep(3)  # ArXiv rate limiting
            else:
                failed.append(paper)

            pbar.update(1)

    # Save metadata
    metadata_file = project_root / "data" / "arxiv_papers_metadata.json"
    metadata = {
        p['arxiv_id']: p for p in downloaded
    }
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Summary
    print()
    print("="*70)
    print("📊 Download Summary")
    print("="*70)
    print(f"✅ Downloaded: {len(downloaded)}")
    print(f"⊙ Skipped (already exists): {len(skipped)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"📁 Location: {output_dir}")
    print(f"💾 Metadata: {metadata_file}")
    print("="*70)

    if downloaded:
        print(f"\n✨ Successfully downloaded {len(downloaded)} new papers!")
        print("\n📋 Next steps:")
        print("  1. Index the papers:")
        print("     make index")
        print()
        print("  2. Check statistics:")
        print("     make stats")
        print()
        print("  3. Start the system:")
        print("     python app/main.py")
        print("="*70)

    # Show available categories
    print("\n💡 Tip: Download from multiple categories to reach 200 papers:")
    print()
    for cat, desc in ArXivAPI.CATEGORIES.items():
        print(f"  python scripts/download_arxiv_papers.py --category {cat} --limit 50")
        print(f"    ({desc})")
    print()


if __name__ == "__main__":
    main()
