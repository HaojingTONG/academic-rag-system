#!/usr/bin/env python3
"""
Download Curated AI Papers
==========================

Download papers from the curated list (scripts/curated_papers.json)
This is simpler and more reliable than the API-based approach.

Usage:
    python scripts/download_curated_papers.py
    python scripts/download_curated_papers.py --limit 50
"""

import argparse
import json
import time
import requests
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def download_arxiv_pdf(arxiv_id: str, output_path: Path) -> bool:
    """Download PDF from ArXiv"""
    # Clean arxiv_id
    arxiv_id = arxiv_id.split('v')[0]
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
    parser = argparse.ArgumentParser(description="Download curated AI papers")
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max number of papers to download (default: 200)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by category (e.g., 'nlp', 'cv', 'transformer')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw_papers",
        help="Output directory (default: data/raw_papers)"
    )

    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    curated_file = project_root / "scripts" / "curated_papers.json"
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load curated papers
    if not curated_file.exists():
        logger.error(f"Curated papers file not found: {curated_file}")
        return

    with open(curated_file, 'r') as f:
        data = json.load(f)

    papers = data.get('papers', [])

    # Filter by category
    if args.category:
        papers = [p for p in papers if p.get('category') == args.category]
        logger.info(f"Filtered to {len(papers)} papers in category '{args.category}'")

    # Sort by citations and limit
    papers = sorted(papers, key=lambda x: x.get('estimated_citations', 0), reverse=True)
    papers = papers[:args.limit]

    print("="*70)
    print("📚 Curated AI Papers Downloader")
    print("="*70)
    print(f"Papers to download: {len(papers)}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    print()

    # Download
    downloaded = []
    failed = []
    skipped = []

    with tqdm(total=len(papers), desc="Downloading") as pbar:
        for paper in papers:
            arxiv_id = paper.get('arxiv_id')
            title = paper.get('title', 'Unknown')

            pbar.set_description(f"Downloading: {title[:40]}...")

            if not arxiv_id:
                logger.warning(f"No ArXiv ID for: {title}")
                failed.append(paper)
                pbar.update(1)
                continue

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
                time.sleep(3)  # Rate limiting (be nice to ArXiv)
            else:
                failed.append(paper)

            pbar.update(1)

    # Summary
    print()
    print("="*70)
    print("📊 Download Summary")
    print("="*70)
    print(f"✅ Downloaded: {len(downloaded)}")
    print(f"⊙ Skipped (already exists): {len(skipped)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"📁 Location: {output_dir}")
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

    if failed:
        print(f"\n⚠️  Failed papers ({len(failed)}):")
        for paper in failed[:5]:
            print(f"  • {paper.get('title')}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")


if __name__ == "__main__":
    main()
