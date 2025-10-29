#!/usr/bin/env python3
"""
Download Academic Papers from arXiv
====================================

Downloads a curated list of important AI/ML papers from arXiv.

Usage:
    python scripts/download_pdfs.py [--custom-list papers.txt]
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict
import urllib.request
import urllib.error

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Curated list of important AI/ML papers
PAPER_LIST = [
    # Transformer & Attention
    {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need (Transformer)",
        "filename": "vaswani2017attention.pdf"
    },
    {
        "arxiv_id": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "filename": "devlin2018bert.pdf"
    },
    {
        "arxiv_id": "2005.14165",
        "title": "GPT-3: Language Models are Few-Shot Learners",
        "filename": "brown2020gpt3.pdf"
    },
    {
        "arxiv_id": "1910.13461",
        "title": "ALBERT: A Lite BERT",
        "filename": "lan2019albert.pdf"
    },
    {
        "arxiv_id": "1910.10683",
        "title": "T5: Exploring the Limits of Transfer Learning",
        "filename": "raffel2019t5.pdf"
    },

    # Computer Vision
    {
        "arxiv_id": "1512.03385",
        "title": "ResNet: Deep Residual Learning for Image Recognition",
        "filename": "he2015resnet.pdf"
    },
    {
        "arxiv_id": "2010.11929",
        "title": "Vision Transformer (ViT)",
        "filename": "dosovitskiy2020vit.pdf"
    },
    {
        "arxiv_id": "1409.1556",
        "title": "VGG: Very Deep Convolutional Networks",
        "filename": "simonyan2014vgg.pdf"
    },

    # Optimization & Training
    {
        "arxiv_id": "1412.6980",
        "title": "Adam: A Method for Stochastic Optimization",
        "filename": "kingma2014adam.pdf"
    },
    {
        "arxiv_id": "1607.06450",
        "title": "Layer Normalization",
        "filename": "ba2016layernorm.pdf"
    },
    {
        "arxiv_id": "1502.03167",
        "title": "Batch Normalization",
        "filename": "ioffe2015batchnorm.pdf"
    },

    # Generative Models
    {
        "arxiv_id": "1406.2661",
        "title": "GAN: Generative Adversarial Networks",
        "filename": "goodfellow2014gan.pdf"
    },
    {
        "arxiv_id": "1312.6114",
        "title": "VAE: Auto-Encoding Variational Bayes",
        "filename": "kingma2013vae.pdf"
    },

    # Reinforcement Learning
    {
        "arxiv_id": "1312.5602",
        "title": "DQN: Playing Atari with Deep Reinforcement Learning",
        "filename": "mnih2013dqn.pdf"
    },
    {
        "arxiv_id": "1707.06347",
        "title": "PPO: Proximal Policy Optimization",
        "filename": "schulman2017ppo.pdf"
    },

    # Neural Architecture
    {
        "arxiv_id": "1409.4842",
        "title": "GRU: Gated Recurrent Unit",
        "filename": "cho2014gru.pdf"
    },
    {
        "arxiv_id": "1505.04597",
        "title": "U-Net for Biomedical Image Segmentation",
        "filename": "ronneberger2015unet.pdf"
    },
]


class PaperDownloader:
    """Download papers from arXiv"""

    def __init__(self, output_dir: str = "data/raw_papers"):
        """Initialize downloader"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.downloaded = 0
        self.skipped = 0
        self.failed = 0

    def download_paper(self, arxiv_id: str, title: str, filename: str) -> bool:
        """
        Download a single paper from arXiv

        Args:
            arxiv_id: arXiv ID (e.g., "1706.03762")
            title: Paper title (for display)
            filename: Output filename

        Returns:
            True if successful, False otherwise
        """
        output_path = self.output_dir / filename

        # Check if already exists
        if output_path.exists():
            print(f"⏭️  Skipping: {title} (already exists)")
            self.skipped += 1
            return True

        # Construct arXiv PDF URL
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        print(f"📥 Downloading: {title}")
        print(f"   URL: {url}")
        print(f"   → {output_path}")

        try:
            # Download with progress
            urllib.request.urlretrieve(url, output_path)

            # Check file size
            file_size = output_path.stat().st_size
            if file_size < 1000:  # Less than 1KB, likely error page
                output_path.unlink()
                print(f"   ❌ Failed: File too small (likely error page)")
                self.failed += 1
                return False

            print(f"   ✅ Downloaded ({file_size / 1024:.1f} KB)")
            self.downloaded += 1

            # Be nice to arXiv servers
            time.sleep(1)

            return True

        except urllib.error.HTTPError as e:
            print(f"   ❌ HTTP Error {e.code}: {e.reason}")
            self.failed += 1
            return False
        except urllib.error.URLError as e:
            print(f"   ❌ URL Error: {e.reason}")
            self.failed += 1
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.failed += 1
            return False

    def download_all(self, paper_list: List[Dict]):
        """Download all papers in the list"""
        print("\n" + "=" * 60)
        print("📚 Downloading Academic Papers")
        print("=" * 60 + "\n")

        print(f"Total papers to download: {len(paper_list)}")
        print(f"Output directory: {self.output_dir.absolute()}\n")

        for i, paper in enumerate(paper_list, 1):
            print(f"\n[{i}/{len(paper_list)}]")
            self.download_paper(
                arxiv_id=paper["arxiv_id"],
                title=paper["title"],
                filename=paper["filename"]
            )

        # Summary
        print("\n" + "=" * 60)
        print("📊 Download Summary")
        print("=" * 60)
        print(f"✅ Downloaded: {self.downloaded}")
        print(f"⏭️  Skipped (existing): {self.skipped}")
        print(f"❌ Failed: {self.failed}")
        print(f"📁 Total files: {len(list(self.output_dir.glob('*.pdf')))}")
        print("=" * 60 + "\n")

        if self.downloaded > 0:
            print("💡 Next steps:")
            print("   1. Run 'make index' to index the papers")
            print("   2. Run 'make smoke-index' to validate quality")
            print()


def load_custom_list(file_path: str) -> List[Dict]:
    """
    Load custom paper list from file

    Format (one per line):
        arxiv_id|title|filename
        1706.03762|Attention Is All You Need|transformer.pdf
    """
    papers = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('|')
            if len(parts) == 3:
                papers.append({
                    'arxiv_id': parts[0].strip(),
                    'title': parts[1].strip(),
                    'filename': parts[2].strip()
                })

    return papers


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Download academic papers from arXiv"
    )
    parser.add_argument(
        '--custom-list',
        type=str,
        help='Path to custom paper list file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw_papers',
        help='Output directory (default: data/raw_papers)'
    )

    args = parser.parse_args()

    # Load paper list
    if args.custom_list:
        print(f"Loading custom paper list: {args.custom_list}")
        paper_list = load_custom_list(args.custom_list)
    else:
        paper_list = PAPER_LIST

    # Download
    downloader = PaperDownloader(output_dir=args.output_dir)
    downloader.download_all(paper_list)

    # Exit code
    if downloader.failed > 0 and downloader.downloaded == 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
