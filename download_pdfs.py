#!/usr/bin/env python3
"""
Script to download PDF files for papers listed in high_quality_papers.json
"""
import json
import os
import requests
import time
from urllib.parse import urlparse

def load_papers():
    """Load papers from high_quality_papers.json"""
    with open('data/high_quality_papers.json', 'r') as f:
        data = json.load(f)
    return data['papers']

def get_existing_pdfs(directory):
    """Get list of existing PDF files in the directory"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return [f for f in os.listdir(directory) if f.endswith('.pdf')]

def extract_arxiv_id(paper_id):
    """Extract arXiv ID from paper ID"""
    return paper_id

def download_pdf(arxiv_id, output_dir):
    """Download PDF from arXiv"""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    output_path = os.path.join(output_dir, f"{arxiv_id}.pdf")
    
    try:
        print(f"Downloading {arxiv_id}...")
        response = requests.get(pdf_url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded {arxiv_id}.pdf")
        return True
    except Exception as e:
        print(f"✗ Failed to download {arxiv_id}: {e}")
        return False

def main():
    # Load papers
    papers = load_papers()
    print(f"Found {len(papers)} papers in high_quality_papers.json")
    
    # Check existing PDFs
    raw_papers_dir = "data/raw_papers"
    existing_pdfs = get_existing_pdfs(raw_papers_dir)
    print(f"Found {len(existing_pdfs)} existing PDFs")
    
    # Extract existing IDs from filenames
    existing_ids = set()
    for pdf_file in existing_pdfs:
        # Handle different filename patterns (e.g., 2009.14409v1.pdf -> 2009.14409)
        base_name = pdf_file.replace('.pdf', '')
        # Remove version suffix if present (e.g., v1, v2)
        if 'v' in base_name:
            base_id = base_name.split('v')[0]
            existing_ids.add(base_id)
        else:
            existing_ids.add(base_name)
    
    # Find papers to download
    papers_to_download = []
    for paper in papers:
        paper_id = paper['id']
        if paper_id not in existing_ids:
            papers_to_download.append(paper_id)
    
    print(f"Need to download {len(papers_to_download)} papers")
    
    if not papers_to_download:
        print("All papers already downloaded!")
        return
    
    # Download missing papers
    success_count = 0
    for i, paper_id in enumerate(papers_to_download, 1):
        print(f"[{i}/{len(papers_to_download)}] Processing {paper_id}")
        
        if download_pdf(paper_id, raw_papers_dir):
            success_count += 1
        
        # Add delay to be respectful to arXiv servers
        if i < len(papers_to_download):
            time.sleep(1)
    
    print(f"\nDownload completed!")
    print(f"Successfully downloaded: {success_count}/{len(papers_to_download)} papers")

if __name__ == "__main__":
    main()