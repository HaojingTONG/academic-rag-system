"""
Incremental Indexer
===================

Main orchestrator for incremental document indexing:
1. Detect new/modified/deleted documents
2. Parse -> Chunk -> Embed -> Upsert
3. Maintain manifest with SHA256 tracking
4. Support namespace versioning

Usage:
    from indexer import IncrementalIndexer

    indexer = IncrementalIndexer()
    indexer.refresh()  # Incremental update
    indexer.reindex()  # Full reindex
"""

from .metadata import ManifestDB, DocumentRecord, compute_text_sha256
from .parse import DocumentParser, ParsedDocument
from .chunking import DocumentChunker, Chunk
from .embed import Embedder, EmbeddingResult
from .upsert import VectorDBUpserter

__all__ = [
    'ManifestDB',
    'DocumentRecord',
    'compute_text_sha256',
    'DocumentParser',
    'ParsedDocument',
    'DocumentChunker',
    'Chunk',
    'Embedder',
    'EmbeddingResult',
    'VectorDBUpserter',
    'IncrementalIndexer'
]

import yaml
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime


class IncrementalIndexer:
    """Orchestrate incremental indexing"""

    def __init__(self, config_path: str = "configs/index.yaml"):
        """Initialize indexer with config"""
        self.config = self._load_config(config_path)

        # Initialize components
        self.manifest = ManifestDB(self.config['manifest']['db_path'])
        self.parser = DocumentParser()
        self.chunker = DocumentChunker(
            chunk_size=self.config['chunking']['chunk_size'],
            overlap=self.config['chunking']['overlap']
        )
        self.embedder = Embedder(
            model_namespace=self.config['embedding']['model_namespace'],
            model_name=self.config['embedding']['model_name'],
            batch_size=self.config['embedding']['batch_size'],
            device=self.config['embedding']['device']
        )

        # Collection name
        namespace = self.config['embedding']['model_namespace'].replace('@', '_')
        collection_name = self.config['vector_db']['collection_template'].format(namespace=namespace)

        self.upserter = VectorDBUpserter(
            collection_name=collection_name,
            distance=self.config['vector_db']['distance']
        )

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def refresh(self, force: bool = False):
        """
        Incremental refresh: process new/modified documents

        Args:
            force: Force reprocessing even if not changed
        """
        print("\n" + "="*60)
        print("🔄 Incremental Index Refresh")
        print("="*60 + "\n")

        raw_dir = Path(self.config['directories']['raw_papers'])

        if not raw_dir.exists():
            print(f"❌ Raw papers directory not found: {raw_dir}")
            return

        # Find all PDF/TXT files
        files = list(raw_dir.glob("*.pdf")) + list(raw_dir.glob("*.txt"))

        print(f"Found {len(files)} files in {raw_dir}\n")

        processed = 0
        skipped = 0
        errors = 0

        for file_path in files:
            try:
                result = self._process_file(file_path, force=force)
                if result == 'processed':
                    processed += 1
                elif result == 'skipped':
                    skipped += 1
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
                errors += 1

        print("\n" + "="*60)
        print(f"✅ Refresh Complete")
        print(f"   Processed: {processed}")
        print(f"   Skipped: {skipped}")
        print(f"   Errors: {errors}")
        print("="*60 + "\n")

        # Show stats
        self._print_stats()

    def _process_file(self, file_path: Path, force: bool = False) -> str:
        """
        Process a single file

        Returns:
            'processed', 'skipped', or 'error'
        """
        print(f"📄 Processing: {file_path.name}")

        # Step 1: Parse
        parsed = self.parser.parse_file(str(file_path))

        # Step 2: Check if needs processing
        if not force and not self.manifest.needs_reindex(parsed.doc_id, parsed.sha256):
            print(f"   ⏭️  Skipped (no changes)\n")
            return 'skipped'

        # Step 3: Register in manifest
        self.manifest.register_document(
            doc_id=parsed.doc_id,
            source_path=str(file_path),
            sha256=parsed.sha256,
            metadata=parsed.metadata
        )

        # Step 4: Save parsed
        self.parser.save_parsed(parsed, self.config['parsing']['output_dir'])
        self.manifest.mark_parsed(parsed.doc_id)

        # Step 5: Chunk
        chunks = self.chunker.chunk_document(
            parsed,
            add_context=self.config['chunking']['add_context']
        )
        print(f"   ✂️  Created {len(chunks)} chunks")
        self.manifest.mark_chunked(parsed.doc_id)

        # Step 6: Embed
        embeddings = self.embedder.embed_chunks(chunks)
        print(f"   🔢 Generated {len(embeddings)} embeddings")

        # Step 7: Upsert to vector DB
        namespace = self.config['embedding']['model_namespace']
        self.upserter.upsert_chunks(chunks, embeddings, namespace)
        self.manifest.mark_upserted(parsed.doc_id, namespace)

        print(f"   ✅ Complete\n")
        return 'processed'

    def reindex(self, namespace: Optional[str] = None):
        """
        Full reindex: reprocess all documents

        Args:
            namespace: Optional new namespace for migration
        """
        print("\n" + "="*60)
        print("🔄 Full Reindex")
        print("="*60 + "\n")

        if namespace:
            print(f"Migrating to new namespace: {namespace}\n")
            # Update config namespace
            self.config['embedding']['model_namespace'] = namespace

        # Refresh with force=True
        self.refresh(force=True)

    def delete_document(self, doc_id: str, soft: bool = True):
        """Delete a document from index"""
        print(f"🗑️  Deleting document: {doc_id}")

        namespace = self.config['embedding']['model_namespace']

        # Delete from vector DB
        self.upserter.delete_document(doc_id, namespace)

        # Mark in manifest
        self.manifest.mark_deleted(doc_id, soft=soft)

        print(f"✅ Deleted: {doc_id}\n")

    def _print_stats(self):
        """Print indexing statistics"""
        print("\n📊 Statistics:")
        print("-" * 60)

        # Manifest stats
        manifest_stats = self.manifest.get_stats()
        print(f"Manifest:")
        print(f"  Active documents: {manifest_stats['total_active']}")
        print(f"  Deleted documents: {manifest_stats['total_deleted']}")
        print(f"  Parsed: {manifest_stats['parsed_count']}")
        print(f"  Chunked: {manifest_stats['chunked_count']}")
        print(f"  Upserted: {manifest_stats['upserted_count']}")

        # Vector DB stats
        vector_stats = self.upserter.get_stats()
        print(f"\nVector DB:")
        print(f"  Total chunks: {vector_stats['total_chunks']}")
        print(f"  Unique documents: {vector_stats['unique_documents']}")
        print(f"  Namespaces: {', '.join(vector_stats['namespaces'])}")

        print()


# CLI entry point
if __name__ == "__main__":
    import sys

    indexer = IncrementalIndexer()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "refresh":
            indexer.refresh()
        elif command == "reindex":
            indexer.reindex()
        elif command == "stats":
            indexer._print_stats()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python -m indexer [refresh|reindex|stats]")
    else:
        # Default: refresh
        indexer.refresh()
