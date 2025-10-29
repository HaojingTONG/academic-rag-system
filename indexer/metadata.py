"""
Manifest Database Manager
=========================

Tracks document indexing state using SQLite with:
- Document ID, source path, SHA256 hash
- Parse/chunk/embed/upsert timestamps
- Embedding model namespace versioning
- Soft delete support (is_active flag)

Schema:
    documents(
        doc_id TEXT PRIMARY KEY,
        source_path TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        title TEXT,
        authors TEXT,
        year INTEGER,
        venue TEXT,
        doi TEXT,
        parsed_at TIMESTAMP,
        chunked_at TIMESTAMP,
        embed_model_ns TEXT,
        upserted_at TIMESTAMP,
        deleted_at TIMESTAMP,
        is_active BOOLEAN DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )

Usage:
    manifest = ManifestDB('data/manifest.db')
    manifest.register_document(doc_id, source_path, sha256, metadata)
    needs_reindex = manifest.needs_reindex(doc_id, new_sha256)
"""

import sqlite3
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict


@dataclass
class DocumentRecord:
    """Document record in manifest"""
    doc_id: str
    source_path: str
    sha256: str
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    parsed_at: Optional[str] = None
    chunked_at: Optional[str] = None
    embed_model_ns: Optional[str] = None
    upserted_at: Optional[str] = None
    deleted_at: Optional[str] = None
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ManifestDB:
    """Manage document indexing manifest"""

    def __init__(self, db_path: str = "data/manifest.db"):
        """
        Initialize manifest database

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    title TEXT,
                    authors TEXT,
                    year INTEGER,
                    venue TEXT,
                    doi TEXT,
                    parsed_at TIMESTAMP,
                    chunked_at TIMESTAMP,
                    embed_model_ns TEXT,
                    upserted_at TIMESTAMP,
                    deleted_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_active
                ON documents(is_active) WHERE is_active = 1
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embed_ns
                ON documents(embed_model_ns)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source
                ON documents(source_path)
            """)

            conn.commit()

    def register_document(
        self,
        doc_id: str,
        source_path: str,
        sha256: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Register or update document in manifest

        Args:
            doc_id: Unique document identifier
            source_path: Path to source file
            sha256: SHA256 hash of parsed text
            metadata: Optional metadata (title, authors, year, venue, doi)

        Returns:
            True if new document, False if updated existing
        """
        metadata = metadata or {}

        with sqlite3.connect(self.db_path) as conn:
            # Check if exists
            cursor = conn.execute(
                "SELECT doc_id FROM documents WHERE doc_id = ?",
                (doc_id,)
            )
            exists = cursor.fetchone() is not None

            if exists:
                # Update existing
                conn.execute("""
                    UPDATE documents
                    SET source_path = ?,
                        sha256 = ?,
                        title = ?,
                        authors = ?,
                        year = ?,
                        venue = ?,
                        doi = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE doc_id = ?
                """, (
                    source_path,
                    sha256,
                    metadata.get('title'),
                    metadata.get('authors'),
                    metadata.get('year'),
                    metadata.get('venue'),
                    metadata.get('doi'),
                    doc_id
                ))
            else:
                # Insert new
                conn.execute("""
                    INSERT INTO documents (
                        doc_id, source_path, sha256,
                        title, authors, year, venue, doi
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id,
                    source_path,
                    sha256,
                    metadata.get('title'),
                    metadata.get('authors'),
                    metadata.get('year'),
                    metadata.get('venue'),
                    metadata.get('doi')
                ))

            conn.commit()
            return not exists

    def needs_reindex(self, doc_id: str, current_sha256: str) -> bool:
        """
        Check if document needs reindexing

        Args:
            doc_id: Document identifier
            current_sha256: Current SHA256 hash

        Returns:
            True if needs reindexing (new or changed)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT sha256 FROM documents WHERE doc_id = ?",
                (doc_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return True  # New document

            stored_sha256 = row[0]
            return stored_sha256 != current_sha256

    def mark_parsed(self, doc_id: str, timestamp: Optional[str] = None):
        """Mark document as parsed"""
        timestamp = timestamp or datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE documents SET parsed_at = ?, updated_at = CURRENT_TIMESTAMP WHERE doc_id = ?",
                (timestamp, doc_id)
            )
            conn.commit()

    def mark_chunked(self, doc_id: str, timestamp: Optional[str] = None):
        """Mark document as chunked"""
        timestamp = timestamp or datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE documents SET chunked_at = ?, updated_at = CURRENT_TIMESTAMP WHERE doc_id = ?",
                (timestamp, doc_id)
            )
            conn.commit()

    def mark_upserted(
        self,
        doc_id: str,
        embed_model_ns: str,
        timestamp: Optional[str] = None
    ):
        """Mark document as upserted to vector DB"""
        timestamp = timestamp or datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE documents
                SET embed_model_ns = ?,
                    upserted_at = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE doc_id = ?
            """, (embed_model_ns, timestamp, doc_id))
            conn.commit()

    def mark_deleted(self, doc_id: str, soft: bool = True):
        """
        Mark document as deleted

        Args:
            doc_id: Document identifier
            soft: If True, soft delete (set is_active=0), else hard delete
        """
        with sqlite3.connect(self.db_path) as conn:
            if soft:
                conn.execute("""
                    UPDATE documents
                    SET is_active = 0,
                        deleted_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE doc_id = ?
                """, (doc_id,))
            else:
                conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

            conn.commit()

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """Get document record"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?",
                (doc_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return DocumentRecord(**dict(row))

    def list_active_documents(self) -> List[DocumentRecord]:
        """List all active documents"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM documents WHERE is_active = 1 ORDER BY created_at"
            )
            return [DocumentRecord(**dict(row)) for row in cursor.fetchall()]

    def list_by_namespace(self, embed_model_ns: str) -> List[DocumentRecord]:
        """List documents by embedding model namespace"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM documents WHERE embed_model_ns = ? AND is_active = 1",
                (embed_model_ns,)
            )
            return [DocumentRecord(**dict(row)) for row in cursor.fetchall()]

    def list_needing_migration(
        self,
        old_namespace: str,
        new_namespace: str
    ) -> List[DocumentRecord]:
        """List documents needing namespace migration"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM documents
                WHERE embed_model_ns = ?
                  AND is_active = 1
                ORDER BY created_at
            """, (old_namespace,))
            return [DocumentRecord(**dict(row)) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """Get manifest statistics"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}

            # Total counts
            cursor = conn.execute("SELECT COUNT(*) FROM documents WHERE is_active = 1")
            stats['total_active'] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM documents WHERE is_active = 0")
            stats['total_deleted'] = cursor.fetchone()[0]

            # By namespace
            cursor = conn.execute("""
                SELECT embed_model_ns, COUNT(*)
                FROM documents
                WHERE is_active = 1
                GROUP BY embed_model_ns
            """)
            stats['by_namespace'] = {row[0]: row[1] for row in cursor.fetchall()}

            # Indexing stages
            cursor = conn.execute("""
                SELECT
                    COUNT(CASE WHEN parsed_at IS NOT NULL THEN 1 END) as parsed,
                    COUNT(CASE WHEN chunked_at IS NOT NULL THEN 1 END) as chunked,
                    COUNT(CASE WHEN upserted_at IS NOT NULL THEN 1 END) as upserted
                FROM documents
                WHERE is_active = 1
            """)
            row = cursor.fetchone()
            stats['parsed_count'] = row[0]
            stats['chunked_count'] = row[1]
            stats['upserted_count'] = row[2]

            return stats


def compute_text_sha256(text: str) -> str:
    """
    Compute SHA256 hash of text

    Args:
        text: Text content

    Returns:
        Hex digest of SHA256 hash
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def compute_file_sha256(file_path: str) -> str:
    """
    Compute SHA256 hash of file

    Args:
        file_path: Path to file

    Returns:
        Hex digest of SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# Example usage
if __name__ == "__main__":
    manifest = ManifestDB("data/manifest.db")

    # Register document
    doc_id = "vaswani2017attention"
    manifest.register_document(
        doc_id=doc_id,
        source_path="data/raw_papers/attention_is_all_you_need.pdf",
        sha256="abc123...",
        metadata={
            'title': 'Attention Is All You Need',
            'authors': 'Vaswani et al.',
            'year': 2017,
            'venue': 'NeurIPS'
        }
    )

    # Check if needs reindex
    needs_reindex = manifest.needs_reindex(doc_id, "abc123...")
    print(f"Needs reindex: {needs_reindex}")

    # Mark stages
    manifest.mark_parsed(doc_id)
    manifest.mark_chunked(doc_id)
    manifest.mark_upserted(doc_id, "bge-m3@1024@v1")

    # Get stats
    stats = manifest.get_stats()
    print(f"Stats: {stats}")
