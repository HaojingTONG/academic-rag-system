"""
Indexer CLI Entry Point
========================

Command-line interface for incremental indexer.

Usage:
    python -m indexer refresh [--force]
    python -m indexer reindex
    python -m indexer stats
    python -m indexer delete <doc_id>
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer import IncrementalIndexer


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Incremental Indexer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Refresh command
    refresh_parser = subparsers.add_parser(
        'refresh',
        help='Incremental refresh (process new/modified documents)'
    )
    refresh_parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing of all documents'
    )

    # Reindex command
    reindex_parser = subparsers.add_parser(
        'reindex',
        help='Full reindex (force reprocess all documents)'
    )
    reindex_parser.add_argument(
        '--namespace',
        type=str,
        help='Optional: new namespace for migration'
    )

    # Stats command
    stats_parser = subparsers.add_parser(
        'stats',
        help='Show indexing statistics'
    )

    # Delete command
    delete_parser = subparsers.add_parser(
        'delete',
        help='Delete a document from index'
    )
    delete_parser.add_argument(
        'doc_id',
        type=str,
        help='Document ID to delete'
    )
    delete_parser.add_argument(
        '--hard',
        action='store_true',
        help='Hard delete (vs soft delete)'
    )

    args = parser.parse_args()

    # Default to refresh if no command
    if not args.command:
        args.command = 'refresh'
        args.force = False

    # Initialize indexer
    try:
        indexer = IncrementalIndexer()
    except Exception as e:
        print(f"❌ Failed to initialize indexer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Execute command
    try:
        if args.command == 'refresh':
            indexer.refresh(force=args.force)

        elif args.command == 'reindex':
            indexer.reindex(namespace=getattr(args, 'namespace', None))

        elif args.command == 'stats':
            indexer._print_stats()

        elif args.command == 'delete':
            soft = not args.hard
            indexer.delete_document(args.doc_id, soft=soft)

        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n❌ Command failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
