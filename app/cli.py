"""
Unified CLI for Academic RAG System
====================================

Command-line interface for interacting with the RAG system.

Commands:
    query - Ask a question
    status - Show system status
    index - Build/rebuild vector index
    evaluate - Run evaluation
    config - Show configuration

Usage:
    python -m app.cli query "What is BERT?"
    python -m app.cli query --interactive
    python -m app.cli status
    python -m app.cli evaluate

    # Or use Makefile
    make run
    make query Q="your question"
"""

import click
import sys
import json
from pathlib import Path
from typing import Optional
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import components
try:
    from rag import RAGPipeline, RAGConfig
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"⚠️ RAG module not available: {e}")

try:
    from configs import config
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False


# ============================================================================
# CLI Group
# ============================================================================

@click.group()
@click.version_option(version="2.0.0", prog_name="Academic RAG System")
def cli():
    """
    Academic RAG System - Command Line Interface

    A powerful question-answering system for academic papers.
    """
    pass


# ============================================================================
# Query Command
# ============================================================================

@cli.command()
@click.argument('question', required=False)
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
@click.option('--top-k', '-k', default=5, type=int, help='Number of sources')
@click.option('--no-rerank', is_flag=True, help='Disable reranking')
@click.option('--no-sources', is_flag=True, help='Hide sources')
@click.option('--json-output', '-j', is_flag=True, help='Output JSON format')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def query(question, interactive, top_k, no_rerank, no_sources, json_output, verbose):
    """
    Query the RAG system

    Examples:
        app cli query "What is the transformer architecture?"
        app cli query --interactive
        app cli query -k 10 "How does BERT work?"
    """
    if not RAG_AVAILABLE:
        click.echo("❌ RAG module not available. Please check installation.", err=True)
        sys.exit(1)

    # Initialize pipeline
    click.echo("🚀 Initializing RAG system...", err=True)

    try:
        rag_config = RAGConfig.from_global_config() if USE_CONFIG else RAGConfig()
        rag_config.enable_reranking = not no_rerank
        rag_config.top_k = top_k

        pipeline = RAGPipeline(rag_config)
        pipeline.initialize()

        click.echo("✅ System ready!\n", err=True)

    except Exception as e:
        click.echo(f"❌ Failed to initialize: {e}", err=True)
        sys.exit(1)

    # Interactive mode
    if interactive:
        _interactive_query_loop(pipeline, no_sources, verbose)
        return

    # Single query mode
    if not question:
        click.echo("❌ Please provide a question or use --interactive mode", err=True)
        sys.exit(1)

    _execute_query(pipeline, question, no_sources, json_output, verbose)


def _interactive_query_loop(pipeline, no_sources, verbose):
    """Interactive query loop"""
    click.echo("="*60)
    click.echo("📚 Academic RAG System - Interactive Mode")
    click.echo("="*60)
    click.echo("\nCommands:")
    click.echo("  query: <your question>")
    click.echo("  status: Show system status")
    click.echo("  help: Show help")
    click.echo("  quit: Exit")
    click.echo("")

    while True:
        try:
            user_input = click.prompt("\n> ", type=str)

            if not user_input.strip():
                continue

            # Parse command
            if user_input.lower() in ['quit', 'exit', 'q']:
                click.echo("\n👋 Goodbye!")
                break

            elif user_input.lower() in ['help', 'h', '?']:
                _show_help()

            elif user_input.lower() in ['status', 'stat']:
                _show_status(pipeline)

            elif user_input.lower().startswith('query:'):
                question = user_input[6:].strip()
                if question:
                    _execute_query(pipeline, question, no_sources, False, verbose)
                else:
                    click.echo("⚠️ Please provide a question after 'query:'")

            else:
                # Treat as direct question
                _execute_query(pipeline, user_input, no_sources, False, verbose)

        except KeyboardInterrupt:
            click.echo("\n\n👋 Goodbye!")
            break
        except EOFError:
            click.echo("\n\n👋 Goodbye!")
            break
        except Exception as e:
            click.echo(f"\n❌ Error: {e}", err=True)


def _execute_query(pipeline, question, no_sources, json_output, verbose):
    """Execute a single query"""
    if verbose:
        click.echo(f"\n{'='*60}")
        click.echo(f"📝 Query: {question}")
        click.echo(f"{'='*60}\n")

    try:
        result = pipeline.query(question, return_metadata=True)

        if json_output:
            # JSON output
            click.echo(json.dumps(result, indent=2))
        else:
            # Human-readable output
            if not verbose:
                click.echo(f"\n{'='*60}")
                click.echo(f"📝 Query: {question}")
                click.echo(f"{'='*60}\n")

            click.echo("🤖 Answer:")
            click.echo("-" * 60)
            click.echo(result.get('answer', 'No answer generated'))
            click.echo("-" * 60)

            if not no_sources and result.get('sources'):
                click.echo(f"\n📚 Sources ({len(result['sources'])}):")
                for source in result['sources']:
                    click.echo(f"\n  [{source['index']}] {source.get('title', 'Unknown')}")
                    click.echo(f"      Score: {source['score']:.3f}")
                    if verbose:
                        click.echo(f"      Content: {source['content'][:150]}...")

            if verbose and result.get('metadata'):
                click.echo(f"\n⏱️  Performance:")
                meta = result['metadata']
                click.echo(f"      Retrieval: {meta.get('retrieval_time', 0):.2f}s")
                click.echo(f"      Ranking: {meta.get('ranking_time', 0):.2f}s")
                click.echo(f"      Generation: {meta.get('generation_time', 0):.2f}s")
                click.echo(f"      Total: {meta.get('total_time', 0):.2f}s")

    except Exception as e:
        click.echo(f"\n❌ Query failed: {e}", err=True)


def _show_help():
    """Show help message"""
    click.echo("\n" + "="*60)
    click.echo("📖 Help - Available Commands")
    click.echo("="*60)
    click.echo("\n  query: <question>  - Ask a question")
    click.echo("  status             - Show system status")
    click.echo("  help               - Show this help")
    click.echo("  quit               - Exit interactive mode")
    click.echo("\n" + "="*60 + "\n")


def _show_status(pipeline):
    """Show system status"""
    try:
        status = pipeline.get_status()

        click.echo("\n" + "="*60)
        click.echo("📊 System Status")
        click.echo("="*60)

        click.echo(f"\nInitialized: {'✅' if status['initialized'] else '❌'}")
        click.echo(f"Vector Store: {'✅' if status['vector_store'] else '❌'}")
        click.echo(f"Retriever: {'✅' if status['retriever'] else '❌'}")
        click.echo(f"Ranker: {'✅' if status['ranker'] else '❌'}")
        click.echo(f"LLM Client: {'✅' if status['llm_client'] else '❌'}")

        if 'vector_store_stats' in status:
            stats = status['vector_store_stats']
            click.echo(f"\n📚 Vector Store:")
            click.echo(f"   Documents: {stats.get('total_documents', 0)}")
            click.echo(f"   Model: {stats.get('embedding_model', 'unknown')}")
            click.echo(f"   Dimension: {stats.get('embedding_dimension', 0)}")
            click.echo(f"   Device: {stats.get('device', 'unknown')}")

        click.echo("\n" + "="*60 + "\n")

    except Exception as e:
        click.echo(f"\n❌ Failed to get status: {e}", err=True)


# ============================================================================
# Status Command
# ============================================================================

@cli.command()
@click.option('--json-output', '-j', is_flag=True, help='Output JSON format')
def status(json_output):
    """Show system status"""
    if not RAG_AVAILABLE:
        click.echo("❌ RAG module not available", err=True)
        sys.exit(1)

    try:
        click.echo("🔍 Checking system status...\n", err=True)

        rag_config = RAGConfig.from_global_config() if USE_CONFIG else RAGConfig()
        pipeline = RAGPipeline(rag_config)
        pipeline.initialize()

        status_info = pipeline.get_status()

        if json_output:
            click.echo(json.dumps(status_info, indent=2))
        else:
            _show_status(pipeline)

    except Exception as e:
        click.echo(f"❌ Failed to get status: {e}", err=True)
        sys.exit(1)


# ============================================================================
# Config Command
# ============================================================================

@cli.command()
@click.option('--json-output', '-j', is_flag=True, help='Output JSON format')
def show_config(json_output):
    """Show configuration"""
    if not USE_CONFIG:
        click.echo("❌ Config module not available", err=True)
        sys.exit(1)

    try:
        from dataclasses import asdict

        config_dict = {
            'system': asdict(config.system),
            'embedding': asdict(config.embedding),
            'retrieval': asdict(config.retrieval),
            'generation': asdict(config.generation),
        }

        if json_output:
            click.echo(json.dumps(config_dict, indent=2))
        else:
            click.echo("\n" + "="*60)
            click.echo("⚙️  Configuration")
            click.echo("="*60)

            for section, values in config_dict.items():
                click.echo(f"\n[{section}]")
                for key, value in values.items():
                    click.echo(f"  {key}: {value}")

            click.echo("\n" + "="*60 + "\n")

    except Exception as e:
        click.echo(f"❌ Failed to load config: {e}", err=True)
        sys.exit(1)


# ============================================================================
# Evaluate Command
# ============================================================================

@cli.command()
@click.option('--dataset', '-d', type=click.Path(exists=True), help='Path to evaluation dataset')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def evaluate(dataset, output, verbose):
    """Run system evaluation"""
    click.echo("⚠️ Evaluation functionality to be implemented", err=True)
    click.echo("Please use: python evaluate_rag_system.py", err=True)


# ============================================================================
# Index Command
# ============================================================================

@cli.command()
@click.option('--pdf-dir', type=click.Path(exists=True), default='data/raw_papers', help='PDF directory')
@click.option('--rebuild', is_flag=True, help='Rebuild entire index')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def index(pdf_dir, rebuild, verbose):
    """Build/rebuild vector index"""
    click.echo("⚠️ Indexing functionality to be implemented", err=True)
    click.echo("Please use: python process_pdf_fulltext.py", err=True)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    cli()
