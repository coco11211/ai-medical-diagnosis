#!/usr/bin/env python
"""
Main entry point for AI Medical Diagnosis System.
Provides CLI interface for common operations.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger
from loguru import logger


def start_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server."""
    import uvicorn
    from src.api.main import app

    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload
    )


def run_tests(verbose: bool = False):
    """Run the test suite."""
    import pytest

    args = ["-v"] if verbose else []
    exit_code = pytest.main(args)
    sys.exit(exit_code)


def create_sample_data():
    """Create sample medical knowledge graph."""
    from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder

    logger.info("Creating sample medical knowledge graph...")
    builder = KnowledgeGraphBuilder()
    builder.create_sample_knowledge_graph()
    builder.save_graph("data/sample_knowledge_graph.gexf")
    logger.info("Sample knowledge graph saved to data/sample_knowledge_graph.gexf")


def check_system():
    """Check system configuration and dependencies."""
    import torch

    logger.info("=== System Check ===")

    # Python version
    logger.info(f"Python version: {sys.version}")

    # PyTorch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Check imports
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
    except ImportError:
        logger.warning("Transformers not installed")

    try:
        import fastapi
        logger.info(f"FastAPI version: {fastapi.__version__}")
    except ImportError:
        logger.warning("FastAPI not installed")

    try:
        import shap
        logger.info(f"SHAP version: {shap.__version__}")
    except ImportError:
        logger.warning("SHAP not installed")

    logger.info("=== System Check Complete ===")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Medical Diagnosis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py server                    # Start API server
  python main.py server --port 8080        # Start on custom port
  python main.py test                      # Run tests
  python main.py test --verbose            # Run tests with verbose output
  python main.py sample-data               # Create sample knowledge graph
  python main.py check                     # Check system configuration
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    server_parser.add_argument("--port", type=int, default=8000, help="Server port")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Sample data command
    subparsers.add_parser("sample-data", help="Create sample knowledge graph")

    # Check command
    subparsers.add_parser("check", help="Check system configuration")

    args = parser.parse_args()

    # Setup logger
    setup_logger()

    # Execute command
    if args.command == "server":
        start_api_server(args.host, args.port, args.reload)
    elif args.command == "test":
        run_tests(args.verbose)
    elif args.command == "sample-data":
        create_sample_data()
    elif args.command == "check":
        check_system()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
