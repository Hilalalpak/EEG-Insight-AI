# cli.py
import typer
import os
import sys
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from infrastructure.conf.config_loader import ConfigLoader
    from infrastructure.conf.interfaces import (DBConfigInterface, LLMConfigInterface,
                                                PipelineConfigInterface, LoggingConfigInterface)
    from src.rag.core.logging_config import LoggingConfig

    from src.rag.ingestion.document_pipeline.document_pipeline import DocumentPipeline
    from src.rag.ingestion.transcript_pipeline.transcript_pipeline import TranscriptPipeline
    from src.rag.ingestion.signal_pipeline.signal_pipeline import SignalPipeline

except ImportError as e:
    print(f"Import error - check PYTHONPATH")
    print(f"PYTHONPATH: {sys.path}")
    print(f"Error: {e}")
    sys.exit(1)

CONFIG_PATHS = {
    "base": os.path.join(PROJECT_ROOT, "infrastructure/conf/base.yml"),
    "rag_strategy": os.path.join(PROJECT_ROOT, "infrastructure/conf/rag_strategy.yaml"),
    "models": os.path.join(PROJECT_ROOT, "infrastructure/conf/llm/models.yml"),
    "llm_env": os.path.join(PROJECT_ROOT, "infrastructure/conf/llm/env_dev.yaml"),
    "signal": os.path.join(PROJECT_ROOT, "infrastructure/conf/pipeline/signal.yml"),
    "data_sources": os.path.join(PROJECT_ROOT, "infrastructure/conf/pipeline/data_sources.yml")}

try:
    loader = ConfigLoader(config_paths=CONFIG_PATHS)
    db_config: DBConfigInterface = loader.get_db_config()
    llm_config: LLMConfigInterface = loader.get_llm_config()
    pipeline_config: PipelineConfigInterface = loader.get_pipeline_config()
    log_config: LoggingConfigInterface = loader.get_logging_config()

    LoggingConfig.setup_logging(log_config)
    logger = logging.getLogger(__name__)
    logger.info("CLI initialized ")

except Exception as e:
    print(f"Config load failed: {e}")
    sys.exit(1)

app = typer.Typer(help="EEG-RAG ingestion pipeline CLI")

@app.command()
def ingest_document():
    logger.info("Document pipeline starting...")
    try:
        indexer = DocumentPipeline(
            db_config=db_config,
            llm_config=llm_config,
            pipeline_config=pipeline_config,
            logger=logger)

        doc_name = pipeline_config.get_acns_document_name()
        filepath = os.path.join(PROJECT_ROOT, "data/documents", doc_name)
        logger.info(f"Target PDF (from config): {filepath}")
        indexer.index_pdf(filepath=filepath, use_semantic_chunking=True)
        logger.info("Document processing complete.")

    except Exception as e:
        logger.critical(f"Document pipeline failed: {e}", exc_info=True)
        sys.exit(1)

@app.command()
def ingest_signal():
    logger.info("Processing signal data from S3...")

    try:
        processor = SignalPipeline(
            db_config=db_config,
            llm_config=llm_config,
            pipeline_config=pipeline_config,
            logger=logger)

        processor.run_pipeline()
        logger.info("Signal processing complete.")

    except Exception as e:
        logger.critical(f"Signal pipeline failed: {e}", exc_info=True)
        sys.exit(1)

@app.command()
def ingest_transcript():
    logger.info("Processing transcripts from video list...")
    try:
        ingestor = TranscriptPipeline(
            db_config=db_config,
            llm_config=llm_config,
            pipeline_config=pipeline_config,
            logger=logger)

        ingestor.run_pipeline()
        logger.info("Transcript processing complete.")

    except Exception as e:
        logger.critical(f"Transcript pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    app()