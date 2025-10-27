from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from .interfaces import (DBConfigInterface, LLMConfigInterface, PipelineConfigInterface,RAGCoreConfigInterface, LoggingConfigInterface, MiscConfigInterface)

class DBConfigModel(BaseModel, DBConfigInterface):
    """DB and Storage settings."""
    chromadb_host: str = Field(..., alias="chromadb__host")
    chromadb_port: int = Field(..., alias="chromadb__port")
    collections: Dict[str, str] = Field(..., alias="chromadb__collections")
    s3_bucket_name: str = Field(..., alias="s3__bucket_name")
    s3_container_endpoint: str = Field(..., alias="s3__container_endpoint")
    s3_local_endpoint: Optional[str] = Field(None, alias="s3__local_endpoint")

    def get_chroma_host(self) -> str: return self.chromadb_host
    def get_chroma_port(self) -> int: return self.chromadb_port
    def get_collection_name(self, source: str) -> str:
        return self.collections.get(source, f"{source}_default_collection")
    def get_s3_container_endpoint(self) -> str: return self.s3_container_endpoint
    def get_s3_local_endpoint(self) -> Optional[str]: return self.s3_local_endpoint
    def get_s3_bucket_name(self) -> str: return self.s3_bucket_name
class LLMConfigModel(BaseModel, LLMConfigInterface):
    """LLM service and model settings."""
    ollama_endpoint: str = Field(..., alias="ollama__endpoint")
    llm_model_name: str = Field(..., alias="ollama__model")
    llm_options: Dict[str, Any] = Field(default={}, alias="ollama__options")
    embedding_model_name: str = Field(..., alias="models__embedding")
    reranker_model_name: str = Field(..., alias="models__reranker")

    def get_ollama_endpoint(self) -> str: return self.ollama_endpoint
    def get_llm_model_name(self) -> str: return self.llm_model_name
    def get_llm_options(self) -> Dict[str, Any]: return self.llm_options
    def get_embedding_model_name(self) -> str: return self.embedding_model_name
    def get_reranker_model_name(self) -> str: return self.reranker_model_name
class PipelineConfigModel(BaseModel, PipelineConfigInterface):
    """Data processing pipeline settings."""
    signal_sampling_rate: int = Field(..., alias="signal__sampling_rate")
    signal_freq_range: List[float] = Field(..., alias="signal__freq_range")
    signal_notch_freq: Optional[int] = Field(None, alias="signal__notch_freq")
    document_chunk_params: Dict[str, int] = Field(..., alias="chunking__document")
    transcript_chunk_params: Dict[str, int] = Field(..., alias="chunking__transcript")
    youtube_videos: List[str] = Field(default=[], alias="data_sources__youtube_videos")
    acns_document_name: str = Field(..., alias="data_sources__acns_document")

    def get_signal_sampling_rate(self) -> int: return self.signal_sampling_rate
    def get_signal_freq_range(self) -> List[float]: return self.signal_freq_range
    def get_signal_notch_freq(self) -> Optional[int]: return self.signal_notch_freq
    def get_document_chunk_params(self) -> Dict[str, int]: return self.document_chunk_params
    def get_transcript_chunk_params(self) -> Dict[str, int]: return self.transcript_chunk_params
    def get_youtube_videos(self) -> List[str]: return self.youtube_videos
    def get_acns_document_name(self) -> str: return self.acns_document_name
class RAGCoreConfigModel(BaseModel, RAGCoreConfigInterface):
    """RAG strategy settings."""
    n_search: int = Field(..., alias="rag__n_search")
    n_final_signal: int = Field(..., alias="rag__n_final_signal")
    n_final_document: int = Field(..., alias="rag__n_final_document")
    n_final_transcript: int = Field(..., alias="rag__n_final_transcript")
    rrf_k: int = Field(default=60, alias="rag__rrf_k")

    def get_n_search(self) -> int: return self.n_search
    def get_n_final_signal(self) -> int: return self.n_final_signal
    def get_n_final_document(self) -> int: return self.n_final_document
    def get_n_final_transcript(self) -> int: return self.n_final_transcript
    def get_rrf_k(self) -> int: return self.rrf_k
class LoggingConfigModel(BaseModel, LoggingConfigInterface):
    """Logging settings."""
    level: str = Field(default="INFO", alias="logging__level")
    # Format/datefmt are optional as they might be hardcoded in logging setup
    format_str: Optional[str] = Field(None, alias="logging__format")
    datefmt: Optional[str] = Field(None, alias="logging__datefmt")

    def get_log_level(self) -> str: return self.level
    def get_log_format(self) -> Optional[str]: return self.format_str
    def get_log_datefmt(self) -> Optional[str]: return self.datefmt
class MiscConfigModel(BaseModel, MiscConfigInterface):
    """Misc settings."""
    competition_name: Optional[str] = Field(None, alias="kaggle__competition_name")

    def get_kaggle_competition_name(self) -> Optional[str]: return self.competition_name
