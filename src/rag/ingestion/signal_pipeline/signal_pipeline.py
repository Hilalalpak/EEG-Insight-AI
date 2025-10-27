"""
Main EEG processing pipeline. Coordinates S3 loading, summarization and indexing to ChromaDB.
"""
import logging
from infrastructure.conf.interfaces import (DBConfigInterface, LLMConfigInterface, PipelineConfigInterface)

from src.rag.ingestion.signal_pipeline.s3_utils import S3Utils
from src.rag.ingestion.signal_pipeline.summarizer import SegmentSummarizer
from src.rag.ingestion.signal_pipeline.features import SignalFeatures

from src.rag.core.embedder import TextEmbedder
from src.rag.core.retriever import ChromaDB

class SignalPipeline:

    def __init__(self,
                 db_config: DBConfigInterface,
                 llm_config: LLMConfigInterface,
                 pipeline_config: PipelineConfigInterface,
                 logger: logging.Logger):

        self.logger = logger

        self.collection_name = db_config.get_collection_name("signal")
        self.sampling_rate = pipeline_config.get_signal_sampling_rate()

        s3_bucket = db_config.get_s3_bucket_name()
        s3_endpoint = db_config.get_s3_container_endpoint()

        fs = pipeline_config.get_signal_sampling_rate()
        freq_range = pipeline_config.get_signal_freq_range()
        notch_freq = pipeline_config.get_signal_notch_freq()

        embed_model = llm_config.get_embedding_model_name()

        chroma_host = db_config.get_chroma_host()
        chroma_port = db_config.get_chroma_port()

        self.logger.info(f"SignalPipeline initialized. Collection: '{self.collection_name}'")

        signal_features_helper = SignalFeatures(self.logger, fs=fs, freq_range=freq_range, notch_freq=notch_freq)
        self.data_loader = S3Utils(self.logger, bucket_name=s3_bucket, endpoint_url=s3_endpoint)
        self.summarizer = SegmentSummarizer(self.logger, signal_features_instance=signal_features_helper)
        self.chroma_db = ChromaDB(self.logger, self.collection_name, host=chroma_host, port=chroma_port)
        self.embedder = TextEmbedder(self.logger, hf_model_path=embed_model)

        self.logger.info("Services ready: S3, Chroma, Embedder, Features")

    def _index_eeg_file(self, eeg_id: str):
        """Process a single EEG recording."""
        self.logger.info(f"Processing EEG ID: {eeg_id} ---")

        # Check if the first segment is already indexed to skip the whole file
        first_object_key = f"{eeg_id}/{eeg_id}.parquet_sec_0"

        try:
            if self.chroma_db.is_doc_indexed(first_object_key):
                self.logger.info(f"EEG {eeg_id} already indexed, skipping.")
                return
        except Exception as e:
            self.logger.error(f"DB check failed for {eeg_id}. Skipping this ID. Error: {e}", exc_info=True)
            return

        try:
            metadata_df = self.data_loader.load_metadata(eeg_id)
            full_df = self.data_loader.load_eeg_data(eeg_id)

            if metadata_df is None or full_df is None:
                self.logger.error(f"S3 data load failed for {eeg_id} (metadata or parquet). Skipping.")
                return

            self.logger.info(f"Loaded data for {eeg_id}. Found {len(metadata_df)} labels in metadata.")
            self._index_segments(eeg_id, full_df, metadata_df)

        except Exception as e:
            self.logger.error(f"An error occurred while processing {eeg_id}: {e}", exc_info=True)

    def _index_segments(self, eeg_id: str, full_df, metadata_df):
        """Process all segments of an EEG recording"""
        segment_size = self.sampling_rate
        total_secs = len(full_df) // segment_size

        if total_secs == 0:
            self.logger.warning(f"No full segments for EEG {eeg_id}. Skipping.")
            return

        self.logger.info(f"Processing {total_secs} segments for {eeg_id}")

        labeled_count = 0
        unlabeled_count = 0

        for i in range(total_secs):
            try:
                start_row = i * segment_size
                end_row = (i + 1) * segment_size
                segment_df = full_df.iloc[start_row:end_row]

                object_key = f"{eeg_id}/{eeg_id}.parquet_sec_{i}"

                summary_txt, chroma_metadata = self.summarizer.build_summary(segment_df, eeg_id, i, metadata_df)

                if summary_txt is None or chroma_metadata is None:
                    self.logger.warning(f"Segment {i} for {eeg_id} failed")
                    continue

                if chroma_metadata["expert_consensus"] == "unknown":
                    unlabeled_count += 1
                else:
                    labeled_count += 1

                embedding = self.embedder.embed(summary_txt)
                self.chroma_db.index_segment(summary_txt, embedding, chroma_metadata, object_key)

            except Exception as e:
                self.logger.error(f"Segment failed: {eeg_id} sec {i}: {e}")
                continue

        self.logger.info(f"Done {eeg_id}: {total_secs} segments")
        self.logger.info(f"Stats: labeled={labeled_count}, unlabeled={unlabeled_count}")

    def run_pipeline(self):
        self.logger.info("Starting EEG processing...")
        try:
            eeg_recordings = self.data_loader.list_eeg_recordings()
            if not eeg_recordings:
                self.logger.warning("No EEG recordings in S3. Pipeline finished.")
                return

            self.logger.info(f"Found {len(eeg_recordings)} EEG IDs in S3.")
            for prefix in eeg_recordings:
                eeg_id = prefix["Prefix"].replace("/", "")
                if eeg_id:
                    self._index_eeg_file(eeg_id)
            self.logger.info("Finished processing all EEG data.")

        except Exception as e:
            self.logger.critical(f"Pipeline error: {e}", exc_info=True)

