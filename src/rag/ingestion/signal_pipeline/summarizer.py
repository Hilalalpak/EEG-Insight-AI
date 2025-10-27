"""
Creates summary text and metadata for 1-sec EEG segments.
Combines signal features and expert votes
"""
from src.rag.ingestion.signal_pipeline.features import SignalFeatures
from src.rag.ingestion.signal_pipeline.labels import LabelAggregator
import logging
from typing import Any

class SegmentSummarizer:
    def __init__(self, logger: logging.Logger, signal_features_instance: SignalFeatures):

        self.logger = logger
        self.signal_features = signal_features_instance
        self.label_agg = LabelAggregator()

    def _create_summary_text(self, features: dict, label_dist: dict, eeg_id: str, start_second: int, metadata_df) -> str:
        """Creates a human-readable summary of the segment's characteristics."""
        consensus = label_dist['consensus']
        confidence = label_dist['confidence']
        num_windows = label_dist['num_overlapping_windows']
        total_votes = label_dist['total_votes']
        probs = label_dist['probabilities']

        # Basic identification
        summary = f"EEG segment from patient {eeg_id} at second {start_second}. "

        # Label information with clinical context
        if num_windows == 0 or consensus == 'unknown':
            summary += (
                f"No expert annotations available - unlabeled baseline or artifact. ")
        else:
            if confidence >= 0.85:
                certainty = "high expert consensus"
                clinical_note = "indicating a clear, well-defined pattern"
            elif confidence >= 0.65:
                certainty = "moderate agreement"
                clinical_note = "suggesting typical presentation with some variability"
            else:
                certainty = "mixed expert opinions"
                clinical_note = "indicating an edge case or transitional pattern"

            # Main classification
            summary += (
                f"Classified as {consensus.upper()} with {certainty} "
                f"({confidence:.0%} agreement from {total_votes} expert votes), "
                f"{clinical_note}. ")

            # If mixed, explain the alternatives
            if confidence < 0.7:
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_probs) >= 2 and sorted_probs[1][1] > 0.15:
                    secondary = sorted_probs[1]
                    summary += (
                        f"Alternative interpretation as {secondary[0].upper()} "
                        f"({secondary[1]:.0%}) suggests this segment exhibits "
                        f"characteristics bridging both patterns, which is clinically "
                        f"significant for understanding seizure evolution or "
                        f"interictal-ictal transitions. ")

            # Multiple window perspective
            if num_windows > 1:
                summary += (
                    f"This assessment is based on {num_windows} overlapping "
                    f"50-second expert review windows, providing multiple "
                    f"temporal perspectives on the same brain activity. ")

        # Signal characteristics
        power = features['mean_power']
        snr = features['mean_snr']
        sef = features['mean_sef']

        summary += f"Signal analysis reveals: "

        # Power interpretation
        # TODO: These power thresholds (100, 20, 5) are more on log-scale
        if power > 100:
            power_note = "very high amplitude activity"
        elif power > 20:
            power_note = "elevated amplitude"
        elif power > 5:
            power_note = "moderate amplitude"
        else:
            power_note = "low amplitude"

        summary += f"mean power of {power:.2f} indicating {power_note}, "

        # SNR interpretation
        if snr > 10:
            snr_note = "excellent signal quality with minimal artifact"
        elif snr > 5:
            snr_note = "good signal quality"
        elif snr > 2:
            snr_note = "acceptable quality with some noise"
        else:
            snr_note = "noisy signal suggesting possible artifact or low voltage activity"

        summary += f"signal-to-noise ratio of {snr:.2f} showing {snr_note}, "

        # SEF interpretation
        # Note: SEF (Spectral Edge Freq), which is 95% of the power
        # Indicates that it is collected below the frequency. Low SEF slow activity (Delta),
        # high SEF means fast activity (Beta/Gamma).
        if sef > 25:
            sef_note = "fast frequency dominant activity, typical of muscle artifact or high-frequency seizure components"
        elif sef > 15:
            sef_note = "beta-dominant activity suggesting arousal or ictal patterns"
        elif sef > 8:
            sef_note = "alpha-range activity consistent with relaxed wakefulness or posterior rhythms"
        elif sef > 4:
            sef_note = "theta-dominant activity seen in drowsiness or temporal lobe pathology"
        else:
            sef_note = "delta-dominant activity characteristic of deep sleep, encephalopathy, or ictal patterns"

        summary += (
            f"and spectral edge frequency at {sef:.2f} Hz representing "
            f"{sef_note}. ")

        # Pattern-Specific clinical insights
        if consensus == 'Seizure' or (probs.get('seizure', 0) > 0.3):
            summary += (
                "Seizure activity represents abnormal synchronized neuronal "
                "discharges requiring immediate clinical attention. ")
        elif consensus == 'LPD':
            summary += (
                "Lateralized periodic discharges indicate focal cortical irritability, "
                "often associated with acute structural lesions or increased seizure risk. ")
        elif consensus == 'GPD':
            summary += (
                "Generalized periodic discharges suggest diffuse cortical dysfunction, "
                "commonly seen in anoxic injury, encephalopathy, or status epilepticus. ")
        elif consensus == 'LRDA':
            summary += (
                "Lateralized rhythmic delta activity indicates focal hemispheric "
                "dysfunction with potential for seizure evolution. ")
        elif consensus == 'GRDA':
            summary += (
                "Generalized rhythmic delta activity represents diffuse cerebral "
                "dysfunction, seen in encephalopathy or non-convulsive status epilepticus. ")

        # Searchable keywords
        keywords = self._create_keywords(consensus, confidence, power, snr, sef)
        summary += f"Keywords: {', '.join(keywords)}."

        return summary

    def _create_keywords(self, consensus, confidence, power, snr, sef):
        """Generate searchable keywords for semantic retrieval"""
        keywords = []

        # Pattern keywords
        if consensus != 'unknown':
            keywords.append(consensus.lower())
            if consensus == 'Seizure':
                keywords.extend(['ictal', 'epileptic', 'convulsive'])
            elif 'PD' in consensus:
                keywords.extend(['periodic', 'discharges'])
            elif 'RDA' in consensus:
                keywords.extend(['rhythmic', 'delta'])

        # Confidence keywords
        if confidence >= 0.8:
            keywords.append('high-confidence')
        elif confidence < 0.6:
            keywords.extend(['edge-case', 'mixed-pattern', 'uncertain'])

        # Signal keywords
        if power > 20:
            keywords.append('high-amplitude')
        if snr > 5:
            keywords.append('clean-signal')
        if sef > 20:
            keywords.append('fast-activity')
        elif sef < 5:
            keywords.append('slow-activity')

        # TODO: Add more complex clinical terms
        return keywords

    def _create_metadata(self, features: dict, label_dist: dict, eeg_id: str, start_second: int, metadata_df) -> dict[str, Any]:
        """Creates a flat dictionary of metadata for vector store filtering."""
        consensus = LabelAggregator.map_consensus_to_standard(label_dist['consensus'])

        # Find which expert review windows include this second
        context_windows = []
        for _, row in metadata_df.iterrows():
            offset = row["eeg_label_offset_seconds"]
            if offset <= start_second < offset + 10:
                context_windows.append({
                    'sub_id': int(row['eeg_sub_id']),
                    'offset': int(offset),
                    'range_start': int(offset),
                    'range_end': int(offset + 50)})

        metadata = {
            # Identifiers
            "eeg_id": str(eeg_id),
            "start_second": int(start_second),

            # Label info
            "expert_consensus": consensus,
            "confidence": float(label_dist['confidence']),
            "is_high_confidence": bool(label_dist['confidence'] >= 0.8),
            "is_mixed_pattern": bool(label_dist['confidence'] < 0.6),
            "is_edge_case": bool(0.4 < label_dist['confidence'] < 0.6),
            "num_overlapping_windows": int(label_dist['num_overlapping_windows']),
            "total_votes": int(label_dist['total_votes']),

            # Probability distribution
            "prob_seizure": float(label_dist['probabilities']['seizure']),
            "prob_lpd": float(label_dist['probabilities']['lpd']),
            "prob_gpd": float(label_dist['probabilities']['gpd']),
            "prob_lrda": float(label_dist['probabilities']['lrda']),
            "prob_grda": float(label_dist['probabilities']['grda']),
            "prob_other": float(label_dist['probabilities']['other']),

            # Vote counts
            "total_seizure_votes": int(label_dist['vote_counts']['seizure']),
            "total_lpd_votes": int(label_dist['vote_counts']['lpd']),
            "total_gpd_votes": int(label_dist['vote_counts']['gpd']),
            "total_lrda_votes": int(label_dist['vote_counts']['lrda']),
            "total_grda_votes": int(label_dist['vote_counts']['grda']),
            "total_other_votes": int(label_dist['vote_counts']['other']),

            # Signal feat.
            "mean_power": float(features["mean_power"]),
            "mean_snr": float(features["mean_snr"]),
            "mean_sef": float(features["mean_sef"]),

            # Signal cat.
            "has_high_amplitude": bool(features["mean_power"] > 20),
            "has_clean_signal": bool(features["mean_snr"] > 5),
            "has_fast_activity": bool(features["mean_sef"] > 20),
            "has_slow_activity": bool(features["mean_sef"] < 5),

            # Context info
            "num_context_windows": len(context_windows),
            "context_window_ids": ",".join([str(w['sub_id']) for w in context_windows]) if context_windows else "",}

        return metadata

    def build_summary(self, df, eeg_id: str, start_second: int, metadata_df) -> tuple[str | None, dict | None]:

        if df.empty:
            self.logger.warning(f"Empty dataframe for {eeg_id} at second {start_second}.")
            return None, None

        # Extract signal features
        features = self.signal_features.get_eeg_features(df)

        # Get aggregated label distribution
        label_dist = self.label_agg.get_vote_summary(start_second, metadata_df)

        # Generate text for semantic search
        rich_text = self._create_summary_text(features, label_dist, eeg_id, start_second, metadata_df)

        # Metadata for filtering
        metadata = self._create_metadata(features, label_dist, eeg_id, start_second, metadata_df)

        return rich_text, metadata