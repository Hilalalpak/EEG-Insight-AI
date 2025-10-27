"""
Helper utilities for aggregating expert EEG label votes
"""

class LabelAggregator:

    @staticmethod
    def get_labels_for_second(start_second: int, metadata_df):
        """Finds all expert tags that coincide with the given second."""
        matching = []
        for _, row in metadata_df.iterrows():
            start = row["eeg_label_offset_seconds"]
            # In the dataset, the labels are in 10-second windows.
            end = start + 10
            if start <= start_second < end:
                matching.append({
                    "expert_consensus": row["expert_consensus"],
                    "seizure_vote": row["seizure_vote"],
                    "lpd_vote": row["lpd_vote"],
                    "gpd_vote": row["gpd_vote"],
                    "lrda_vote": row["lrda_vote"],
                    "grda_vote": row["grda_vote"],
                    "other_vote": row["other_vote"],
                    "offset": start,
                    "eeg_sub_id": row["eeg_sub_id"]})
        return matching

    @staticmethod
    def get_vote_summary(start_second, metadata_df):
        """
        Collects votes in all overlapping windows for a single second.
        """
        vote_sum = {'seizure': 0,'lpd': 0,'gpd': 0,'lrda': 0,'grda': 0,'other': 0}
        matching_count = 0

        for _, row in metadata_df.iterrows():
            start = row["eeg_label_offset_seconds"]
            end = start + 10

            if start <= start_second < end:
                matching_count += 1
                vote_sum['seizure'] += int(row['seizure_vote'])
                vote_sum['lpd'] += int(row['lpd_vote'])
                vote_sum['gpd'] += int(row['gpd_vote'])
                vote_sum['lrda'] += int(row['lrda_vote'])
                vote_sum['grda'] += int(row['grda_vote'])
                vote_sum['other'] += int(row['other_vote'])

        total_votes = sum(vote_sum.values())

        if total_votes > 0:
            probabilities = {k: v / total_votes for k, v in vote_sum.items()}
            # Find consensus (label with highest probability)
            consensus = max(probabilities.items(), key=lambda x: x[1])[0]
            confidence = probabilities[consensus]
        else:
            probabilities = {k: 0.0 for k in vote_sum.keys()}
            consensus = 'unknown'
            confidence = 0.0

        return {
            'consensus': consensus,
            'confidence': float(confidence),
            'probabilities': probabilities,
            'vote_counts': vote_sum,
            'total_votes': total_votes,
            'num_overlapping_windows': matching_count}

    @staticmethod
    def map_consensus_to_standard(consensus):
        """Standardizes label formats."""
        mapping = {
            'seizure': 'Seizure',
            'lpd': 'LPD',
            'gpd': 'GPD',
            'lrda': 'LRDA',
            'grda': 'GRDA',
            'other': 'Other',
            'unknown': 'unknown'}
        return mapping.get(consensus.lower(), consensus)