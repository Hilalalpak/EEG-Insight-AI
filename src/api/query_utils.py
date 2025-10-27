"""
Query parsing and type detection utilities
"""
import re

def find_patient_id(query: str) -> str | None:
    """Extract patient ID from query string."""
    patterns = [
        r'patient\s+(?:id\s+)?(\d+)',
        r'patient_id[:\s]+(\d+)',
        r'\b(\d{10})\b'] # TODO: is this 10-digit always valid?

    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    return None
def find_eeg_id(query: str) -> str | None:
    """Extract EEG ID from query string."""
    patterns = [
        r'eeg\s+(?:id\s+)?(\d+)',
        r'eeg[:\s]+(\d+)']
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    return None
def get_query_type(query: str) -> str:
    """
    Tries to guess the query type based on keywords.
    Returns: 'reasoning', 'definition', 'patient_data', 'general'
    """
    query_lower = query.lower()

    reasoning_keywords = ['how', 'why', 'explain', 'difference', 'compare', 'detect', 'analyze', 'step', 'process', 'method']
    definition_keywords = ['what is', 'define', 'definition', 'means']
    patient_keywords = ['patient', 'eeg', 'show', 'find', 'segment']

    if any(kw in query_lower for kw in reasoning_keywords):
        return 'reasoning'
    elif any(kw in query_lower for kw in definition_keywords):
        return 'definition'
    elif any(kw in query_lower for kw in patient_keywords):
        return 'patient_data'
    else:
        return 'general'