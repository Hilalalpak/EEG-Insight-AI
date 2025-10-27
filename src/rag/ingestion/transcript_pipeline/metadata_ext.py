
import re
from typing import Dict, List

class MetadataExtractor:

    def __init__(self):

        # These patterns, talking about ACNS terminology
        # to catch the jargon used by experts
        self.TOPIC_PATTERNS = {
        'LPD': r'\blpd\b|lateralized periodic',
        'LRDA': r'\blrda\b|lateralized rhythmic',
        'GPD': r'\bgpd\b|generalized periodic',
        'GRDA': r'\bgrda\b|generalized rhythmic',
        'Seizure': r'\bseizure\b|ictal|electrographic',
        'BIRDS': r'\bbird[s]?\b|brief.*rhythmic',
        'IIC': r'\biic\b|continuum',
        'Plus_Modifiers': r'plus [frs]|\+[frs]',
        'Evolution': r'evolv|diagonal',
        'Frequency': r'frequency|hertz|\bhz\b',
        'Main_Terms': r'main term',
        'Modifiers': r'modifier',
        'Breaks': r'break between'}

    def _detect_reasoning(self, text: str) -> dict[str, bool]:
        """Detects clinical reasoning patterns in a text chunk."""
        text_lower = text.lower()
        # Regex patterns to catch the expert's thought process (Q&A, steps)
        return {
            'has_step_by_step': bool(re.search(r'step \d+|first.*then|next', text_lower)),
            'has_question': bool(re.search(r'(is there|are there|how many|what|why)', text_lower)),
            'has_example': 'example' in text_lower or 'so here' in text.lower(),
            'has_definition': bool(re.search(r'definition|means|defined as', text_lower)),
            'has_frequency_logic': bool(re.search(r'count.*second|hertz|hz', text_lower)),
            'has_risk': bool(re.search(r'\d+%|percent|risk|association', text_lower))}

    def _extract_topics(self, text: str) -> list[str]:
        """Extracts ACNS topics from a text chunk."""
        topics = []
        text_lower = text.lower()
        for topic, pattern in self.TOPIC_PATTERNS.items():
            if re.search(pattern, text_lower):
                topics.append(topic)
        return topics if topics else ['General']

    def extract_all(self, text: str) -> Dict:
        reasoning_meta = self._detect_reasoning(text)
        topics_list = self._extract_topics(text)

        all_metadata = {"topics": ",".join(topics_list), **reasoning_meta}
        return all_metadata