"""
PII Detection using regex patterns and NER
"""
import re
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class PIIDetector:
    """Detects Personally Identifiable Information"""
    
    def __init__(self):
        # Regex patterns for common PII
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'passport': r'\b[A-Z]{1,2}[0-9]{6,9}\b',
            'zip_code': r'\b\d{5}(?:-\d{4})?\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        }
    
    def detect(self, text: str) -> Dict:
        """Detect PII in text"""
        entities = []
        
        for entity_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'type': entity_type,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return {
            'has_pii': len(entities) > 0,
            'entities': entities,
            'count': len(entities)
        }
    
    def redact(self, text: str) -> str:
        """Redact PII from text"""
        redacted_text = text
        
        for entity_type, pattern in self.patterns.items():
            if entity_type == 'email':
                redacted_text = re.sub(pattern, '[EMAIL_REDACTED]', redacted_text)
            elif entity_type == 'phone':
                redacted_text = re.sub(pattern, '[PHONE_REDACTED]', redacted_text)
            elif entity_type == 'passport':
                redacted_text = re.sub(pattern, '[PASSPORT_REDACTED]', redacted_text)
            elif entity_type == 'credit_card':
                redacted_text = re.sub(pattern, '[CARD_REDACTED]', redacted_text)
            else:
                redacted_text = re.sub(pattern, f'[{entity_type.upper()}_REDACTED]', redacted_text)
        
        return redacted_text