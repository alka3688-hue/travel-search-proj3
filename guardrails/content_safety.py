"""
Combination of Manual Content Safety Checker
Azure Content Safety
- https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview
- https://learn.microsoft.com/en-us/python/api/overview/azure/ai-contentsafety-readme?view=azure-python
- https://github.com/Azure-Samples/AzureAIContentSafety

"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class ContentSafety:
    """Checks content for safety violations"""
    
    def __init__(self):
        # Keywords for different violation categories
        self.unsafe_patterns = {
            'violence': ['kill', 'murder', 'attack', 'harm', 'hurt'],
            'hate_speech': ['hate', 'discrimination', 'racist', 'sexist'],
            'profanity': ['fuck', 'shit', 'damn', 'hell'],
            'personal_attack': ['stupid', 'idiot', 'moron', 'dumb']
        }
        
        # Travel-specific unsafe patterns
        self.travel_red_flags = [
            'fraud', 'fake booking', 'scam', 'cheat',
            'illegal travel', 'smuggle'
        ]
    
    def check(self, text: str) -> Dict:
        """Check text for safety violations"""
        text_lower = text.lower()
        flags = []
        
        # Check general unsafe patterns
        for category, keywords in self.unsafe_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    flags.append({
                        'category': category,
                        'keyword': keyword,
                        'severity': 'medium'
                    })
        
        # Check travel-specific red flags
        for red_flag in self.travel_red_flags:
            if red_flag in text_lower:
                flags.append({
                    'category': 'travel_fraud',
                    'keyword': red_flag,
                    'severity': 'high'
                })
        
        return {
            'is_safe': len(flags) == 0,
            'flags': flags,
            'severity': max([f['severity'] for f in flags], default='none')
        }
    
    def get_safety_score(self, text: str) -> float:
        """Calculate safety score (0-1)"""
        result = self.check(text)
        
        if result['is_safe']:
            return 1.0
        
        # Reduce score based on violations
        penalty = 0.2 * len(result['flags'])
        return max(0.0, 1.0 - penalty)