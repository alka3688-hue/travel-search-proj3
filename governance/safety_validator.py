import re
from typing import Dict, Any, List
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety.models import AnalyzeTextOptions
from src.config import Config
from guardrails.content_safety import ContentSafety

class SafetyValidator:
    """
    Validates content safety and detects adversarial attacks (jailbreaks).
    """
    def __init__(self):
        self.content_safety = ContentSafety()
        
        # Additional Patterns for prompt injection (if not covered by ContentSafety)
        self.injection_patterns = [
            r"ignore previous instructions",
            r"system overload",
            r"delete all data",
            r"you are now DAN",
            r"bypass safety protocols",
            r"you are now in developer mode",
        ]
        
        self.client = None
        if Config.AZURE_CONTENT_SAFETY_ENDPOINT and Config.AZURE_CONTENT_SAFETY_KEY:
            try:
                self.client = ContentSafetyClient(
                    endpoint=Config.AZURE_CONTENT_SAFETY_ENDPOINT,
                    credential=AzureKeyCredential(Config.AZURE_CONTENT_SAFETY_KEY)
                )
            except Exception as e:
                print(f"Warning: Failed to init Azure Content Safety: {e}")

    def validate(self, text: str, severity_threshold: str = "high") -> Dict[str, Any]:
        """
        Validates the text for safety violations.
        """
        flags = []
        is_safe = True
        
        # 1. Local Content Safety Guardrail (Keywords & Regex)
        local_result = self.content_safety.check(text)
        if not local_result['is_safe']:
            is_safe = False
            for flag in local_result['flags']:
                flags.append(f"Unsafe Keyword ({flag['category']}): {flag['keyword']}")

        # 2. Specific Injection Checks
        for pattern in self.injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                is_safe = False
                flags.append(f"Prompt Injection Detected: {pattern}")
        
        # 3. Azure Content Safety Check
        if self.client:
            try:
                request = AnalyzeTextOptions(text=text)
                response = self.client.analyze_text(request)
                
                # Check categories_analysis for high severity flags (SDK 1.0.0+)
                if response.categories_analysis:
                    for analysis in response.categories_analysis:
                        if analysis.severity > 2:
                            is_safe = False
                            flags.append(f"Azure Content Safety Violation: {analysis.category} ({analysis.severity})")
                    
            except HttpResponseError as e:
                # Suppress logging for auth errors (401, invalid credentials)
                # Content Safety is optional; don't block on auth failures
                if e.status_code not in [401, 403, 404]:
                    print(f"Azure Content Safety check failed: {e}")
            except Exception as e:
                # Silently handle any other errors (network, timeout, etc.)
                # Content Safety is optional guardrail
                pass
        
        return {
            "is_safe": is_safe,
            "flags": flags,
            "severity": "high" if not is_safe else "low"
        }