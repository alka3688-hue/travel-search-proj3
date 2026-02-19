import re
from typing import Dict, Any, List
from guardrails.pii_detector import PIIDetector

class ComplianceChecker:
    """
    Ensures content meets regulatory and organizational compliance requirements.
    Detects PII and checks for compliance with standards like GDPR, HIPAA, etc.
    """
    def __init__(self):
        self.pii_detector = PIIDetector()

    def check_compliance(self, text: str, compliance_standards: List[str] = None, industry: str = "travel") -> Dict[str, Any]:
        """
        Checks the text for compliance violations.
        """
        compliance_standards = compliance_standards or []
        violations = []
        is_compliant = True
        
        # Check for PII using Guardrail PII Detector
        pii_result = self.pii_detector.detect(text)
        
        detected_pii = []
        if pii_result['has_pii']:
            for entity in pii_result['entities']:
                detected_pii.append(f"{entity['type']}: {entity['value']}")
        
        if detected_pii:
            violations.append(f"PII Detected: {', '.join(detected_pii[:5])}...")
            # If strict compliance needed, mark as non-compliant
            if "GDPR" in compliance_standards or "HIPAA" in compliance_standards:
                is_compliant = False
        
        remediation = "Redact PII" if detected_pii else "None"
        
        return {
            "compliant": is_compliant,
            "violations": violations,
            "remediation": remediation,
            "detected_pii_count": pii_result['count']
        }