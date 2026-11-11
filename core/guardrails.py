import re
from typing import Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field

# Strict Prompt Injection Patterns (Handbook Section 8.5)
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior)\s+instructions",
    r"system\s+prompt\s+override",
    r"you\s+are\s+now\s+(an?\s+)?admin",
    r"export\s+all\s+(customer|database|system)\s+records",
    r"bypass\s+(all\s+)?(security|guardrails|policy)",
    r"drop\s+table",
    r"select\s+\*\s+from\s+users",
]

# Sensitive PII Patterns
CREDIT_CARD_PATTERN = r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"


class GuardrailCheckResult(BaseModel):
    is_safe: bool
    violation_type: Optional[str] = None
    sanitized_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


def sanitize_and_check_injection(user_input: str) -> GuardrailCheckResult:
    """
    Deterministic input guardrail.
    Inspects text for prompt injection signatures and masks potential PII.
    """
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, user_input, re.IGNORECASE):
            return GuardrailCheckResult(
                is_safe=False,
                violation_type="PROMPT_INJECTION_ATTEMPT",
                sanitized_text="[FLAGGED_MALICIOUS_INPUT]",
                metadata={"pattern_matched": pattern}
            )

    # Redact sensitive PII before context ingestion
    sanitized = re.sub(CREDIT_CARD_PATTERN, "[REDACTED_CREDIT_CARD]", user_input)
    sanitized = re.sub(SSN_PATTERN, "[REDACTED_SSN]", sanitized)

    return GuardrailCheckResult(
        is_safe=True,
        violation_type=None,
        sanitized_text=sanitized
    )


def validate_refund_policy(amount: float, order_exists: bool, days_since_delivery: int = 10) -> Tuple[bool, str]:
    """
    Deterministic policy guardrail (Handbook Section 6.4, 8.4).
    Enforces business rules independent of model predictions.
    """
    if not order_exists:
        return False, "Order ID could not be verified in system of record."
    if days_since_delivery > 30:
        return False, "Order exceeds 30-day resolution return window."
    if amount <= 0:
        return False, "Refund amount must be strictly greater than $0.00."
    return True, "Policy check verified."
