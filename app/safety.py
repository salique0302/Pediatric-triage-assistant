"""
app/safety.py

Safety checks: emergency keyword detection and out-of-scope query detection.
These are hardcoded rules — they run before any LLM call.
"""

from dataclasses import dataclass


# Emergency keywords in English — any match triggers severity="emergency"
EMERGENCY_KEYWORDS_EN = [
    "seizure", "convulsion", "convulsing", "not breathing", "stopped breathing",
    "can't breathe", "cannot breathe", "unconscious", "unresponsive",
    "blue lips", "blue face", "blue tongue", "blue fingernails",
    "limp", "floppy", "won't wake", "can't wake", "won't wake up",
    "severe allergic", "anaphylaxis", "epipen",
    "choking", "not responding", "collapsed", "collapse",
]

# Emergency keywords in Arabic — transliterated and native forms
EMERGENCY_KEYWORDS_AR = [
    "تشنج", "تشنجات", "لا يتنفس", "لا تتنفس", "توقف عن التنفس",
    "فاقد الوعي", "فاقدة الوعي", "شفاه زرقاء", "شفتين زرقاوتين",
    "غير مستجيب", "غير مستجيبة", "مغمى عليه", "مغمى عليها",
    "لا يستجيب", "لا تستجيب", "اختناق", "يختنق", "تختنق",
    "تحسس شديد", "صدمة تحسسية",
]

# Keywords indicating adult symptoms (not pediatric)
ADULT_SYMPTOM_KEYWORDS = [
    "my husband", "my wife", "my mother", "my father", "my sister", "my brother",
    "i have", "i feel", "i am sick", "i'm sick", "i am feeling", "i'm feeling",
    "i have a fever", "i have pain", "my pain", "i am in pain",
]

# Keywords indicating non-health queries
NON_HEALTH_KEYWORDS = [
    "stroller", "pram", "car seat", "toy", "toys", "recommend a product",
    "shipping", "delivery", "order", "return", "refund", "account",
    "password", "login", "discount", "coupon", "promo", "recipe",
    "how to cook", "what to buy", "best product", "price",
]


@dataclass
class OutOfScopeResult:
    """Result of out-of-scope check"""
    out_of_scope: bool
    reason: str | None


def check_emergency_keywords(symptoms: str) -> bool:
    """
    Check if symptoms contain any emergency keywords.

    Returns True if emergency keywords found — caller must set severity="emergency".
    Case-insensitive check for both English and Arabic keywords.
    """
    text_lower = symptoms.lower()

    for keyword in EMERGENCY_KEYWORDS_EN:
        if keyword in text_lower:
            print(f"Emergency keyword detected: '{keyword}'")
            return True

    for keyword in EMERGENCY_KEYWORDS_AR:
        if keyword in symptoms:  # Arabic is case-insensitive by nature
            print(f"Emergency keyword detected (AR): '{keyword}'")
            return True

    return False


def check_out_of_scope(symptoms: str, child_age_months: int) -> OutOfScopeResult:
    """
    Check if the query is outside the scope of pediatric symptom triage.

    Returns OutOfScopeResult with out_of_scope flag and reason.
    """
    text_lower = symptoms.lower()

    # Check for adult symptom indicators
    for keyword in ADULT_SYMPTOM_KEYWORDS:
        if keyword in text_lower:
            return OutOfScopeResult(
                out_of_scope=True,
                reason=(
                    f"This assistant is for pediatric (child) symptom triage only. "
                    f"The query appears to describe adult symptoms. "
                    f"Please consult a doctor or adult medical resource."
                ),
            )

    # Check for non-health queries
    for keyword in NON_HEALTH_KEYWORDS:
        if keyword in text_lower:
            return OutOfScopeResult(
                out_of_scope=True,
                reason=(
                    f"This assistant handles pediatric health symptom triage only. "
                    f"For product recommendations, orders, or other queries, "
                    f"please contact Mumzworld customer support."
                ),
            )

    return OutOfScopeResult(out_of_scope=False, reason=None)
