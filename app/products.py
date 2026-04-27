"""
app/products.py

Product catalog loading and recommendation via keyword matching.
"""

import json
from pathlib import Path
from typing import Literal
from app.models import ProductRef


PRODUCTS_PATH = Path(__file__).parent.parent / "data" / "products.json"
MAX_PRODUCTS = 3

# Singleton: loaded once at startup
_catalog: list[dict] = []


def load_product_catalog():
    """Load product catalog from JSON file (call once at startup)."""
    global _catalog
    _catalog = json.loads(PRODUCTS_PATH.read_text(encoding="utf-8"))
    print(f"Product catalog loaded: {len(_catalog)} products")


def get_relevant_products(
    home_care: list[str],
    severity: str,
    language: Literal["en", "ar"],
) -> list[ProductRef]:
    """
    Match home care steps against product keywords to find relevant products.

    Never suggests products during emergencies — medical care takes priority.
    Returns at most MAX_PRODUCTS results.
    """
    # No products during emergencies — do not distract from urgent care
    if severity == "emergency":
        return []

    if not _catalog:
        return []

    # Build a single searchable string from all home care steps
    home_care_text = " ".join(home_care).lower()

    matched = []
    seen_ids = set()

    for product in _catalog:
        if product["id"] in seen_ids:
            continue

        # Check if any product keyword appears in the home care text
        keywords = [kw.lower() for kw in product.get("use_case_keywords", [])]
        if any(kw in home_care_text for kw in keywords):
            name = product["name_ar"] if language == "ar" else product["name_en"]
            matched.append(ProductRef(id=product["id"], name=name))
            seen_ids.add(product["id"])

        if len(matched) >= MAX_PRODUCTS:
            break

    return matched
