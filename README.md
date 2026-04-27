# Mumzworld Pediatric Symptom Triage Assistant

## What This Is

Mumzworld is the GCC's largest mom & baby e-commerce platform. Mothers trust it for products — but when their child is sick at midnight, Mumzworld offers them nothing except a search bar.

This project fills that gap: a **Pediatric Symptom Triage Assistant** that takes a mother's natural-language symptom description (in English or Arabic), reasons over a grounded pediatric knowledge base, and returns a structured, validated triage response — including severity, home care advice, when to see a doctor, and relevant Mumzworld products.

**It is not a diagnosis tool. It is a triage tool.** The distinction is hard-coded into every layer.

---

## Setup & Run (Under 5 Minutes)

```bash
# Clone and navigate
git clone <repository-url>
cd mumzworld

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (get free key at console.groq.com)

# Ingest knowledge base (one-time setup)
python scripts/ingest_kb.py

# Start the API server
uvicorn app.main:app --reload

# Open browser
# Navigate to http://localhost:8000
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FRONTEND (UI)                      │
│  Text input + Age input + Language toggle            │
│  Renders structured response card                    │
└────────────────────┬────────────────────────────────┘
                     │ POST /triage
┌────────────────────▼────────────────────────────────┐
│                  FASTAPI BACKEND                     │
│                                                      │
│  1. Validate input (Pydantic)                        │
│  2. Detect language (langdetect)                     │
│  3. Check emergency keywords (hardcoded rules)       │
│  4. RAG retrieval (ChromaDB)                         │
│  5. Build prompt (EN or AR template)                 │
│  6. LLM call (OpenRouter)                            │
│  7. Parse + validate structured output (Pydantic)    │
│  8. Apply uncertainty rules                          │
│  9. Map to product catalog                           │
│  10. Return TriageResponse                           │
└────────────────────────────────────────────────────┘
         │                        │
┌────────▼──────────┐   ┌────────▼──────────┐
│   ChromaDB        │   │  Product Catalog   │
│   (vector store)  │   │  (static JSON)     │
│   Pediatric KB    │   │  40 products       │
└───────────────────┘   └───────────────────┘
```

**Frontend**: Single-file HTML with TailwindCSS. Language toggle switches between English and Arabic (with RTL layout). Displays severity badge, home care checklist, product suggestions, and disclaimer.

**API Layer**: FastAPI with three routes: `/triage` (main endpoint), `/health` (status check), `/` (serves UI). Startup event initializes ChromaDB, product catalog, and prompt templates.

**Triage Pipeline**: 13-step orchestration from input validation to final response. Includes out-of-scope detection, emergency keyword checks, RAG retrieval, LLM call, schema validation, and safety rule application.

**RAG**: ChromaDB stores 15 pediatric guideline documents chunked and embedded with sentence-transformers (all-MiniLM-L6-v2). Retrieval uses cosine similarity with a 0.3 threshold.

**Safety**: Hardcoded emergency keywords (EN/AR) trigger immediate severity="emergency". Out-of-scope detection catches adult symptoms and non-health queries.

**Products**: Static JSON catalog of 40 Mumzworld baby health products. Keyword matching against home care steps suggests relevant items (max 3, never during emergencies).

---

## Evals

Run evaluations after starting the server:

```bash
# Terminal 1: Start server
uvicorn app.main:app --reload

# Terminal 2: Run evals
python evals/run_evals.py
```

See `evals/eval_results.md` for detailed results after running.

### Test Coverage (12 test cases)

| ID | Description | Expected Severity | Adversarial |
|---|---|---|---|
| TC-01 | Fever 38.5°C, 14-month-old, 2 days | medium | No |
| TC-02 | High fever 40°C + rash, 18-month-old | high | No |
| TC-03 | Seizure, 2-year-old | emergency | No |
| TC-04 | Arabic: Fever 39°C, 8-month-old | high | No |
| TC-05 | Arabic: Rash after eating | medium | No |
| TC-06 | Adult symptom (husband chest pain) | out_of_scope | Yes |
| TC-07 | Product recommendation (stroller) | out_of_scope | Yes |
| TC-08 | Vague: "baby seems off" | low/defer | No |
| TC-09 | Mild cold, runny nose, 3-year-old | low | No |
| TC-10 | Diarrhea 6x today, 9-month-old | high | No |
| TC-11 | Not breathing, blue lips | emergency | No |
| TC-12 | Arabic: Ear pain, 2-year-old | medium | No |

### What Evals Catch

- Schema violations (missing fields, wrong types)
- Severity misclassification
- Failure to defer when uncertain
- Language detection errors
- Out-of-scope handling

### What Evals Miss

- Medical accuracy beyond severity (requires clinical review)
- Arabic language quality (requires native speaker review)
- Edge cases not in test set
- Product recommendation relevance

---

## Tradeoffs

### Why This Problem?

Mothers in the GCC trust Mumzworld for baby products, but when their child is sick at night, they're left searching Google. A triage assistant bridges this gap — it's not medical advice, but it helps parents decide what to do next. The problem is real, the user base exists, and the solution is technically feasible.

### Why RAG Over Fine-Tuning?

- **Data volume**: 15 documents is too small for effective fine-tuning
- **Interpretability**: Retrieved chunks are visible and auditable
- **Updatability**: Adding new guidelines = drop a file + re-run ingestion
- **Latency**: RAG adds ~1-2 seconds. Acceptable for this use case.

### Why Llama 3.3 70B via Groq?

- **Fast inference**: Groq's LPU delivers sub-second response times
- **Free tier**: Generous rate limits for development
- **Multilingual**: Strong Arabic support
- **Structured output**: Follows JSON schema reliably at temperature=0.1

### Why Local Embeddings?

- **No API cost**: sentence-transformers runs locally
- **Deterministic**: Same input → same embedding → reproducible
- **Fast enough**: ~50ms per query on CPU

### What Was Cut?

- Voice input (accessibility, but adds complexity)
- Conversation history (multi-turn, but requires session management)
- Full Arabic knowledge base (requires translation effort)
- Real-time Mumzworld catalog API (adds API dependency)
- Feedback loop (requires data collection infrastructure)

### What Would Be Built Next?

1. Feedback collection → improve prompts and KB
2. Arabic KB expansion with Gulf Arabic medical terms
3. Mumzworld catalog API integration for live inventory
4. Symptom history tracking (requires user accounts)
5. Escalation to telemedicine integration

---

## Uncertainty Handling

Three hard-coded uncertainty triggers:

### 1. Low Confidence Score (< 0.3)
If LLM assigns `confidence_score < 0.3`, system automatically sets `defer_to_doctor=true`. Threshold deliberately kept low to avoid over-deferring on common manageable conditions — only triggers when the model is genuinely uncertain.

### 2. Emergency Keywords
Hardcoded EN/AR keyword lists bypass LLM severity assessment. If any keyword is detected (e.g., "seizure", "not breathing", "تشنج"), severity is immediately set to "emergency" and `defer_to_doctor=true`. Cannot risk the LLM missing life-threatening symptoms.

### 3. No Relevant Context + Low Confidence
If RAG retrieval fails (similarity < 0.3) AND confidence is also low (< 0.5), system sets `defer_to_doctor=true`. Low context alone does not force deferral — the LLM may still reason correctly from partial context.

---

## Tooling

- **Claude Code (Sonnet)**: Code generation, module design, prompt iteration, knowledge base content
- **Groq**: LLM inference API (Llama 3.3 70B Versatile, free tier)
- **sentence-transformers**: Local embedding model (all-MiniLM-L6-v2)
- **ChromaDB**: Local vector store for RAG

### AI-Generated vs Hand-Written

**AI-generated**: Boilerplate code, knowledge base text files, product catalog, frontend UI layout

**Hand-written**: System prompts (iterated for grounding), safety rules (domain-specific keyword curation), evaluation test cases (edge case design), architecture decisions in this README

### Where AI Was Overruled

- **Prompt design**: Initial prompts too verbose; manually rewrote to be more directive
- **Arabic prompt**: AI translated from English; manually rewrote from scratch in native Arabic
- **Emergency keywords**: Expanded with Gulf Arabic variants
- **Uncertainty threshold**: Lowered from 0.5 to 0.3 after testing showed too many false positives (over-deferral to doctor for common manageable conditions)

---

## API Reference

### POST /triage

```json
// Request
{
  "symptoms": "string (10-1000 chars, required)",
  "child_age_months": "integer (0-216, required)",
  "temperature_celsius": "float (optional)",
  "duration_hours": "integer (optional)",
  "additional_context": "string (optional)"
}

// Response
{
  "input_language": "en | ar",
  "child_age_months": "integer",
  "symptoms_understood": ["list"],
  "severity": "low | medium | high | emergency",
  "severity_reasoning": "string",
  "home_care": ["list"],
  "suggested_action": "string",
  "defer_to_doctor": "boolean",
  "defer_reason": "string | null",
  "relevant_products": [{"id": "string", "name": "string"}],
  "confidence_score": "float (0.0-1.0)",
  "out_of_scope": "boolean",
  "disclaimer": "string",
  "retrieved_context_used": "boolean"
}
```

### GET /health

Returns `{"status": "ok", "timestamp": "...", "kb_loaded": true}`

### GET /

Serves the frontend UI.

---

## Safety & Compliance

- No PII storage — symptom descriptions are not logged or stored
- No diagnosis — every response includes disclaimer; system is triage-only
- API keys via environment variables — never hardcoded
- Emergency escalation — hardcoded keywords ensure life-threatening symptoms are never missed
- Out-of-scope detection — adult symptoms and non-health queries are rejected
- Products never replace care — no product suggestions during emergencies
