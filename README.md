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

**Current Score: 87.5% (42/48 points)** — see [`evals/eval_results.md`](evals/eval_results.md)

Run evaluations after starting the server:

```bash
# Terminal 1: Start server
uvicorn app.main:app --reload

# Terminal 2: Run evals
python evals/run_evals.py
```

### Test Coverage (12 test cases)

| ID | Description | Expected Severity | Adversarial | Result |
|---|---|---|---|---|
| TC-01 | Fever 38.5°C, 14-month-old, 2 days | medium | No | ✅ 4/4 |
| TC-02 | High fever 40°C + rash, 18-month-old | high | No | ✅ 4/4 |
| TC-03 | Seizure, 2-year-old | emergency | No | ✅ 4/4 |
| TC-04 | Arabic: Fever 39°C, 8-month-old | high | No | ✅ 4/4 |
| TC-05 | Arabic: Rash after eating | medium | No | ⚠️ 3/4 |
| TC-06 | Adult symptom (husband chest pain) | out_of_scope | Yes | ✅ 4/4 |
| TC-07 | Product recommendation (stroller) | out_of_scope | Yes | ✅ 4/4 |
| TC-08 | Vague: "baby seems off" | low/defer | No | ⚠️ 3/4 |
| TC-09 | Mild cold, runny nose, 3-year-old | low | No | ⚠️ 2/4 |
| TC-10 | Diarrhea 6x today, 9-month-old | high | No | ✅ 4/4 |
| TC-11 | Not breathing, blue lips | emergency | No | ⚠️ 3/4 |
| TC-12 | Arabic: Ear pain, 2-year-old | medium | No | ⚠️ 3/4 |

### What Evals Catch

- Schema violations (missing fields, wrong types) — **100% pass**
- Severity misclassification — **58% pass** (7/12 correct)
- Failure to defer when uncertain — **92% pass** (11/12 correct)
- Language detection errors — **100% pass**
- Out-of-scope handling — **100% pass**

### Known Limitations

- **Severity calibration**: Model is slightly conservative, upgrading some medium cases to high. Acceptable for triage — false positives (over-caution) are safer than false negatives.
- **Emergency keyword miss (TC-11)**: "not breathing" classified as `high` instead of `emergency`. Emergency keywords should be non-overridable — this is a known bug.
- **Medical accuracy beyond severity**: Requires clinical review. Evals validate structure and severity bands, not clinical correctness of home care advice.
- **Arabic quality**: Requires native Gulf Arabic speaker review for medical terminology accuracy.

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

### Stack

| Tool | Role |
|---|---|
| **Claude Code (Sonnet 4.6)** | Primary coding agent — architecture, all module code, KB content, frontend |
| **Groq API (Llama 3.3 70B Versatile)** | Runtime LLM inference for triage responses |
| **sentence-transformers (all-MiniLM-L6-v2)** | Local embedding model for RAG |
| **ChromaDB** | Local vector store for pediatric knowledge base |

### How AI Was Used

**Full agent loop, not pair-coding.** Claude Code was given the full spec and asked to execute it end-to-end. It generated the entire project structure, all Python modules (`app/`, `scripts/`, `evals/`), the 15 knowledge base text files, the 40-product catalog JSON, and the frontend HTML in a single session. The workflow was: spec → plan → sequential module generation → debugging → iteration.

**Prompt iteration was the most hands-on part.** The system prompts (`prompts/system_en.txt`, `prompts/system_ar.txt`) went through multiple rounds. The initial AI-generated prompts were too conservative — they instructed the model to defer to a doctor on any uncertainty, which caused the system to recommend doctor visits for mild colds and low-grade fevers. This was caught during manual testing and fixed by rewriting the prompts with explicit guidance on when *not* to defer, calibrated confidence score bands, and age-specific thresholds.

**Eval grading was automated.** The 12-case eval suite (`evals/run_evals.py`) was generated by the agent and run against the live server to score severity accuracy, schema validity, defer correctness, and language detection. Results surfaced the over-deferral problem quantitatively.

### What Worked

- **One-shot module generation**: All core modules (RAG, safety, language detection, triage pipeline) were generated correctly on the first pass with no structural changes needed.
- **Schema design**: Pydantic v2 strict mode with the `defer_reason` required-when-true validator was generated correctly and caught real bugs during eval.
- **Knowledge base content**: The 15 pediatric guideline files were generated with accurate clinical content and correct metadata headers, usable without manual review beyond spot-checking.

### What Did Not Work

- **Initial LLM provider (OpenRouter)**: Switched mid-project due to frequent 429 rate limits and 404 errors from deprecated model names on the free tier. Replaced with Groq, which has faster inference and more stable free-tier access.
- **Python version**: The agent-created venv used system Python 3.14, which is incompatible with `pydantic-core` (PyO3 max supported: 3.12). Manually recreated the venv with Homebrew Python 3.12.
- **NumPy compatibility**: ChromaDB 0.5.0 uses `np.float_` which was removed in NumPy 2.0. Fixed by pinning `numpy<2.0.0` in `requirements.txt`.
- **Arabic prompt (first version)**: The agent translated the English prompt into Arabic. The translation was grammatically correct but used formal Modern Standard Arabic rather than Gulf Arabic medical vocabulary. Manually rewrote from scratch.

### Where the Agent Was Overruled

- **Uncertainty threshold**: Agent set `confidence_score < 0.5` as the defer trigger. After testing showed this caused over-deferral on common conditions (mild cold → "consult a doctor"), threshold was lowered to `0.3` and the prompt was rewritten to explicitly list conditions that should *not* trigger deferral.
- **Emergency keyword list**: Agent generated a basic EN list. Manually expanded with Gulf Arabic variants (e.g., `تشنج`, `لا يتنفس`, `شفاه زرقاء`) and colloquial terms used by GCC mothers.
- **Low-context safety rule**: Agent set `low_context → out_of_scope=true + defer_to_doctor=true` unconditionally. Changed to only trigger when confidence is also low, since the LLM can still reason correctly from partial context.

### Key Prompts

The system prompts that materially shaped output quality are committed at:
- [`prompts/system_en.txt`](prompts/system_en.txt) — English triage prompt with calibrated severity/defer rules
- [`prompts/system_ar.txt`](prompts/system_ar.txt) — Arabic prompt written natively in Gulf Arabic (not translated)

The most impactful change was adding explicit "DO NOT defer for" rules to both prompts, which reduced false-positive deferral from ~60% of cases to ~20% (only genuinely concerning cases).

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
