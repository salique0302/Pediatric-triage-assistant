# Evaluation Results

**Overall Score:** TBD (run `python evals/run_evals.py` after starting the server)

**Pass Threshold:** 80%

**Status:** Not yet run

## Instructions

1. Start the server: `uvicorn app.main:app --reload`
2. Run evaluations: `python evals/run_evals.py`
3. Results will be updated in this file

## Test Coverage

The evaluation suite includes 12 test cases covering:
- 3 clear in-scope EN symptoms (fever, cold, diarrhea)
- 3 clear in-scope AR symptoms (fever, rash, ear pain)
- 2 adversarial/out-of-scope inputs (adult symptom, product question)
- 2 ambiguous/low-confidence inputs (vague symptoms)
- 2 emergency keyword inputs (seizure, not breathing)

Each test case is scored on 4 dimensions:
1. **Schema validity**: Response parses and validates against TriageResponse schema
2. **Severity accuracy**: Severity level matches expected
3. **Uncertainty handling**: defer_to_doctor flag is correct
4. **Language correctness**: Response is in the correct language (EN or AR)
