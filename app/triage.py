"""
app/triage.py

Core triage orchestration: the 13-step pipeline from input to validated response.
"""

import json
from app.models import TriageRequest, TriageResponse, ProductRef
from app.safety import check_emergency_keywords, check_out_of_scope
from app.language import detect_language, get_prompt_template
from app.rag import retrieve_context
from app.llm import call_llm, TriageLLMError
from app.products import get_relevant_products


# Disclaimers in both languages
DISCLAIMER_EN = "This is not medical advice. Always consult a qualified healthcare professional for diagnosis and treatment."
DISCLAIMER_AR = "هذا ليس نصيحة طبية. يُرجى دائماً استشارة طبيب مختص للتشخيص والعلاج."


async def run_triage(request: TriageRequest) -> TriageResponse:
    """
    Main triage orchestration function.

    13-step pipeline:
    1. Out-of-scope check
    2. Emergency keyword check
    3. Language detection
    4. RAG retrieval
    5. Build user message
    6. Get system prompt
    7. LLM call
    8. Parse JSON response
    9. Validate against schema
    10. Apply hard safety rules
    11. Get product recommendations
    12. Attach disclaimer
    13. Return final response
    """

    # Step 1: Out-of-scope check
    scope_check = check_out_of_scope(request.symptoms, request.child_age_months)
    if scope_check.out_of_scope:
        # Return minimal response for out-of-scope queries
        language = detect_language(request.symptoms)
        disclaimer = DISCLAIMER_AR if language == "ar" else DISCLAIMER_EN
        return TriageResponse(
            input_language=language,
            child_age_months=request.child_age_months,
            symptoms_understood=[],
            severity="low",
            severity_reasoning=scope_check.reason or "Query is out of scope",
            home_care=[],
            suggested_action=scope_check.reason or "This query is outside the scope of pediatric symptom triage.",
            defer_to_doctor=True,
            defer_reason=scope_check.reason,
            relevant_products=[],
            confidence_score=0.0,
            out_of_scope=True,
            disclaimer=disclaimer,
            retrieved_context_used=False,
        )

    # Step 2: Emergency keyword check
    is_emergency = check_emergency_keywords(request.symptoms)

    # Step 3: Language detection
    language = detect_language(request.symptoms)

    # Step 4: RAG retrieval
    retrieval = retrieve_context(request.symptoms, n_results=4)
    low_context = not retrieval.retrieved

    # Step 5: Build user message
    context_text = "\n\n---\n\n".join(retrieval.chunks) if retrieval.chunks else "No relevant context found."

    user_message = f"""Child age: {request.child_age_months} months.
Symptoms: {request.symptoms}"""

    if request.temperature_celsius:
        user_message += f"\nTemperature: {request.temperature_celsius}°C"
    if request.duration_hours:
        user_message += f"\nDuration: {request.duration_hours} hours"
    if request.additional_context:
        user_message += f"\nAdditional context: {request.additional_context}"

    user_message += f"\n\nContext from medical guidelines:\n{context_text}"

    # Step 6: Get system prompt
    system_prompt = get_prompt_template(language)

    # Step 7: LLM call
    try:
        llm_response = await call_llm(system_prompt, user_message)
    except TriageLLMError as e:
        # LLM call failed — return error response
        disclaimer = DISCLAIMER_AR if language == "ar" else DISCLAIMER_EN
        return TriageResponse(
            input_language=language,
            child_age_months=request.child_age_months,
            symptoms_understood=[],
            severity="high",
            severity_reasoning=f"Unable to complete triage due to system error: {str(e)}",
            home_care=[],
            suggested_action="Please consult a doctor immediately due to system unavailability.",
            defer_to_doctor=True,
            defer_reason=f"System error: {str(e)}",
            relevant_products=[],
            confidence_score=0.0,
            out_of_scope=False,
            disclaimer=disclaimer,
            retrieved_context_used=retrieval.retrieved,
        )

    # Step 8: Parse JSON response
    try:
        # Strip markdown fences if present
        response_text = llm_response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        parsed = json.loads(response_text)
    except json.JSONDecodeError as e:
        # JSON parse failed — return error response
        disclaimer = DISCLAIMER_AR if language == "ar" else DISCLAIMER_EN
        return TriageResponse(
            input_language=language,
            child_age_months=request.child_age_months,
            symptoms_understood=[],
            severity="high",
            severity_reasoning="Unable to parse triage response",
            home_care=[],
            suggested_action="Please consult a doctor immediately.",
            defer_to_doctor=True,
            defer_reason=f"Response parsing error: {str(e)}",
            relevant_products=[],
            confidence_score=0.0,
            out_of_scope=False,
            disclaimer=disclaimer,
            retrieved_context_used=retrieval.retrieved,
        )

    # Step 9: Validate against schema
    try:
        response = TriageResponse.model_validate(parsed)
    except Exception as e:
        # Schema validation failed — return error response
        disclaimer = DISCLAIMER_AR if language == "ar" else DISCLAIMER_EN
        return TriageResponse(
            input_language=language,
            child_age_months=request.child_age_months,
            symptoms_understood=[],
            severity="high",
            severity_reasoning="Unable to validate triage response",
            home_care=[],
            suggested_action="Please consult a doctor immediately.",
            defer_to_doctor=True,
            defer_reason=f"Response validation error: {str(e)}",
            relevant_products=[],
            confidence_score=0.0,
            out_of_scope=False,
            disclaimer=disclaimer,
            retrieved_context_used=retrieval.retrieved,
        )

    # Step 10: Apply hard safety rules
    if is_emergency:
        response.severity = "emergency"
        response.defer_to_doctor = True
        if not response.defer_reason:
            response.defer_reason = "Emergency keywords detected in symptoms"

    # Only override defer for very low confidence
    if response.confidence_score < 0.3:
        response.defer_to_doctor = True
        if not response.defer_reason:
            response.defer_reason = "Insufficient confidence in assessment"

    # Low context warning but don't force defer unless confidence is also low
    if low_context and response.confidence_score < 0.5:
        response.defer_to_doctor = True
        if not response.defer_reason:
            response.defer_reason = "Limited relevant medical context available"

    # Step 11: Get product recommendations
    response.relevant_products = get_relevant_products(
        home_care=response.home_care,
        severity=response.severity,
        language=language,
    )

    # Step 12: Attach disclaimer in correct language
    response.disclaimer = DISCLAIMER_AR if language == "ar" else DISCLAIMER_EN

    # Step 13: Return final response
    return response
