"""
evals/run_evals.py

Evaluation runner: tests the triage API against predefined test cases.
Server must be running at http://localhost:8000 before running this script.
"""

import json
import sys
from pathlib import Path
import httpx


API_URL = "http://localhost:8000/triage"
TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.md"


def load_test_cases():
    """Load test cases from JSON file."""
    return json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))


def run_test_case(test_case: dict) -> dict:
    """Run a single test case against the API."""
    try:
        response = httpx.post(
            API_URL,
            json=test_case["input"],
            timeout=20.0,
        )

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"API returned {response.status_code}: {response.text}",
                "response": None,
            }

        return {
            "success": True,
            "error": None,
            "response": response.json(),
        }

    except httpx.RequestError as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}",
            "response": None,
        }


def score_response(test_case: dict, result: dict) -> dict:
    """
    Score a test case response on 4 dimensions:
    - schema_valid: did response parse and validate?
    - severity_match: does severity match expected?
    - uncertainty_correct: is defer_to_doctor correct?
    - language_correct: is response in correct language?

    Returns dict with scores (0 or 1 each) and total.
    """
    if not result["success"]:
        return {
            "schema_valid": 0,
            "severity_match": 0,
            "uncertainty_correct": 0,
            "language_correct": 0,
            "total": 0,
            "max": 4,
        }

    response = result["response"]
    expected = test_case["expected"]

    # Schema valid: if we got here, it parsed successfully
    schema_valid = 1

    # Severity match
    severity_match = 1 if response["severity"] == expected["severity"] else 0

    # Uncertainty correct
    uncertainty_correct = 1 if response["defer_to_doctor"] == expected["defer_to_doctor"] else 0

    # Language correct
    language_correct = 1 if response["input_language"] == expected["language"] else 0

    total = schema_valid + severity_match + uncertainty_correct + language_correct

    return {
        "schema_valid": schema_valid,
        "severity_match": severity_match,
        "uncertainty_correct": uncertainty_correct,
        "language_correct": language_correct,
        "total": total,
        "max": 4,
    }


def run_evals():
    """Main evaluation function."""
    print("=== Mumzworld Triage Evaluation ===\n")
    print(f"API URL: {API_URL}")
    print(f"Test cases: {TEST_CASES_PATH}\n")

    # Check if server is running
    try:
        httpx.get("http://localhost:8000/health", timeout=5.0)
    except httpx.RequestError:
        print("ERROR: Server is not running at http://localhost:8000")
        print("Start the server first: uvicorn app.main:app --reload")
        sys.exit(1)

    # Load test cases
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases\n")

    # Run tests
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Running {test_case['id']}: {test_case['description']}")
        result = run_test_case(test_case)
        score = score_response(test_case, result)
        results.append({
            "test_case": test_case,
            "result": result,
            "score": score,
        })
        print(f"  Score: {score['total']}/{score['max']}")
        if not result["success"]:
            print(f"  Error: {result['error']}")
        print()

    # Calculate overall score
    total_score = sum(r["score"]["total"] for r in results)
    max_score = sum(r["score"]["max"] for r in results)
    percentage = (total_score / max_score * 100) if max_score > 0 else 0

    # Print summary table
    print("\n=== Results Summary ===\n")
    print(f"{'ID':<8} {'Description':<40} {'Schema':<7} {'Severity':<9} {'Defer':<7} {'Lang':<6} {'Total':<5}")
    print("-" * 90)

    for r in results:
        tc = r["test_case"]
        s = r["score"]
        print(
            f"{tc['id']:<8} "
            f"{tc['description'][:38]:<40} "
            f"{s['schema_valid']:<7} "
            f"{s['severity_match']:<9} "
            f"{s['uncertainty_correct']:<7} "
            f"{s['language_correct']:<6} "
            f"{s['total']}/{s['max']:<3}"
        )

    print("-" * 90)
    print(f"{'TOTAL':<8} {'':<40} {'':<7} {'':<9} {'':<7} {'':<6} {total_score}/{max_score} ({percentage:.1f}%)")

    # Write results to markdown
    write_results_markdown(results, total_score, max_score, percentage)

    print(f"\nResults saved to: {RESULTS_PATH}")

    # Exit code based on pass threshold (80%)
    if percentage >= 80:
        print(f"\n✅ PASS: {percentage:.1f}% >= 80%")
        sys.exit(0)
    else:
        print(f"\n❌ FAIL: {percentage:.1f}% < 80%")
        sys.exit(1)


def write_results_markdown(results, total_score, max_score, percentage):
    """Write evaluation results to markdown file."""
    lines = [
        "# Evaluation Results\n",
        f"**Overall Score:** {total_score}/{max_score} ({percentage:.1f}%)\n",
        f"**Pass Threshold:** 80%\n",
        f"**Status:** {'✅ PASS' if percentage >= 80 else '❌ FAIL'}\n",
        "\n## Test Case Results\n",
        "| ID | Description | Schema | Severity | Defer | Lang | Total |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in results:
        tc = r["test_case"]
        s = r["score"]
        lines.append(
            f"| {tc['id']} | {tc['description']} | "
            f"{'✅' if s['schema_valid'] else '❌'} | "
            f"{'✅' if s['severity_match'] else '❌'} | "
            f"{'✅' if s['uncertainty_correct'] else '❌'} | "
            f"{'✅' if s['language_correct'] else '❌'} | "
            f"{s['total']}/{s['max']} |"
        )

    lines.append(f"\n**Total:** {total_score}/{max_score} ({percentage:.1f}%)\n")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_evals()
