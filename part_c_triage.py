
from pathlib import Path
import json
import random
import re
import time

import pandas as pd
import mock_llm


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
TICKETS_FILE = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3

# Required demo budget. If no tickets exhaust this budget,
# lower it to 5000 and rerun the budget demonstration.
TOKEN_BUDGET = 6000
FINAL_TOKEN_BUDGET = 100000

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")


# ============================================================
# PROMPTS
# ============================================================

ZERO_SHOT_PROMPT = """
CLASSIFY the customer's support ticket into exactly one category.
Allowed categories: billing, delivery, technical, fraud.
Return JSON only in this form:
{"category": "...", "reason": "..."}
The category must be one of the four allowed categories.
Do not include markdown or text outside the JSON object.
""".strip()


FEW_SHOT_PROMPT = """
CLASSIFY the customer's support ticket into exactly one category.
Allowed categories: billing, delivery, technical, fraud.
Return JSON only in this form:
{"category": "...", "reason": "..."}
The category must be one of the four allowed categories.
Do not include markdown or text outside the JSON object.

EXAMPLES:
Ticket: I was charged twice for the same purchase.
{"category": "billing", "reason": "The customer reports a duplicate charge."}

Ticket: My order has not arrived and delivery is delayed.
{"category": "delivery", "reason": "The customer reports a delayed delivery."}

Ticket: Someone accessed my account and made an unauthorised transaction.
{"category": "fraud", "reason": "The customer reports a potentially unauthorised transaction."}

Ticket: The application crashes when I try to open it.
{"category": "technical", "reason": "The customer reports an application malfunction."}
""".strip()


# ============================================================
# PRIVACY
# ============================================================

def redact_text(text):
    """Apply the required patterns in the required order."""
    text = str(text)
    counts = {}

    text, counts["email"] = EMAIL_PATTERN.subn("[EMAIL]", text)
    text, counts["phone"] = PHONE_PATTERN.subn("[PHONE]", text)
    text, counts["order_id"] = ORDER_PATTERN.subn("[ORDER_ID]", text)

    return text, counts


def contains_unredacted_pii(text):
    """Return True if any required PII pattern remains."""
    return any(
        pattern.search(text)
        for pattern in (EMAIL_PATTERN, PHONE_PATTERN, ORDER_PATTERN)
    )


# ============================================================
# MODEL CALLS AND RETRIES
# ============================================================

def call_with_retries(messages, ticket_id):
    """
    Retry transient failures up to three times after the first attempt.
    Wait formula for retry number n:
    BASE_DELAY * 2**n * random.uniform(0.9, 1.15)
    """
    retries_used = 0

    while True:
        try:
            response = mock_llm.chat(messages)
            return response, retries_used

        except mock_llm.TransientLLMError:
            if retries_used >= MAX_RETRIES:
                raise

            wait = (
                BASE_DELAY
                * (2 ** retries_used)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"RETRY ticket_id={ticket_id} "
                f"retry_number={retries_used} wait_seconds={wait:.4f}"
            )
            time.sleep(wait)
            retries_used += 1


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_reply(reply_text):
    """
    Returns category, status, and reason.

    Valid JSON with an allowed category:
        classified_json

    Non-JSON with exactly one allowed category word:
        classified_keyword

    Anything else:
        flagged_invalid_output
    """
    try:
        parsed = json.loads(reply_text)

        if isinstance(parsed, dict):
            category = parsed.get("category")
            reason = parsed.get("reason", "")

            if category in ALLOWED_CATEGORIES:
                return category, "classified_json", str(reason)

        # Valid JSON with an invalid category must be flagged.
        return "", "flagged_invalid_output", ""

    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", str(reply_text).lower())
        found = [word for word in words if word in ALLOWED_CATEGORIES]

        if len(found) == 1:
            return found[0], "classified_keyword", str(reply_text)

        return "", "flagged_invalid_output", ""


# ============================================================
# LOAD DATA AND PREPARE REDACTED TICKETS
# ============================================================

def load_inputs():
    if not TICKETS_FILE.exists():
        raise FileNotFoundError(f"Missing ticket file: {TICKETS_FILE}")

    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing Part B segment file: {SEGMENTS_FILE}. "
            "Run part_b_analysis.py first."
        )

    tickets = pd.read_csv(TICKETS_FILE, dtype=str, keep_default_na=False)
    segments = pd.read_csv(SEGMENTS_FILE)

    required = {
        "ticket_id", "customer_id", "order_id", "text", "true_category"
    }
    missing = required - set(tickets.columns)
    if missing:
        raise ValueError(f"Ticket file is missing columns: {sorted(missing)}")

    required_segments = {"customer_id", "segment_name", "is_premium"}
    missing_segments = required_segments - set(segments.columns)
    if missing_segments:
        raise ValueError(
            f"Segment file is missing columns: {sorted(missing_segments)}"
        )

    # Preserve the original ticket order.
    tickets = tickets.reset_index(drop=True)

    redacted_texts = []
    totals = {"email": 0, "phone": 0, "order_id": 0}

    for text in tickets["text"]:
        redacted, counts = redact_text(text)
        redacted_texts.append(redacted)

        for key in totals:
            totals[key] += counts[key]

        # Assert that no required PII pattern remains.
        assert not contains_unredacted_pii(redacted), (
            "Unredacted PII remains after redaction."
        )

    tickets["redacted_text"] = redacted_texts

    print("\n" + "=" * 72)
    print("PRIVACY REDACTION AUDIT")
    print("=" * 72)
    print(f"Email substitutions: {totals['email']}")
    print(f"Phone substitutions: {totals['phone']}")
    print(f"Order ID substitutions: {totals['order_id']}")
    print(f"Total substitutions: {sum(totals.values())}")
    print("Re-scan of all redacted ticket texts: 0 PII matches")

    # Also ensure raw ticket identifiers and customer IDs are not
    # accidentally added to the model's user content.
    return tickets, segments


# ============================================================
# RUN ONE PIPELINE
# ============================================================

def run_pipeline(tickets, prompt, token_budget=None, run_name="run"):
    """
    Process tickets in file order.
    Empty and long tickets are never sent to the model.
    User-message content is exactly the redacted ticket text.
    """
    mock_llm.reset()

    results = []
    running_tokens = 0
    total_retries = 0

    for _, ticket in tickets.iterrows():
        ticket_id = ticket["ticket_id"]
        raw_text = ticket["text"]
        redacted_text = ticket["redacted_text"]

        result = {
            "ticket_id": ticket_id,
            "customer_id": ticket["customer_id"],
            "true_category": ticket["true_category"],
            "status": "",
            "category": "",
            "reason": "",
            "tokens_used": 0,
            "retries": 0,
        }

        if not str(raw_text).strip():
            result["status"] = "skipped_empty"
            results.append(result)
            continue

        input_tokens = mock_llm.est_tokens(redacted_text)

        if input_tokens > MAX_INPUT_TOKENS:
            result["status"] = "flagged_long"
            results.append(result)
            continue

        if token_budget is not None and running_tokens >= token_budget:
            result["status"] = "budget_exhausted"
            results.append(result)
            continue

        # Required privacy check immediately before each model call.
        assert not contains_unredacted_pii(redacted_text), (
            f"PII check failed for ticket {ticket_id}"
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": redacted_text},
        ]

        # Ensure the user message contains exactly the redacted ticket text.
        assert messages[1]["content"] == redacted_text
        assert len(messages) == 2

        try:
            response, retries_used = call_with_retries(messages, ticket_id)
            total_retries += retries_used

            message = response.choices[0].message
            reply_text = message.content or ""
            used_tokens = response.usage.total_tokens

            running_tokens += used_tokens
            result["tokens_used"] = used_tokens
            result["retries"] = retries_used

            category, status, reason = validate_reply(reply_text)
            result["category"] = category
            result["status"] = status
            result["reason"] = reason

        except mock_llm.TransientLLMError:
            # Defensive fallback; expected not to occur in this assignment.
            result["status"] = "flagged_invalid_output"
            result["reason"] = "Model call failed after maximum retries."

        results.append(result)

    result_df = pd.DataFrame(results)

    classified = result_df[
        result_df["status"].isin(["classified_json", "classified_keyword"])
    ]
    correct = (
        classified["category"] == classified["true_category"]
    ).sum()

    accuracy = correct / len(classified) if len(classified) else 0.0
    tokens_per_classified = (
        running_tokens / len(classified) if len(classified) else 0.0
    )

    print("\n" + "=" * 72)
    print(f"PIPELINE SUMMARY — {run_name}")
    print("=" * 72)
    print("Status counts:")
    print(result_df["status"].value_counts().to_string())
    print(f"Classified tickets: {len(classified)}")
    print(f"Correct classifications: {correct}")
    print(f"Accuracy on classified tickets: {accuracy:.3%}")
    print(f"Total response tokens: {running_tokens}")
    print(f"Tokens per classified ticket: {tokens_per_classified:.2f}")
    print(f"Transient retries used: {total_retries}")

    return result_df, running_tokens


# ============================================================
# CONFUSION TABLE
# ============================================================

def print_zero_shot_confusion(result_df):
    classified = result_df[
        result_df["status"].isin(["classified_json", "classified_keyword"])
    ]

    confusion = pd.crosstab(
        classified["true_category"],
        classified["category"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    ).reindex(
        index=sorted(ALLOWED_CATEGORIES),
        columns=sorted(ALLOWED_CATEGORIES),
        fill_value=0,
    )

    print("\nZERO-SHOT CONFUSION TABLE")
    print(confusion.to_string())


# ============================================================
# ROUTING
# ============================================================

def make_routing_results(result_df, segments):
    segment_columns = segments[
        ["customer_id", "segment_name", "is_premium"]
    ].drop_duplicates(subset=["customer_id"])

    merged = result_df.merge(
        segment_columns,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    merged["is_premium"] = (
        merged["is_premium"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    merged["segment_name"] = merged["segment_name"].fillna("")

    human_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }

    routes = []
    priorities = []

    for _, row in merged.iterrows():
        if row["status"] in human_statuses:
            routes.append("human_review")
            priorities.append("NORMAL")
        elif row["category"] == "fraud":
            routes.append("security_team")
            priorities.append("HIGH")
        elif row["is_premium"]:
            routes.append("premium_desk")
            priorities.append("HIGH")
        else:
            routes.append("support_queue")
            priorities.append("NORMAL")

    merged["route"] = routes
    merged["priority"] = priorities

    # Leave category and segment_name blank when unavailable.
    merged["category"] = merged["category"].fillna("")
    merged.loc[
        ~merged["status"].isin(["classified_json", "classified_keyword"]),
        "category",
    ] = ""

    final_columns = [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]

    return merged[final_columns]


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickets, segments = load_inputs()

    # Run zero-shot without a token budget.
    zero_results, zero_tokens = run_pipeline(
        tickets,
        ZERO_SHOT_PROMPT,
        token_budget=None,
        run_name="ZERO-SHOT — NO BUDGET",
    )

    print_zero_shot_confusion(zero_results)

    # Run few-shot without a token budget.
    few_results, few_tokens = run_pipeline(
        tickets,
        FEW_SHOT_PROMPT,
        token_budget=None,
        run_name="FEW-SHOT — NO BUDGET",
    )

    zero_classified = zero_results["status"].isin(
        ["classified_json", "classified_keyword"]
    ).sum()
    few_classified = few_results["status"].isin(
        ["classified_json", "classified_keyword"]
    ).sum()

    print("\n" + "=" * 72)
    print("ZERO-SHOT VS FEW-SHOT COST COMPARISON")
    print("=" * 72)
    print(f"Zero-shot total tokens: {zero_tokens}")
    print(
        "Zero-shot tokens per classified ticket:",
        f"{zero_tokens / zero_classified:.2f}" if zero_classified else "N/A",
    )
    print(f"Few-shot total tokens: {few_tokens}")
    print(
        "Few-shot tokens per classified ticket:",
        f"{few_tokens / few_classified:.2f}" if few_classified else "N/A",
    )
    print(f"Extra tokens used by few-shot: {few_tokens - zero_tokens}")

    # Budget demonstration using the zero-shot prompt.
    budget_results, budget_tokens = run_pipeline(
        tickets,
        ZERO_SHOT_PROMPT,
        token_budget=TOKEN_BUDGET,
        run_name=f"ZERO-SHOT — BUDGET {TOKEN_BUDGET}",
    )

    budget_path = OUTPUT_DIR / "triage_budget_demo.csv"
    budget_results.to_csv(budget_path, index=False)
    budget_count = (budget_results["status"] == "budget_exhausted").sum()

    # If 6000 did not exhaust the budget, retry the demo at 5000.
    if budget_count == 0:
        demo_budget = 5000
        budget_results, budget_tokens = run_pipeline(
            tickets,
            ZERO_SHOT_PROMPT,
            token_budget=demo_budget,
            run_name=f"ZERO-SHOT — BUDGET {demo_budget}",
        )
        budget_path = OUTPUT_DIR / "triage_budget_demo.csv"
        budget_results.to_csv(budget_path, index=False)
        budget_count = (
            budget_results["status"] == "budget_exhausted"
        ).sum()
        print(f"Demo budget lowered to {demo_budget}.")

    print(f"Budget demo output: {budget_path}")
    print(f"Budget-exhausted tickets: {budget_count}")

    # Final required run: few-shot, large budget.
    final_results, final_tokens = run_pipeline(
        tickets,
        FEW_SHOT_PROMPT,
        token_budget=FINAL_TOKEN_BUDGET,
        run_name=f"FEW-SHOT — FINAL BUDGET {FINAL_TOKEN_BUDGET}",
    )

    routed = make_routing_results(final_results, segments)
    final_path = OUTPUT_DIR / "triage_results.csv"
    routed.to_csv(final_path, index=False)

    print("\n" + "=" * 72)
    print("PART C OUTPUTS")
    print("=" * 72)
    print(f"Saved: {budget_path}")
    print(f"Saved: {final_path}")
    print(f"Final row count: {len(routed)}")
    print("Final routing counts:")
    print(routed["route"].value_counts().to_string())
    print("Final status counts:")
    print(routed["status"].value_counts().to_string())

    # Acceptance checks.
    assert len(routed) == len(tickets) == 164
    assert budget_count > 0, (
        "Budget demo did not exhaust. Lower TOKEN_BUDGET and rerun."
    )
    assert not contains_unredacted_pii(
        "\n".join(tickets["redacted_text"].tolist())
    )


if __name__ == "__main__":
    main()
