
from pathlib import Path
from datetime import datetime, timezone
import json
import sqlite3
import time
import uuid

import pandas as pd
import mock_llm

from part_d_policy_rag import (
    load_policy_chunks,
    build_retriever,
    retrieve,
)

ROOT = Path(__file__).resolve().parent
DB_FILE = ROOT / "data" / "instamart.db"
SCENARIOS_FILE = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"
LEDGER_FILE = OUTPUT_DIR / "refund_ledger.jsonl"
RUNS_FILE = OUTPUT_DIR / "agent_runs.jsonl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

# Load the same policy chunks and TF-IDF retriever used in Part D RAG.
CHUNKS = load_policy_chunks(overlap=10)
VECTORIZER, CHUNK_MATRIX = build_retriever(CHUNKS)

SYSTEM_PROMPT = """
AGENT
You are an Instamart refund agent. Use the supplied tools to check orders
and policies before deciding. Never invent an order or refund amount.
Only completed orders delivered in more than 45 minutes qualify.
Refunds up to INR 1000 may be issued automatically. Refunds above INR 1000
must be escalated to a human and must not be issued automatically.
Cancelled and on-time orders must not receive refunds.
If a tool reports an error, respect it. Return a concise final answer
beginning with REFUND:, ESCALATE:, DENIED:, or NOT_FOUND: as appropriate.
""".strip()


# ------------------------------------------------------------
# TOOL SPECIFICATIONS
# ------------------------------------------------------------

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order in the Instamart SQLite database.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Retrieve the three most relevant policy chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Calculate a refund for a completed order delivered late.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an eligible automatic refund of INR 1000 or less.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record an eligible refund requiring human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]

# Some mock implementations expect a flat tool specification.
# Keep the name at the top level, as required by this assignment.
TOOL_MAP = {}


# ------------------------------------------------------------
# GENERAL HELPERS
# ------------------------------------------------------------

def json_line_append(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_order_row(order_id):
    with sqlite3.connect(DB_FILE) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            LIMIT 1
            """,
            (order_id,),
        ).fetchone()

    return dict(row) if row else None


def get_order(order_id):
    row = get_order_row(order_id)
    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row["order_id"],
        "status": row["status"],
        "order_amount": (
            float(row["order_amount"])
            if row["order_amount"] is not None else None
        ),
        "delivery_minutes": (
            float(row["delivery_minutes"])
            if row["delivery_minutes"] is not None else None
        ),
    }


def search_policy(query):
    results = retrieve(
        query,
        k=3,
        chunks=CHUNKS,
        vectorizer=VECTORIZER,
        matrix=CHUNK_MATRIX,
    )
    return {"results": results}


def calculate_refund(order_amount, delivery_minutes):
    amount = float(order_amount)
    minutes = float(delivery_minutes)

    if amount < 0 or minutes < 0:
        return {"error": "invalid order amount or delivery time"}

    refund = round(amount, 2) if minutes > 45 else 0.0
    return {"refund_inr": refund}


def already_recorded(order_id, tool_name):
    for entry in read_jsonl(LEDGER_FILE):
        if (
            entry.get("order_id") == order_id
            and entry.get("tool_name") == tool_name
        ):
            return True
    return False


def record_refund_action(order_id, refund_inr, tool_name):
    """Validate the order and apply idempotency before writing the ledger."""
    order = get_order_row(order_id)

    if order is None:
        return {"error": "order not found"}

    if str(order["status"]).lower() != "completed":
        return {"error": "only completed orders can be refunded"}

    if order["delivery_minutes"] is None or float(order["delivery_minutes"]) <= 45:
        return {"error": "order is not eligible for a late-delivery refund"}

    expected_amount = round(float(order["order_amount"]), 2)
    requested_amount = round(float(refund_inr), 2)

    if abs(expected_amount - requested_amount) > 0.01:
        return {"error": "refund amount does not match the order amount"}

    if tool_name == "issue_refund" and requested_amount > 1000:
        return {"error": "refund above INR 1000 requires human approval"}

    if tool_name == "escalate_to_human" and requested_amount <= 1000:
        return {"error": "this refund does not require high-value escalation"}

    # Do not append a second ledger entry for the same action and order.
    if already_recorded(order_id, tool_name):
        return {
            "refund_inr": requested_amount,
            "already_recorded": True,
        }

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": requested_amount,
    }
    json_line_append(LEDGER_FILE, entry)

    return {"refund_inr": requested_amount}


def issue_refund(order_id, refund_inr):
    return record_refund_action(order_id, refund_inr, "issue_refund")


def escalate_to_human(order_id, refund_inr):
    return record_refund_action(order_id, refund_inr, "escalate_to_human")


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


# ------------------------------------------------------------
# MODEL-RESPONSE COMPATIBILITY HELPERS
# ------------------------------------------------------------

def field_value(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def parse_arguments(arguments):
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        return json.loads(arguments)
    return {}


def extract_tool_calls(message):
    calls = field_value(message, "tool_calls", None) or []
    extracted = []

    for call in calls:
        function = field_value(call, "function", None)

        # Support the assignment's flat format and standard function format.
        name = field_value(call, "name", None)
        arguments = field_value(call, "arguments", None)

        if function is not None:
            name = name or field_value(function, "name", None)
            arguments = (
                arguments
                if arguments is not None
                else field_value(function, "arguments", None)
            )

        extracted.append({
            "id": field_value(call, "id", str(uuid.uuid4())),
            "name": name,
            "arguments": parse_arguments(arguments),
        })

    return extracted


def call_model(messages):
    response = mock_llm.chat(messages, tools=TOOLS_SPEC)
    choice = response.choices[0]
    message = choice.message
    token_count = int(
        field_value(
            field_value(response, "usage", {}),
            "total_tokens",
            0,
        ) or 0
    )
    return message, token_count


# ------------------------------------------------------------
# CHECKPOINTS AND REACT LOOP
# ------------------------------------------------------------

def checkpoint_path(run_id):
    return STATE_DIR / f"{run_id}.json"


def save_checkpoint(state):
    checkpoint_path(state["run_id"]).write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_checkpoint(run_id):
    path = checkpoint_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def status_from_answer(answer_text):
    if not answer_text:
        return "FALLBACK"
    return answer_text.split(":", 1)[0].strip().upper()


def fallback_result(state, status, summary):
    final_answer = f"{status}: {summary}"
    state["status"] = status
    state["final_answer"] = final_answer
    state["finished"] = True
    save_checkpoint(state)

    return {
        "status": status,
        "final_answer": final_answer,
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "run_id": state["run_id"],
    }


def run_state(
    state,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None,
):
    started = time.monotonic()

    while state["llm_calls"] < max_iterations:
        # Required timeout check before each model call.
        if time.monotonic() - started >= time_limit_seconds:
            return fallback_result(
                state,
                "FALLBACK_TIMEOUT",
                "Please hand this request to a human agent; the time limit was reached.",
            )

        message, tokens = call_model(state["messages"])
        state["llm_calls"] += 1
        state["tokens_used"] += tokens

        # If the call exceeds the budget, do not execute its requested tools.
        if state["tokens_used"] > token_limit:
            return fallback_result(
                state,
                "FALLBACK_TOKEN_LIMIT",
                "Please hand this request to a human agent; the token limit was reached.",
            )

        tool_calls = extract_tool_calls(message)

        if not tool_calls:
            final_answer = field_value(message, "content", "") or ""
            state["status"] = status_from_answer(final_answer)
            state["final_answer"] = final_answer
            state["finished"] = True
            save_checkpoint(state)

            return {
                "status": state["status"],
                "final_answer": final_answer,
                "steps": state["steps"],
                "llm_calls": state["llm_calls"],
                "tokens_used": state["tokens_used"],
                "run_id": state["run_id"],
            }

        # If the last permitted model call still asks for a tool, stop
        # without executing that tool.
        if state["llm_calls"] >= max_iterations:
            return fallback_result(
                state,
                "FALLBACK_MAX_ITERATIONS",
                "Please hand this request to a human agent; the step limit was reached.",
            )

        assistant_calls = []
        for call in tool_calls:
            assistant_calls.append({
                "id": call["id"],
                "name": call["name"],
                "arguments": call["arguments"],
            })

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_calls,
        })

        for call in tool_calls:
            name = call["name"]
            arguments = call["arguments"]

            if name not in TOOL_MAP:
                observation = {"error": f"unknown tool: {name}"}
            else:
                try:
                    observation = TOOL_MAP[name](**arguments)
                except Exception as exc:
                    observation = {
                        "error": f"tool execution failed: {type(exc).__name__}: {exc}"
                    }

            observation_json = json.dumps(
                observation,
                ensure_ascii=False,
                default=str,
            )

            state["messages"].append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": observation_json,
            })

            state["steps"].append({
                "step": len(state["steps"]) + 1,
                "tool_name": name,
                "arguments": arguments,
                "observation": observation,
            })

            state["executed_tools"].append(name)

            # Save immediately after every tool observation.
            save_checkpoint(state)

            if (
                crash_after_tools is not None
                and len(state["executed_tools"]) == crash_after_tools
            ):
                raise RuntimeError(
                    f"SIMULATED CRASH after tool execution {crash_after_tools}"
                )

    return fallback_result(
        state,
        "FALLBACK_MAX_ITERATIONS",
        "Please hand this request to a human agent; the step limit was reached.",
    )


def new_state(user_message, scenario_id, run_id=None):
    run_id = run_id or f"{scenario_id}_{uuid.uuid4().hex[:8]}"

    return {
        "run_id": run_id,
        "scenario_id": str(scenario_id),
        "user_message": str(user_message),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(user_message)},
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "steps": [],
        "executed_tools": [],
        "finished": False,
        "status": None,
        "final_answer": None,
    }


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None,
):
    state = new_state(user_message, scenario_id, run_id)
    save_checkpoint(state)

    return run_state(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after_tools=crash_after_tools,
    )


def resume_agent(run_id, max_iterations=6, token_limit=6000,
                 time_limit_seconds=10):
    state = load_checkpoint(run_id)

    if state.get("finished"):
        return {
            "status": state["status"],
            "final_answer": state["final_answer"],
            "steps": state["steps"],
            "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"],
            "run_id": state["run_id"],
        }

    return run_state(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
    )


# ------------------------------------------------------------
# SCENARIO EXECUTION AND DEMONSTRATIONS
# ------------------------------------------------------------

def find_column(frame, candidates):
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def append_run_log(scenario_id, result):
    json_line_append(RUNS_FILE, {
        "scenario_id": str(scenario_id),
        "status": result["status"],
        "final_answer": result["final_answer"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
        "steps": result["steps"],
    })


def result_amount(result):
    for step in result["steps"]:
        if step["tool_name"] in ("issue_refund", "escalate_to_human"):
            observation = step["observation"]
            if "refund_inr" in observation:
                return observation["refund_inr"]
    return ""


def main():
    if not DB_FILE.exists():
        raise FileNotFoundError(f"Database not found: {DB_FILE}")
    if not SCENARIOS_FILE.exists():
        raise FileNotFoundError(f"Scenario CSV not found: {SCENARIOS_FILE}")

    scenarios = pd.read_csv(SCENARIOS_FILE, keep_default_na=False)
    id_col = find_column(scenarios, ["scenario_id", "id"])
    message_col = find_column(
        scenarios, ["user_message", "message", "customer_message", "request"]
    )

    if id_col is None or message_col is None:
        raise ValueError(
            "Could not identify scenario ID and user-message columns. "
            f"Columns found: {list(scenarios.columns)}"
        )

    mock_llm.reset()

    # Keep existing logs and ledger; idempotency checks prevent duplicate
    # entries when this script is run again.
    results = []

    for _, row in scenarios.iterrows():
        scenario_id = str(row[id_col])
        user_message = str(row[message_col])

        result = run_agent(
            user_message,
            scenario_id,
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10,
        )

        append_run_log(scenario_id, result)
        results.append({
            "scenario_id": scenario_id,
            "status": result["status"],
            "refund_inr": result_amount(result),
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "final_answer": result["final_answer"],
            "run_id": result["run_id"],
        })

        print(
            f"{scenario_id}: {result['status']} | "
            f"calls={result['llm_calls']} | "
            f"tokens={result['tokens_used']} | "
            f"refund={result_amount(result)}"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "agent_scenario_results.csv", index=False)

    print("\nSeven-scenario summary:")
    print(results_df[
        ["scenario_id", "status", "refund_inr", "llm_calls", "tokens_used"]
    ].to_string(index=False))

    # Required token-limit demonstration.
    s1 = scenarios[scenarios[id_col].astype(str) == "S1"]
    if not s1.empty:
        s1_message = str(s1.iloc[0][message_col])
        token_demo = run_agent(
            s1_message,
            "S1_TOKEN_LIMIT_DEMO",
            token_limit=200,
            time_limit_seconds=10,
        )
        append_run_log("S1_TOKEN_LIMIT_DEMO", token_demo)
        print("\nToken-limit demonstration:", token_demo["status"])

        # Required timeout demonstration: zero seconds means stop before
        # the first model call.
        timeout_demo = run_agent(
            s1_message,
            "S1_TIMEOUT_DEMO",
            token_limit=6000,
            time_limit_seconds=0,
        )
        append_run_log("S1_TIMEOUT_DEMO", timeout_demo)
        print("Timeout demonstration:", timeout_demo["status"])

        # Required crash/resume demonstration. Use a unique run ID so a
        # prior checkpoint does not interfere with this demonstration.
        crash_run_id = f"S1_CRASH_RESUME_{uuid.uuid4().hex[:8]}"

        try:
            run_agent(
                s1_message,
                "S1_CRASH_RESUME",
                token_limit=6000,
                time_limit_seconds=10,
                run_id=crash_run_id,
                crash_after_tools=2,
            )
        except RuntimeError as exc:
            if "SIMULATED CRASH" not in str(exc):
                raise
            print("\nSimulated crash:", exc)

        resumed = resume_agent(
            crash_run_id,
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10,
        )
        append_run_log("S1_CRASH_RESUME", resumed)

        print("Resumed result:", resumed["status"])
        print("Tools executed across crash and resume:")
        print(" -> ".join(resumed["executed_tools"])
              if "executed_tools" in resumed
              else " -> ".join(step["tool_name"] for step in resumed["steps"]))
        print("Refund:", result_amount(resumed))

    print("\nOutputs:")
    print(f"- {OUTPUT_DIR / 'agent_scenario_results.csv'}")
    print(f"- {RUNS_FILE}")
    print(f"- {LEDGER_FILE}")
    print(f"- {STATE_DIR}")


if __name__ == "__main__":
    main()
