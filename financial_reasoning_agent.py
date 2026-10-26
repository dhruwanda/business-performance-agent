"""Stage 1: a read-only LangChain agent over prepared company financial data.

No discovery, PDF processing, web search, data edits, or custom calculations yet.
This file deliberately keeps the data tools independent of LangChain, so they
can be inspected and tested without an API key or an LLM request.

Example (from the repository root):
    python financial_reasoning_agent.py --run-dir data/runs/<run-name>
    python financial_reasoning_agent.py --run-dir data/runs/<run-name> \
        --question "How did revenue and EBITDA margins change?"

Dependencies for an existing Google AI Studio key:
    pip install "langchain[google-genai]"
The key may be in GOOGLE_API_KEY, GEMINI_API_KEY, or local .streamlit/secrets.toml.
Set LLM_MODEL to a LangChain provider:model identifier if using another model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import tomllib
from typing import Any, Literal


PREPARED_FILENAME = "final_prepared_consolidated_facts.csv"
CALCULATED_FILENAME = "calculated_financial_metrics.csv"
SUMMARY_FILENAME = "financial_calculation_summary.json"

SYSTEM_PROMPT = """You are a financial performance analyst investigating company results.
Use the available tools for all company-specific financial numbers. First find
the exact metric keys, then retrieve the relevant values and calculation inputs.
Do not invent, recompute, or estimate figures; a separate calculation tool will
be added in the next stage. Do not conflate reported, adjusted, consolidated,
standalone, quarterly, cumulative, or annual figures. Distinguish percentage
change from percentage-point change. Keep source URLs and PDF page numbers
alongside material findings. When evidence is conflicting, incomplete, or
unavailable, say so clearly. If a question requires management commentary,
external news, segment explanations, or documents not available in these tools,
state that further investigation is required: those tools are not connected yet.
Consult get_data_quality_flags when discrepancies or reliability matter.
Keep answers concise and explain what the numbers do and do not establish.
"""


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _short_sources(row: dict[str, str], source: str) -> list[dict[str, Any]]:
    """Keep a compact but traceable set of source references in tool responses."""
    if source == "reported":
        found = []
        if row.get("source_url"):
            found.append({"url": row["source_url"], "page": row.get("page", "")})
        try:
            supporting = json.loads(row.get("supporting_sources") or "[]")
        except (ValueError, TypeError):
            supporting = []
        if isinstance(supporting, list):
            for item in supporting:
                if isinstance(item, dict) and item.get("url"):
                    found.append({"url": item["url"], "page": item.get("page", "")})
    else:
        found = []
        try:
            inputs = json.loads(row.get("input_facts_json") or "[]")
        except (ValueError, TypeError):
            inputs = []
        if isinstance(inputs, list):
            for item in inputs:
                if isinstance(item, dict) and item.get("source_url"):
                    found.append({"url": item["source_url"], "page": item.get("page", ""),
                                  "input_metric": item.get("metric_key", "")})
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for item in found:
        marker = (str(item.get("url", "")), str(item.get("page", "")),
                  str(item.get("input_metric", "")))
        if marker not in seen:
            seen.add(marker)
            unique.append(item)
    return unique[:8]


class FinancialDataStore:
    """Whitelisted, read-only queries over one existing company run folder."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).resolve()
        self.all_prepared = _load_csv(self.run_dir / PREPARED_FILENAME)
        self.reported = [row for row in self.all_prepared
                         if row.get("usable_for_calculations", "").lower() in ("true", "1", "yes")]
        self.calculated = _load_csv(self.run_dir / CALCULATED_FILENAME)
        summary_path = self.run_dir / SUMMARY_FILENAME
        self.summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    def catalog(self, search: str = "", source: Literal["both", "reported", "calculated"] = "both",
                limit: int = 40) -> dict[str, Any]:
        """Discover exact metric keys, descriptions, families, and coverage."""
        if source not in ("both", "reported", "calculated"):
            return {"error": "source must be both, reported, or calculated"}
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for kind, rows in (("reported", self.reported), ("calculated", self.calculated)):
            if source not in ("both", kind):
                continue
            for row in rows:
                if search.lower() in (row.get("metric_key", "") + " " + row.get("metric_label", "")).lower():
                    grouped[(kind, row.get("metric_key", ""))].append(row)
        entries = []
        for (kind, key), rows in grouped.items():
            periods = sorted({str(r.get("period_end", "")) for r in rows if r.get("period_end")})
            entries.append({
                "source": kind, "metric_key": key,
                "label": rows[0].get("metric_label") or rows[0].get("metric") or key,
                "family": rows[0].get("family", ""),
                "observations": len(rows), "period_types": sorted({r.get("period_type", "") for r in rows}),
                "latest_period_end": periods[-1] if periods else "",
            })
        entries.sort(key=lambda x: (x["metric_key"], x["source"]))
        maximum = max(1, min(int(limit), 100))
        return {"matches": len(entries), "returned": len(entries[:maximum]), "metrics": entries[:maximum],
                "note": "Search again with a narrower phrase if results were truncated."}

    def values(self, metric_key: str, source: Literal["both", "reported", "calculated"] = "both",
               period_type: str = "", fiscal_year: int = 0, limit: int = 25) -> dict[str, Any]:
        """Return a financial time series without combining incompatible reporting bases."""
        if source not in ("both", "reported", "calculated"):
            return {"error": "source must be both, reported, or calculated"}
        matched = []
        for kind, rows in (("reported", self.reported), ("calculated", self.calculated)):
            if source not in ("both", kind):
                continue
            for row in rows:
                if row.get("metric_key") != metric_key:
                    continue
                if period_type and row.get("period_type") != period_type:
                    continue
                if fiscal_year and row.get("fiscal_year") != str(fiscal_year):
                    continue
                entry: dict[str, Any] = {
                    "source": kind, "metric_key": row.get("metric_key", ""),
                    "label": row.get("metric_label") or row.get("metric", ""),
                    "value": row.get("standard_value") if kind == "reported" else row.get("value"),
                    "unit": row.get("standard_unit") if kind == "reported" else row.get("unit"),
                    "period_type": row.get("period_type", ""), "fiscal_year": row.get("fiscal_year", ""),
                    "fiscal_quarter": row.get("fiscal_quarter", ""),
                    "period_start": row.get("period_start", ""), "period_end": row.get("period_end", ""),
                    "scope": row.get("standard_scope", ""), "basis": row.get("standard_basis", ""),
                    "currency": row.get("currency", ""), "sources": _short_sources(row, kind),
                }
                if kind == "calculated":
                    entry.update({"family": row.get("family", ""), "formula": row.get("formula", ""),
                                  "comparison_period_end": row.get("comparison_period_end", ""),
                                  "notes": row.get("notes", "")})
                else:
                    entry.update({"reconciliation_status": row.get("reconciliation_status", ""),
                                  "annual_quarter_check": row.get("annual_quarter_check", "")})
                matched.append(entry)
        matched.sort(key=lambda r: (r["period_end"], r["period_type"], r["scope"], r["basis"]), reverse=True)
        maximum = max(1, min(int(limit), 60))
        return {"metric_key": metric_key, "matches": len(matched),
                "returned": len(matched[:maximum]), "observations": matched[:maximum],
                "note": "Latest periods first. No missing values were synthesized; no reporting variants were combined."}

    def quality(self, limit: int = 20) -> dict[str, Any]:
        """Give source-conflict and annual/quarter discrepancy context, without resolving it."""
        conflicts = []
        inconsistencies = []
        for row in self.all_prepared:
            if row.get("usable_for_calculations", "").lower() not in ("true", "1", "yes"):
                conflicts.append({
                    "metric_key": row.get("metric_key", ""), "period": row.get("period", ""),
                    "period_end": row.get("period_end", ""), "value": row.get("standard_value", ""),
                    "unit": row.get("standard_unit", ""), "basis": row.get("standard_basis", ""),
                    "source_url": row.get("source_url", ""), "page": row.get("page", ""),
                    "reason": row.get("reconciliation_status", ""),
                })
            check = row.get("annual_quarter_check", "")
            if check in ("material_mismatch", "mismatch_ambiguous_quarters", "possible_match_ambiguous_quarters"):
                inconsistencies.append({
                    "metric_key": row.get("metric_key", ""), "period": row.get("period", ""),
                    "period_end": row.get("period_end", ""), "check": check,
                    "difference": row.get("annual_quarter_difference", ""),
                    "details": row.get("annual_quarter_details", ""),
                    "source_url": row.get("source_url", ""),
                })
        maximum = max(1, min(int(limit), 100))
        return {"accepted_facts": len(self.reported), "calculated_metrics": len(self.calculated),
                "source_conflict_rows": len(conflicts), "annual_quarter_flag_rows": len(inconsistencies),
                "source_conflicts": conflicts[:maximum], "annual_quarter_flags": inconsistencies[:maximum],
                "note": "These are observations/flags, not necessarily distinct economic discrepancies. The agent must not silently choose a winning source."}


def create_financial_agent(store: FinancialDataStore, model_id: str):
    """Expose the three safe Python data functions to LangChain."""
    try:
        from langchain.agents import create_agent
        from langchain.tools import tool
    except ImportError as exc:
        raise RuntimeError('Install LangChain first: pip install "langchain[google-genai]"') from exc

    @tool
    def list_available_metrics(search: str = "", source: Literal["both", "reported", "calculated"] = "both",
                               limit: int = 40) -> str:
        """Find exact financial metric keys and available period coverage. Search by name before requesting a time series."""
        return json.dumps(store.catalog(search, source, limit), ensure_ascii=False)

    @tool
    def get_metric_values(metric_key: str, source: Literal["both", "reported", "calculated"] = "both",
                          period_type: str = "", fiscal_year: int = 0, limit: int = 25) -> str:
        """Read reported or deterministic calculated values for one exact metric key, with periods, basis, formulas and original source references."""
        return json.dumps(store.values(metric_key, source, period_type, fiscal_year, limit), ensure_ascii=False)

    @tool
    def get_data_quality_flags(limit: int = 20) -> str:
        """Inspect unresolved source conflicts and mismatches between annual figures and quarterly totals. Do not resolve conflicts automatically."""
        return json.dumps(store.quality(limit), ensure_ascii=False)

    if model_id.startswith("google_genai:") and not os.getenv("GOOGLE_API_KEY"):
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            local_secrets = Path(".streamlit/secrets.toml")
            if local_secrets.is_file():
                with local_secrets.open("rb") as f:
                    settings = tomllib.load(f)
                key = settings.get("GOOGLE_API_KEY") or settings.get("GEMINI_API_KEY", "")
        if key:
            os.environ["GOOGLE_API_KEY"] = key

    model: Any = model_id
    if model_id.startswith("google_genai:"):
        # Keep simple financial lookups responsive and fail instead of waiting
        # through several minutes of silent retries on a stalled API call.
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model=model_id.split(":", 1)[1],
            thinking_level="low",
            timeout=45,
            max_retries=0,
        )

    return create_agent(
        model=model,
        tools=[list_available_metrics, get_metric_values, get_data_quality_flags],
        system_prompt=SYSTEM_PROMPT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 LangChain financial reasoning prototype (read-only).")
    parser.add_argument("--run-dir", type=Path, required=True, help="Folder containing both prepared and calculated CSVs")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "google_genai:gemini-3.8-flash"),
                        help="LangChain provider:model ID; or set LLM_MODEL")
    parser.add_argument("--question", help="Ask one question and exit; omit for an interactive session")
    args = parser.parse_args()

    store = FinancialDataStore(args.run_dir)
    print(f"Loaded {len(store.reported)} accepted reported facts and {len(store.calculated)} calculations.")
    agent = create_financial_agent(store, args.model)

    # Keep the full tool-call history across turns. No external persistence yet.
    messages: list[Any] = []
    while True:
        if args.question:
            question = args.question
        else:
            try:
                question = input("\nYour question (or 'exit'): ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if question.lower() in ("exit", "quit"):
                break
            if not question:
                continue
        messages.append({"role": "user", "content": question})
        print("Contacting the LLM (up to ~45 seconds per model request)...", flush=True)
        # Streaming step updates avoids a silent wait for the *whole* agent run.
        # Retain the original AI/tool messages to preserve tool-call state.
        new_messages = list(messages)
        for update in agent.stream(
            {"messages": messages}, config={"recursion_limit": 10}, stream_mode="updates"
        ):
            for node, payload in update.items():
                for message in payload.get("messages", []):
                    new_messages.append(message)
                    if node == "model" and getattr(message, "tool_calls", None):
                        for call in message.tool_calls:
                            print(f"LLM requested: {call['name']}({json.dumps(call.get('args', {}), ensure_ascii=False)})", flush=True)
                    elif node == "tools":
                        print(f"Tool finished: {getattr(message, 'name', 'tool')}", flush=True)
                    elif node == "model":
                        print("LLM finished answering.", flush=True)
        messages = new_messages
        answer = messages[-1].content
        if isinstance(answer, list):
            answer = "\n".join(block.get("text", "") for block in answer
                               if isinstance(block, dict) and block.get("type") == "text")
        print("\nAgent:", answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False))
        if args.question:
            break


if __name__ == "__main__":
    main()
