"""Streamlit interface for company financial analysis and report processing."""

from collections import Counter
from dataclasses import asdict, replace
from datetime import date, datetime
from io import StringIO
import csv
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
import streamlit as st

from discovery_workflow import Document, discover_documents
from financial_extraction import (
    ExtractionResult,
    FinancialFact,
    extract_document,
    select_documents,
    verify_unknown_periods,
)


st.set_page_config(
    page_title="Business Performance Analyst 🤖",
    layout="centered",
)


# ---------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------

def gemini_api_key() -> str:
    """Use an existing API key without displaying it in the app."""
    if os.getenv("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]

    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except FileNotFoundError:
        return ""


# ---------------------------------------------------------------------
# Saved extraction progress
# ---------------------------------------------------------------------

def load_extraction_progress(
    website: str,
    fiscal_end_month: int,
) -> dict | None:
    """Resume the latest saved run, even after a browser refresh."""

    current = st.session_state.get("extraction_progress")

    if current and (
        current["website"],
        current["fiscal_end_month"],
    ) == (
        website,
        fiscal_end_month,
    ):
        return current

    # Support progress saved by the earlier version of the app.
    if (
        st.session_state.get("extraction_website") == website
        and st.session_state.get("extraction_run_dir")
        and st.session_state.get("extraction_output")
    ):
        earlier, _ = st.session_state["extraction_output"]

        restored = {
            "website": website,
            "fiscal_end_month": fiscal_end_month,
            "run_dir": st.session_state["extraction_run_dir"],
            "completed": {
                item.document.url: item
                for item in earlier
            },
            "failures": {},
        }

        save_extraction_progress(restored)

        st.session_state["extraction_progress"] = restored
        st.session_state.pop("extraction_output", None)

        return restored

    runs = Path("data/runs")

    if runs.is_dir():
        for folder in sorted(runs.iterdir(), reverse=True):
            manifest = folder / "extraction_manifest.json"

            if not manifest.is_file():
                continue

            try:
                saved = json.loads(
                    manifest.read_text(encoding="utf-8")
                )

                if (
                    saved.get("website"),
                    saved.get("fiscal_end_month"),
                ) != (
                    website,
                    fiscal_end_month,
                ):
                    continue

                completed = {}

                for url, item in saved.get("completed", {}).items():
                    completed[url] = ExtractionResult(
                        document=Document(**item["document"]),
                        saved_pdf=item["saved_pdf"],
                        page_count=item["page_count"],
                        pages_sent=item["pages_sent"],
                        facts=[
                            FinancialFact(**fact)
                            for fact in item["facts"]
                        ],
                        issues=item["issues"],
                    )

                saved["completed"] = completed

                st.session_state["extraction_progress"] = saved

                return saved

            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
            ):
                continue

    st.session_state.pop("extraction_progress", None)

    return None


def save_extraction_progress(progress: dict) -> None:
    """Save source-attributed financial facts after every batch."""

    folder = Path(progress["run_dir"])
    folder.mkdir(parents=True, exist_ok=True)

    facts = [
        asdict(fact)
        for result in progress["completed"].values()
        for fact in result.facts
    ]

    csv_path = folder / "consolidated_facts.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as output:
        headers = list(FinancialFact.__dataclass_fields__)

        writer = csv.DictWriter(
            output,
            fieldnames=headers,
        )

        writer.writeheader()
        writer.writerows(facts)

    manifest = {
        **{
            key: value
            for key, value in progress.items()
            if key != "completed"
        },
        "completed": {
            url: asdict(result)
            for url, result in progress["completed"].items()
        },
    }

    temp = folder / "extraction_manifest.tmp"

    temp.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temp.replace(
        folder / "extraction_manifest.json"
    )


def process_pdf_batch(
    documents: list[Document],
    progress: dict,
) -> None:
    """Process PDFs and save progress after each document."""

    try:
        model = os.getenv("GEMINI_MODEL") or st.secrets.get(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        )
    except FileNotFoundError:
        model = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        )

    indicator = st.progress(
        0,
        text="Preparing PDF extraction...",
    )

    for index, document in enumerate(documents, 1):
        indicator.progress(
            (index - 1) / len(documents),
            text=(
                f"Reading {index}/{len(documents)}: "
                f"{document.title}"
            ),
        )

        try:
            result = extract_document(
                document,
                api_key=gemini_api_key(),
                model=model,
                cache_dir=Path(progress["run_dir"]),
                shared_cache_dir=Path("data/cache"),
            )

            progress["completed"][document.url] = result
            progress["failures"].pop(
                document.url,
                None,
            )

        except (
            requests.RequestException,
            RuntimeError,
            ValueError,
            OSError,
        ) as error:
            progress["failures"][document.url] = (
                f"{document.title}: {error}"
            )

        save_extraction_progress(progress)

    indicator.progress(
        1.0,
        text="Batch completed",
    )

    st.session_state["extraction_progress"] = progress


# ---------------------------------------------------------------------
# Identify completed company analyses
# ---------------------------------------------------------------------

def website_domain(url: str) -> str:
    """Treat www.example.com and example.com as the same website."""

    url = url.strip().lower()

    if not url:
        return ""

    if "://" not in url:
        url = "https://" + url

    host = urlparse(url).hostname or ""

    if host.startswith("www."):
        host = host[4:]

    return host


def completed_runs_for(website: str) -> list[Path]:
    """Find completed financial datasets for a company."""

    root = Path("data/runs")
    domain = website_domain(website)

    if not root.is_dir() or not domain:
        return []

    matched = []

    for folder in root.iterdir():
        if not folder.is_dir():
            continue

        prepared = (
            folder / "final_prepared_consolidated_facts.csv"
        )
        calculated = (
            folder / "calculated_financial_metrics.csv"
        )
        manifest = (
            folder / "extraction_manifest.json"
        )

        if not prepared.is_file():
            continue

        if not calculated.is_file():
            continue

        if not manifest.is_file():
            continue

        try:
            saved = json.loads(
                manifest.read_text(encoding="utf-8")
            )

            saved_website = saved.get("website", "")

        except (
            OSError,
            ValueError,
            TypeError,
            AttributeError,
        ):
            continue

        if website_domain(saved_website) == domain:
            matched.append(folder)

    return sorted(matched, reverse=True)


def readable_run_name(folder: Path) -> str:
    """Display a readable date instead of an internal run identifier."""

    try:
        when = datetime.strptime(
            folder.name[:19],
            "%Y-%m-%d_%H-%M-%S",
        )

        return (
            f"Analysis from "
            f"{when:%d %b %Y, %I:%M %p}"
        )

    except ValueError:
        return "Saved analysis"


# ---------------------------------------------------------------------
# Conversational financial reasoning
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_qa_agent(
    run_dir: str,
    data_version: tuple[int, int],
):
    """Reuse the agent and document index between questions."""

    from document_search import DocumentSearchStore

    from financial_reasoning_agent import (
        FinancialDataStore,
        create_financial_agent,
    )

    key = gemini_api_key()

    if key:
        os.environ.setdefault(
            "GOOGLE_API_KEY",
            key,
        )

    financial_store = FinancialDataStore(run_dir)
    documents = DocumentSearchStore(run_dir)

    model = os.getenv(
        "LLM_MODEL",
        "google_genai:gemini-3.6-flash",
    )

    return create_financial_agent(
        financial_store,
        model,
        documents=documents,
    )


def answer_text(content) -> str:
    """Convert an LLM response into displayable text."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
            )
        )

    return str(content)


def display_answer(answer: str) -> None:
    """
    Display the analysis separately from its source links.

    Direct document URLs are collected under References.
    """

    pattern = re.compile(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)"
    )

    references = []
    seen = set()

    # Collect Markdown links.
    for title, url in pattern.findall(answer):
        if url not in seen:
            references.append((title, url))
            seen.add(url)

    # Collect any plain URLs.
    for url in re.findall(
        r"https?://[^\s)<>\]]+",
        answer,
    ):
        url = url.rstrip(".,;")

        if url not in seen:
            references.append(
                ("Open source document", url)
            )
            seen.add(url)

    # Remove a separate references heading if the LLM supplied one.
    body = re.split(
        (
            r"(?im)^\s*#{1,6}\s*"
            r"(?:Sources?|References?)\s*:?\s*$"
        ),
        answer,
        maxsplit=1,
    )[0]

    # Remove standalone source lines.
    body = "\n".join(
        line
        for line in body.splitlines()
        if not re.match(
            (
                r"^\s*[\*(]*\s*"
                r"(?:Sources?|References?)\s*:"
            ),
            line,
            flags=re.IGNORECASE,
        )
    )

    # Keep linked text in the answer, but put URLs in References.
    body = pattern.sub(
        lambda match: match.group(1),
        body,
    )

    body = re.sub(
        r"https?://[^\s)<>\]]+",
        "",
        body,
    ).strip()

    answer_tab, references_tab = st.tabs(
        ["Answer", "References"]
    )

    with answer_tab:
        st.markdown(
            body or "No answer available."
        )

    with references_tab:
        if references:
            for title, url in references:
                st.markdown(
                    f"- [{title}]({url})"
                )
        else:
            st.write(
                "No direct source links were included."
            )

def message_column(role: str):
    """Position user messages on the right and agent messages on the left."""
    if role == "user":
        _, right = st.columns([1, 3])
        return right

    left, _ = st.columns([3, 1])
    return left

def show_previous(entry: dict) -> None:
    """Display conversation history as left/right message bubbles."""

    with message_column(entry["role"]):
        with st.container(border=True):

            if entry["role"] == "user":
                st.markdown("**You**")
                st.write(entry["content"])

            else:
                st.markdown("**Agent**")

                if entry.get("tool_log"):
                    st.code(
                        "\n".join(entry["tool_log"]),
                        language="text",
                    )

                if entry.get("successful", True):
                    display_answer(entry["content"])
                else:
                    st.text(entry["content"])

# ---------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------

st.title("Business Performance Analyst")

st.write(
    "Enter a company website and a question. "
    "I'll examine its financial reports, explore performance trends "
    "and business drivers, and provide an evidence-backed answer."
)


# ---------------------------------------------------------------------
# Company website and question
# ---------------------------------------------------------------------

with st.form("company_analysis_form"):
    website = st.text_input(
        "Company website",
        value="https://honasa.in",
        placeholder="https://www.company.com",
        help=(
            "Enter the company's official website "
            "or Investor Relations page."
        ),
    )

    question = st.text_area(
        "What would you like to know?",
        placeholder=(
            "For example: What drove the change "
            "in EBITDA margin this year?"
        ),
        height=90,
    )

    ask_clicked = st.form_submit_button(
        "Ask about performance",
        type="primary",
    )

    submitted = st.form_submit_button(
        "Find or refresh financial reports"
    )


# ---------------------------------------------------------------------
# Existing company analysis
# ---------------------------------------------------------------------

matching_runs = completed_runs_for(website)

if matching_runs:

    company = (
        website_domain(website)
        .split(".")[0]
        .capitalize()
    )

    st.markdown(
        f"**{company} · Analysis ready**"
    )

    if len(matching_runs) > 1:
        selected_run = st.selectbox(
            "Saved analysis",
            matching_runs,
            format_func=readable_run_name,
        )
    else:
        selected_run = matching_runs[0]

    run_path = str(selected_run.resolve())

    data_version = (
        (
            selected_run
            / "final_prepared_consolidated_facts.csv"
        ).stat().st_mtime_ns,
        (
            selected_run
            / "calculated_financial_metrics.csv"
        ).stat().st_mtime_ns,
    )

    run_key = (
        run_path,
        data_version,
    )

    # A different company or dataset starts a new conversation.
    if st.session_state.get("qa_run_key") != run_key:
        st.session_state["qa_run_key"] = run_key
        st.session_state["qa_history"] = []
        st.session_state["qa_display"] = []

    # Show earlier questions and responses in plain format.
    for entry in st.session_state["qa_display"]:
        show_previous(entry)

    # -------------------------------------------------------------
    # Ask the financial reasoning agent
    # -------------------------------------------------------------

    if ask_clicked:

        if not question.strip():
            st.write(
                "Enter a question to begin."
            )

        else:
            # User question: right-aligned bubble
            with message_column("user"):
                with st.container(border=True):
                    st.markdown("**You**")
                    st.write(question)

            # Agent response: left-aligned bubble
            with message_column("assistant"):
                agent_bubble = st.container(border=True)

            with agent_bubble:
                st.markdown("**Agent**")
                st.text("Contacting the LLM...")

                tool_log = []
                trace_display = st.empty()

            
            try:
                agent = load_qa_agent(
                    run_path,
                    data_version,
                )

                conversation = [
                    *st.session_state["qa_history"],
                    {
                        "role": "user",
                        "content": question,
                    },
                ]

                updated_history = list(conversation)

                def record_activity(line: str) -> None:
                    """Update the visible tool log as events arrive."""

                    tool_log.append(line)

                    trace_display.code(
                        "\n".join(tool_log),
                        language="text",
                    )

                # Stream LangGraph events so users can see the
                # exact tool calls and completions as they happen.
                for update in agent.stream(
                    {"messages": conversation},
                    config={"recursion_limit": 20},
                    stream_mode="updates",
                ):

                    for node, payload in update.items():

                        for message in payload.get(
                            "messages",
                            [],
                        ):

                            updated_history.append(message)

                            if node == "model":

                                if getattr(
                                    message,
                                    "tool_calls",
                                    None,
                                ):

                                    for call in message.tool_calls:
                                        args = json.dumps(
                                            call.get(
                                                "args",
                                                {},
                                            ),
                                            ensure_ascii=False,
                                        )

                                        record_activity(
                                            (
                                                f"LLM requested: "
                                                f"{call['name']}({args})"
                                            )
                                        )

                                else:
                                    record_activity(
                                        "LLM finished answering."
                                    )

                            elif node == "tools":
                                record_activity(
                                    (
                                        "Tool finished: "
                                        f"{getattr(message, 'name', 'tool')}"
                                    )
                                )

                if len(updated_history) == len(conversation):
                    raise RuntimeError(
                        "No response was returned by the agent."
                    )

                answer = answer_text(
                    updated_history[-1].content
                ).strip()

                if not answer:
                    raise RuntimeError(
                        "The agent returned an empty answer."
                    )

                # Save the full LangChain message history,
                # including tool calls, for follow-up questions.
                st.session_state["qa_history"] = updated_history

                successful = True

            except Exception as error:
                details = str(error)

                if (
                    "429" in details
                    and "RESOURCE_EXHAUSTED" in details
                ):
                    answer = (
                        "LLM API rate limit reached (429). "
                        "Please wait and retry."
                    )

                elif "503 UNAVAILABLE" in details:
                    answer = (
                        "LLM service is temporarily overloaded (503). "
                        "Please try again later."
                    )

                else:
                    first_line = (
                        details.splitlines()[0][:180]
                        if details.splitlines()
                        else "Unknown error"
                    )

                    answer = (
                        f"Analysis could not finish: {first_line}"
                    )

                successful = False

            # Store a separate display history.
            st.session_state["qa_display"].extend(
                [
                    {
                        "role": "user",
                        "content": question,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                        "tool_log": tool_log,
                        "successful": successful,
                    },
                ]
            )
            with agent_bubble:
                if successful:
                    display_answer(answer)
                else:
                    st.text(answer)


else:
    st.write(
        "No completed analysis is available for this company yet. "
        "Select 'Find or refresh financial reports' to begin."
    )

    if ask_clicked:
        st.write(
            "Financial data and calculations must be prepared "
            "before answering."
        )


# ---------------------------------------------------------------------
# Document discovery and financial data extraction
# ---------------------------------------------------------------------

# For a company with completed analysis, the processing interface
# stays hidden unless the user explicitly asks to refresh reports.
# For a new company, it appears automatically.

site_key = website_domain(website)

if st.session_state.get("reports_site") != site_key:
    st.session_state["reports_site"] = site_key
    st.session_state["reports_open"] = not bool(matching_runs)

if submitted:
    st.session_state["reports_open"] = True

show_details = (
    not bool(matching_runs)
    or st.session_state.get(
        "reports_open",
        False,
    )
)

if show_details:

    if matching_runs and st.button(
        "Hide report processing"
    ):
        st.session_state["reports_open"] = False
        st.rerun()

    # -------------------------------------------------------------
    # Discover financial documents
    # -------------------------------------------------------------

    if submitted:

        with st.spinner(
            "Finding financial reports..."
        ):

            try:
                st.session_state[
                    "discovery_result"
                ] = discover_documents(
                    website,
                    api_key=gemini_api_key(),
                )

                st.session_state[
                    "searched_website"
                ] = website

            except (
                requests.RequestException,
                RuntimeError,
                ValueError,
            ) as error:

                st.session_state.pop(
                    "discovery_result",
                    None,
                )

                st.error(
                    f"Couldn't complete discovery: {error}"
                )

    result = (
        st.session_state.get("discovery_result")
        if st.session_state.get("searched_website") == website
        else None
    )

    if result is not None:

        documents = result.documents

        with st.expander(
            f"Pages visited ({len(result.pages_visited)})"
        ):
            for page in result.pages_visited:
                st.write(page)

        if not documents:

            st.warning(
                "No matching PDF links found. "
                "Try entering the company's Investor Relations "
                "URL directly. Some websites require JavaScript "
                "or use other document formats."
            )

        else:

            counts = Counter(
                document.document_type
                for document in documents
            )

            st.success(
                f"Found {len(documents)} potential "
                "financial documents"
            )

            st.caption(
                " · ".join(
                    f"{kind}: {count}"
                    for kind, count in sorted(counts.items())
                )
            )

            missing_periods = sum(
                document.period == "Not identified"
                for document in documents
            )

            if missing_periods:
                st.info(
                    f"{missing_periods} document(s) have no "
                    "period identified from their links. "
                    "You can verify these reporting periods "
                    "before processing."
                )

            # -----------------------------------------------------
            # Compact document table
            # -----------------------------------------------------

            document_types = [
                "All",
                *sorted(counts),
            ]

            selected_type = st.selectbox(
                "Filter by document type",
                document_types,
            )

            visible = [
                document
                for document in documents
                if (
                    selected_type == "All"
                    or document.document_type == selected_type
                )
            ]

            st.dataframe(
                [
                    asdict(document)
                    for document in visible
                ],
                column_order=[
                    "period",
                    "document_type",
                    "title",
                    "url",
                    "source_page",
                ],
                height=260,
                row_height=32,
                column_config={
                    "period": "Reporting period",
                    "document_type": "Document type",
                    "title": "Title",
                    "url": st.column_config.LinkColumn(
                        "Document",
                        display_text="Open PDF",
                    ),
                    "source_page": st.column_config.LinkColumn(
                        "Source",
                        display_text="Official page",
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )

            output = StringIO()

            writer = csv.DictWriter(
                output,
                fieldnames=list(
                    asdict(documents[0]).keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                asdict(document)
                for document in documents
            )

            st.download_button(
                "Download full document list (CSV)",
                data=output.getvalue(),
                file_name="discovered_documents.csv",
                mime="text/csv",
            )

            # -----------------------------------------------------
            # Financial data extraction
            # -----------------------------------------------------

            st.divider()

            st.subheader(
                "Financial Data Extraction"
            )

            with st.expander(
                "Reporting-period settings"
            ):

                fiscal_end_month = st.selectbox(
                    "Financial year ends in (month)",
                    list(range(1, 13)),
                    index=2,
                    format_func=lambda month: (
                        date(
                            2000,
                            month,
                            1,
                        ).strftime("%B")
                    ),
                    help=(
                        "March is the default for Honasa "
                        "and many Indian companies."
                    ),
                )

            previously_checked = (
                st.session_state.get(
                    "verified_periods",
                    {},
                )
            )

            if (
                st.session_state.get(
                    "verified_periods_site"
                ) != website
            ):
                previously_checked = {}

            documents_for_extraction = [
                replace(
                    item,
                    period=previously_checked.get(
                        item.url,
                        item.period,
                    ),
                )
                for item in documents
            ]

            selection = select_documents(
                documents_for_extraction,
                fiscal_year_end_month=fiscal_end_month,
            )

            first_fy, second_fy = (
                selection.financial_years
            )

            st.write(
                f"Reporting window: "
                f"**FY{first_fy - 1}–{first_fy}**, "
                f"**FY{second_fy - 1}–{second_fy}**, "
                "plus available current-year quarters."
            )

            st.caption(
                "Quarterly periods: "
                + (
                    ", ".join(selection.quarters)
                    or "Not identified"
                )
            )

            selection_counts = Counter(
                item.document_type
                for item in selection.documents
            )

            if selection_counts:
                st.write(
                    " · ".join(
                        f"{kind}: {count}"
                        for kind, count in sorted(
                            selection_counts.items()
                        )
                    )
                )

            # -----------------------------------------------------
            # Verify uncertain reporting periods
            # -----------------------------------------------------

            if selection.unresolved:

                st.warning(
                    f"{len(selection.unresolved)} potentially "
                    "useful documents have uncertain periods. "
                    "You can verify their reporting periods "
                    "before processing."
                )

                if st.button(
                    "Check unknown reporting periods"
                ):

                    targets = selection.unresolved[:8]

                    checked, issues = (
                        verify_unknown_periods(
                            targets,
                            fiscal_year_end_month=(
                                fiscal_end_month
                            ),
                            cache_dir=Path(
                                "data/cache/pdfs"
                            ),
                        )
                    )

                    identified = [
                        item
                        for item in checked
                        if item.period not in {
                            None,
                            "Not identified",
                        }
                    ]

                    st.session_state[
                        "verified_periods"
                    ] = {
                        **previously_checked,
                        **{
                            item.url: item.period
                            for item in identified
                        },
                    }

                    st.session_state[
                        "verified_periods_site"
                    ] = website

                    st.session_state[
                        "period_check_report"
                    ] = {
                        "website": website,
                        "fiscal_end_month": (
                            fiscal_end_month
                        ),
                        "checked": len(checked),
                        "identified": identified,
                        "unresolved": [
                            item.title
                            for item in checked
                            if item not in identified
                        ],
                        "issues": issues,
                    }

                    st.rerun()

            period_report = (
                st.session_state.get(
                    "period_check_report",
                    {},
                )
            )

            if (
                period_report.get("website") == website
                and period_report.get(
                    "fiscal_end_month"
                ) == fiscal_end_month
            ):

                found = period_report[
                    "identified"
                ]

                st.info(
                    "Period check completed: "
                    f"{period_report['checked']} PDF(s) "
                    "inspected; "
                    f"{len(found)} period(s) identified, "
                    f"{len(period_report['unresolved'])} "
                    "still uncertain."
                )

                if found:
                    st.caption(
                        "Identified: "
                        + ", ".join(
                            (
                                f"{item.title} "
                                f"→ {item.period}"
                            )
                            for item in found
                        )
                    )

                if period_report["unresolved"]:

                    with st.expander(
                        "Why some periods could not "
                        "be confirmed"
                    ):

                        st.write(
                            "The first three PDF pages "
                            "did not contain a reliably "
                            "recognisable quarter/FY label. "
                            "These documents stay out of "
                            "the automatic selection until "
                            "confirmed."
                        )

                        for title in period_report[
                            "unresolved"
                        ]:
                            st.write(title)

                        for issue in period_report[
                            "issues"
                        ]:
                            st.caption(issue)

            # -----------------------------------------------------
            # Selected documents and extraction progress
            # -----------------------------------------------------

            if selection.documents:

                with st.expander(
                    f"Selected reports "
                    f"({len(selection.documents)})"
                ):

                    st.dataframe(
                        [
                            asdict(item)
                            for item in selection.documents
                        ],
                        height=240,
                        hide_index=True,
                        use_container_width=True,
                    )

                progress = load_extraction_progress(
                    website,
                    fiscal_end_month,
                )

                already_done = (
                    set(progress["completed"])
                    if progress
                    else set()
                )

                failed_urls = (
                    set(progress["failures"])
                    if progress
                    else set()
                )

                remaining = [
                    item
                    for item in selection.documents
                    if item.url not in (
                        already_done | failed_urls
                    )
                ]

                # -------------------------------------------------
                # Run PDF extraction
                # -------------------------------------------------

                if remaining:

                    n_files = st.number_input(
                        "How many documents should "
                        "we process next?",
                        min_value=1,
                        max_value=len(remaining),
                        value=min(
                            3,
                            len(remaining),
                        ),
                        step=1,
                        help=(
                            "The default is 3, "
                            "but you can process all "
                            "remaining PDFs at once."
                        ),
                        key=(
                            f"pdf_batch_count_"
                            f"{len(remaining)}"
                        ),
                    )

                    left, right = st.columns(2)

                    with left:
                        next_batch = st.button(
                            (
                                f"Process next "
                                f"{n_files} document(s)"
                            ),
                            type="primary",
                            use_container_width=True,
                        )

                    with right:
                        all_remaining = st.button(
                            (
                                f"Process all "
                                f"{len(remaining)} remaining"
                            ),
                            use_container_width=True,
                        )

                    if next_batch or all_remaining:

                        if progress is None:

                            run_dir = (
                                Path("data/runs")
                                / datetime.now().strftime(
                                    "%Y-%m-%d_%H-%M-%S_%f"
                                )
                            )

                            progress = {
                                "website": website,
                                "fiscal_end_month": (
                                    fiscal_end_month
                                ),
                                "run_dir": str(run_dir),
                                "completed": {},
                                "failures": {},
                            }

                        chosen = (
                            remaining
                            if all_remaining
                            else remaining[:n_files]
                        )

                        process_pdf_batch(
                            chosen,
                            progress,
                        )

                        st.rerun()

                else:
                    st.caption(
                        "All selected documents "
                        "have been processed."
                    )

                # -------------------------------------------------
                # Accurate progress count
                # -------------------------------------------------

                if progress:

                    selected_urls = {
                        doc.url
                        for doc in selection.documents
                    }

                    extracted = [
                        result
                        for url, result in (
                            progress["completed"].items()
                        )
                        if url in selected_urls
                    ]

                    facts = [
                        fact
                        for item in extracted
                        for fact in item.facts
                    ]

                    st.success(
                        f"Processed {len(extracted)} "
                        f"of {len(selected_urls)} documents. "
                        f"Extracted {len(facts)} "
                        "financial data points."
                    )

                    with st.expander(
                        "Processing details"
                    ):

                        st.caption(
                            "Data file: "
                            f"{progress['run_dir']}"
                            "/consolidated_facts.csv"
                        )

                        for item in extracted:
                            for issue in item.issues:
                                st.caption(
                                    f"{item.document.title}: "
                                    f"{issue}"
                                )

                    # ---------------------------------------------
                    # Retry failed documents
                    # ---------------------------------------------

                    if progress["failures"]:

                        with st.expander(
                            "Documents needing a retry "
                            f"({len(progress['failures'])})"
                        ):

                            for issue in (
                                progress["failures"].values()
                            ):
                                st.write(issue)

                        if st.button(
                            "Retry failed PDFs"
                        ):

                            retry_docs = [
                                item
                                for item in selection.documents
                                if item.url in (
                                    progress["failures"]
                                )
                            ]

                            process_pdf_batch(
                                retry_docs,
                                progress,
                            )

                            st.rerun()