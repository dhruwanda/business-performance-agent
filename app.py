"""Streamlit UI for company financial-document discovery."""

from collections import Counter
from dataclasses import asdict, replace
from datetime import date, datetime
from io import StringIO
import csv
import json
import os
from pathlib import Path

import requests
import streamlit as st

from discovery_workflow import Document, discover_documents
from financial_extraction import (
    ExtractionResult, FinancialFact, extract_document, select_documents,
    verify_unknown_periods,
)


def gemini_api_key() -> str:
    """Use an existing key without displaying it in the app."""
    if os.getenv("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except FileNotFoundError:
        return ""


def load_extraction_progress(website: str, fiscal_end_month: int) -> dict | None:
    """Resume the latest saved run for this company, even after a browser refresh."""
    current = st.session_state.get("extraction_progress")
    if current and (current["website"], current["fiscal_end_month"]) == (
        website, fiscal_end_month
    ):
        return current

    # One-time handover from the previous UI, which kept the first batch only
    # in session state. This avoids unnecessarily reprocessing those PDFs.
    if (st.session_state.get("extraction_website") == website
            and st.session_state.get("extraction_run_dir")
            and st.session_state.get("extraction_output")):
        earlier, _ = st.session_state["extraction_output"]
        restored = {
            "website": website, "fiscal_end_month": fiscal_end_month,
            "run_dir": st.session_state["extraction_run_dir"],
            "completed": {item.document.url: item for item in earlier},
            "failures": {},
        }
        save_extraction_progress(restored)
        st.session_state["extraction_progress"] = restored
        # Prevent accidentally adopting the old progress for another fiscal year.
        st.session_state.pop("extraction_output", None)
        return restored

    runs = Path("data/runs")
    if runs.is_dir():
        for folder in sorted(runs.iterdir(), reverse=True):
            manifest = folder / "extraction_manifest.json"
            if not manifest.is_file():
                continue
            try:
                saved = json.loads(manifest.read_text(encoding="utf-8"))
                if (saved.get("website"), saved.get("fiscal_end_month")) != (
                    website, fiscal_end_month
                ):
                    continue
                completed = {}
                for url, item in saved.get("completed", {}).items():
                    completed[url] = ExtractionResult(
                        document=Document(**item["document"]),
                        saved_pdf=item["saved_pdf"], page_count=item["page_count"],
                        pages_sent=item["pages_sent"],
                        facts=[FinancialFact(**fact) for fact in item["facts"]],
                        issues=item["issues"],
                    )
                saved["completed"] = completed
                st.session_state["extraction_progress"] = saved
                return saved
            except (OSError, ValueError, TypeError, KeyError):
                continue
    st.session_state.pop("extraction_progress", None)
    return None


def save_extraction_progress(progress: dict) -> None:
    """Keep one source-attributed fact ledger across all PDF batches."""
    folder = Path(progress["run_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    facts = [asdict(fact) for result in progress["completed"].values()
             for fact in result.facts]
    # Retain one row per source statement. Reconciliation across documents
    # belongs to the later financial-calculation/consolidation step.
    csv_path = folder / "consolidated_facts.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as output:
        headers = list(FinancialFact.__dataclass_fields__)
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(facts)

    manifest = {
        **{key: value for key, value in progress.items() if key != "completed"},
        "completed": {
            url: asdict(result) for url, result in progress["completed"].items()
        },
    }
    temp = folder / "extraction_manifest.tmp"
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(folder / "extraction_manifest.json")


def process_pdf_batch(documents: list[Document], progress: dict) -> None:
    try:
        model = os.getenv("GEMINI_MODEL") or st.secrets.get(
            "GEMINI_MODEL", "gemini-2.5-flash"
        )
    except FileNotFoundError:
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    indicator = st.progress(0, text="Preparing PDF extraction...")
    for index, document in enumerate(documents, 1):
        indicator.progress(
            (index - 1) / len(documents),
            text=f"Reading {index}/{len(documents)}: {document.title}",
        )
        try:
            result = extract_document(
                document, api_key=gemini_api_key(), model=model,
                cache_dir=Path(progress["run_dir"]),
                shared_cache_dir=Path("data/cache"),
            )
            progress["completed"][document.url] = result
            progress["failures"].pop(document.url, None)
        except (requests.RequestException, RuntimeError, ValueError, OSError) as error:
            progress["failures"][document.url] = f"{document.title}: {error}"
        # Save after each PDF so even an interrupted batch keeps its progress.
        save_extraction_progress(progress)
    indicator.progress(1.0, text="Batch completed")
    st.session_state["extraction_progress"] = progress


st.set_page_config(page_title="Business Performance Analyst", layout="centered")

st.title("Business Performance Analyst")
st.caption("Steps 2–3 of 8 · Document discovery and extraction")
st.write(
    "Enter a company website. Python reads the available links, and the LLM "
    "selects which pages to inspect for financial reports."
)

with st.form("website_form"):
    website = st.text_input(
        "Company website",
        value="https://honasa.in",
        placeholder="https://www.company.com",
        help="Enter the company's official website or its Investor Relations page.",
    )
    submitted = st.form_submit_button("Discover documents", type="primary")

if submitted:
    with st.spinner("The LLM is selecting pages and finding financial documents..."):
        try:
            st.session_state["discovery_result"] = discover_documents(
                website, api_key=gemini_api_key()
            )
            st.session_state["searched_website"] = website
        except (requests.RequestException, RuntimeError, ValueError) as error:
            st.session_state.pop("discovery_result", None)
            st.error(f"Couldn't complete discovery: {error}")

result = (
    st.session_state.get("discovery_result")
    if st.session_state.get("searched_website") == website else None
)
if result is not None:
    documents = result.documents
    with st.expander(f"Pages visited ({len(result.pages_visited)})"):
        for page in result.pages_visited:
            st.write(page)
    if not documents:
        st.warning("No matching PDF links found. Try entering the company's Investor Relations URL directly. Some websites require JavaScript or use other document formats.")
    else:
        counts = Counter(document.document_type for document in documents)
        st.success(f"Found {len(documents)} candidate financial-document links")
        st.caption(" · ".join(f"{kind}: {count}" for kind, count in sorted(counts.items())))
        missing_periods = sum(document.period == "Not identified" for document in documents)
        if missing_periods:
            st.info(
                f"{missing_periods} document(s) have no period identified from their links. "
                "We can check uncertain periods from PDF contents in Step 3."
            )

        document_types = ["All", *sorted(counts)]
        selected_type = st.selectbox("Filter by document type", document_types)
        visible = [
            document for document in documents
            if selected_type == "All" or document.document_type == selected_type
        ]

        st.dataframe(
            [asdict(document) for document in visible],
            column_order=["period", "document_type", "title", "url", "source_page"],
            column_config={
                "period": "Reporting period",
                "document_type": "Document type",
                "title": "Title",
                "url": st.column_config.LinkColumn("Document", display_text="Open PDF"),
                "source_page": st.column_config.LinkColumn("Source", display_text="Official page"),
            },
            hide_index=True,
            use_container_width=True,
        )

        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=list(asdict(documents[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(document) for document in documents)
        st.download_button(
            "Download full document list (CSV)",
            data=output.getvalue(),
            file_name="discovered_documents.csv",
            mime="text/csv",
        )

        st.caption("The LLM classified these links from the website. PDF contents and reporting periods are checked during extraction.")

        # Step 3: keep the UI here; PDF processing lives in financial_extraction.py.
        st.divider()
        st.subheader("Step 3 · Financial data extraction")
        with st.expander("Reporting-period settings"):
            fiscal_end_month = st.selectbox(
                "Financial year ends in (month)",
                list(range(1, 13)), index=2,
                format_func=lambda month: date(2000, month, 1).strftime("%B"),
                help="March is the default for Honasa and many Indian companies.",
            )

        previously_checked = st.session_state.get("verified_periods", {})
        if st.session_state.get("verified_periods_site") != website:
            previously_checked = {}
        documents_for_extraction = [
            replace(item, period=previously_checked.get(item.url, item.period))
            for item in documents
        ]
        selection = select_documents(
            documents_for_extraction, fiscal_year_end_month=fiscal_end_month
        )
        first_fy, second_fy = selection.financial_years
        st.write(
            f"Reporting window: **FY{first_fy - 1}–{first_fy}**, "
            f"**FY{second_fy - 1}–{second_fy}**, plus available current-year quarters."
        )
        st.caption("Quarterly periods: " + (", ".join(selection.quarters) or "Not identified"))
        selection_counts = Counter(item.document_type for item in selection.documents)
        if selection_counts:
            st.write(" · ".join(
                f"{kind}: {count}" for kind, count in sorted(selection_counts.items())
            ))

        if selection.unresolved:
            st.warning(
                f"{len(selection.unresolved)} potentially useful documents have uncertain periods. "
                "You can check their opening pages without using the LLM."
            )
            if st.button("Check unknown reporting periods"):
                targets = selection.unresolved[:8]
                checked, issues = verify_unknown_periods(
                    targets, fiscal_year_end_month=fiscal_end_month,
                    cache_dir=Path("data/cache/pdfs"),
                )
                identified = [item for item in checked
                              if item.period not in {None, "Not identified"}]
                st.session_state["verified_periods"] = {
                    **previously_checked,
                    **{item.url: item.period for item in identified},
                }
                st.session_state["verified_periods_site"] = website
                st.session_state["period_check_report"] = {
                    "website": website, "fiscal_end_month": fiscal_end_month,
                    "checked": len(checked), "identified": identified,
                    "unresolved": [item.title for item in checked if item not in identified],
                    "issues": issues,
                }
                st.rerun()

        period_report = st.session_state.get("period_check_report", {})
        if (period_report.get("website") == website
                and period_report.get("fiscal_end_month") == fiscal_end_month):
            found = period_report["identified"]
            st.info(
                f"Period check completed: {period_report['checked']} PDF(s) inspected; "
                f"{len(found)} period(s) identified, "
                f"{len(period_report['unresolved'])} still uncertain."
            )
            if found:
                st.caption("Identified: " + ", ".join(
                    f"{item.title} → {item.period}" for item in found
                ))
            if period_report["unresolved"]:
                with st.expander("Why some periods could not be confirmed"):
                    st.write(
                        "The first three PDF pages did not contain a reliably "
                        "recognisable quarter/FY label. These documents stay out "
                        "of the automatic selection until confirmed."
                    )
                    for title in period_report["unresolved"]:
                        st.write(title)
                    for issue in period_report["issues"]:
                        st.caption(issue)

        if selection.documents:
            with st.expander(f"Selected PDFs ({len(selection.documents)})"):
                st.dataframe(
                    [asdict(item) for item in selection.documents],
                    hide_index=True, use_container_width=True,
                )
            progress = load_extraction_progress(website, fiscal_end_month)
            already_done = set(progress["completed"]) if progress else set()
            failed_urls = set(progress["failures"]) if progress else set()
            remaining = [item for item in selection.documents
                         if item.url not in already_done | failed_urls]

            if remaining:
                n_files = st.number_input(
                    "How many PDFs should we process next?", min_value=1,
                    max_value=len(remaining), value=min(3, len(remaining)), step=1,
                    help="The default is 3, but you can process all remaining PDFs at once.",
                    key=f"pdf_batch_count_{len(remaining)}",
                )
                left, right = st.columns(2)
                with left:
                    next_batch = st.button(
                        f"Process next {n_files} PDF(s)", type="primary",
                        use_container_width=True,
                    )
                with right:
                    all_remaining = st.button(
                        f"Process all {len(remaining)} remaining",
                        use_container_width=True,
                    )
                if next_batch or all_remaining:
                    if progress is None:
                        run_dir = Path("data/runs") / datetime.now().strftime(
                            "%Y-%m-%d_%H-%M-%S_%f"
                        )
                        progress = {
                            "website": website, "fiscal_end_month": fiscal_end_month,
                            "run_dir": str(run_dir), "completed": {}, "failures": {},
                        }
                    chosen = remaining if all_remaining else remaining[:n_files]
                    process_pdf_batch(chosen, progress)
                    st.rerun()
            else:
                st.success("No unprocessed PDFs remain in the current selection.")

            if progress:
                extracted = list(progress["completed"].values())
                facts = [fact for item in extracted for fact in item.facts]
                st.success(
                    f"Processed {len(extracted)} of {len(selection.documents)} selected PDFs. "
                    f"Stored {len(facts)} extracted financial data points."
                )
                st.caption(f"Saved together in: {progress['run_dir']}/consolidated_facts.csv")
                st.caption(
                    "The LLM reads selected financial PDF pages as images plus "
                    "extracted text (or reuses cached results). Earnings-call "
                    "transcripts are saved as text for later analysis."
                )
                for item in extracted:
                    for issue in item.issues:
                        st.caption(f"{item.document.title}: {issue}")
                if progress["failures"]:
                    with st.expander(
                        f"PDFs that failed ({len(progress['failures'])})"
                    ):
                        for issue in progress["failures"].values():
                            st.write(issue)
                    if st.button("Retry failed PDFs"):
                        retry_docs = [item for item in selection.documents
                                      if item.url in progress["failures"]]
                        process_pdf_batch(retry_docs, progress)
                        st.rerun()
