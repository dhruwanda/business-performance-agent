"""One entry point for financial metric standardisation and data preparation.

Run from the repository root:
    python financial_data_preparation.py

Uses the existing financial_standardization.py module first, then prepares a
long-format master dataset. No financial-performance ratios are computed here.
All source observations remain in the original CSV/manifest and in the
intermediate standardised_consolidated_facts.csv.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import date, timedelta
from itertools import product
import json
import math
from pathlib import Path
import re
import unicodedata

from financial_standardization import _latest_run, standardise_metrics


MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
DATE_MONTH = r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"


def _month_end(year: int, month: int) -> date:
    return (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1))


def _fiscal_bounds(fy: int, quarter: int | None, end_month: int) -> tuple[date, date]:
    start_month = end_month % 12 + 1
    year_start = fy if start_month == 1 else fy - 1
    start = date(year_start, start_month, 1)
    if quarter is None:
        return start, _month_end(fy, end_month)
    m_index = (start_month - 1) + (quarter - 1) * 3
    q_year, q_month = year_start + m_index // 12, m_index % 12 + 1
    q_start = date(q_year, q_month, 1)
    last_index = m_index + 2
    q_end = _month_end(year_start + last_index // 12, last_index % 12 + 1)
    return q_start, q_end


def _fy_quarter_for_end(end: date, end_month: int) -> tuple[int, int]:
    fy = end.year + (end.month > end_month)
    start_month = end_month % 12 + 1
    quarter = ((end.month - start_month) % 12) // 3 + 1
    return fy, quarter


def _date_in_label(label: str) -> date | None:
    iso = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b", label)
    if iso:
        try:
            return date(int(iso[1]), int(iso[2]), int(iso[3]))
        except ValueError:
            return None
    day_first = re.search(rf"\b([0-3]?\d)\s+({DATE_MONTH})\.?[,]?[ ]+(20\d{{2}})\b", label, re.I)
    month_first = re.search(rf"\b({DATE_MONTH})\.?\s+([0-3]?\d),?\s+(20\d{{2}})\b", label, re.I)
    match = day_first or month_first
    if not match:
        return None
    day, name, year = ((match[1], match[2], match[3]) if day_first
                       else (match[2], match[1], match[3]))
    try:
        return date(int(year), MONTHS[name[:3].lower()], int(day))
    except (ValueError, KeyError):
        return None


def _year_in_label(label: str) -> int | None:
    # Indian FY ranges identify the fiscal *ending* year: FY 24-25 = FY2025.
    # Recognise both abbreviated/full ranges before matching a single FY year.
    span = re.search(
        r"\bFY\s*[- ]?(?:20\d{2}|\d{2})\s*[-/]\s*(20\d{2}|\d{2})\b",
        label, re.I,
    )
    if span:
        tail = int(span[1])
        return 2000 + tail if tail < 100 else tail
    match = re.search(r"\bFY\s*[- ]?(20\d{2}|\d{2})\b", label, re.I)
    if match:
        year = int(match[1])
        return year if year >= 2000 else year + 2000
    span = re.search(r"\b(20\d{2})\s*[-/]\s*(20\d{2}|\d{2})\b", label)
    if span:
        tail = int(span[2])
        return 2000 + tail if tail < 100 else tail
    return None


def classify_statement(category: str, metric_key: str) -> str:
    if category in {"profit_loss", "balance_sheet", "cash_flow"}:
        return category
    if category == "per_share":
        return "profit_loss"
    if metric_key.startswith("custom__"):
        return "operating_or_unclassified"
    return "operating_metrics"


def standardise_period(
    label: str, statement: str, fiscal_end_month: int, document_type: str = "",
) -> dict:
    """Preserve uncertainty instead of assigning guessed reporting periods."""
    raw = str(label or "").strip()
    upper = raw.upper().replace("–", "-").replace("_", " ")
    # LLM extractions commonly join the reporting duration and FY label.
    # Without the space, word-boundary patterns miss H1FY25 and 9MFY25.
    upper = re.sub(r"(Q[1-4]|H[12]|[369]\s*M)(?=FY)", r"\1 ", upper)
    fy = _year_in_label(upper)
    actual_end = _date_in_label(raw)
    match = re.search(r"\bQ\s*([1-4])\b|\bQUARTER\s*([1-4])\b", upper)
    quarter = int(match[1] or match[2]) if match else None
    ytd_q = None
    if re.search(r"\bH1\b|\b6\s*M\b|SIX MONTH|HALF.YEAR ENDED", upper):
        ytd_q = 2
    elif re.search(r"\b9\s*M\b|NINE MONTH", upper):
        ytd_q = 3
    elif re.search(r"\b3\s*M\b|THREE MONTH", upper):
        ytd_q = 1
    half_two = bool(re.search(r"\bH2\b|SECOND HALF", upper))
    is_quarter = bool(quarter or re.search(r"\bQUARTER(?:LY)?\b", upper))
    is_annual = bool(re.search(
        r"\bFY\s*[- ]?(?:20\d{2}|\d{2})\b|FINANCIAL YEAR|YEAR ENDED|ANNUAL|\b20\d{2}\s*[-/]\s*\d{2,4}\b",
        upper,
    ))

    if actual_end and fy is None:
        fy, end_quarter = _fy_quarter_for_end(actual_end, fiscal_end_month)
        if is_quarter or statement == "balance_sheet":
            quarter = quarter or end_quarter
    if fy is None:
        return {"period_type": "unknown", "fiscal_year": "", "fiscal_quarter": "",
                "period_start": "", "period_end": ""}

    start, annual_end = _fiscal_bounds(fy, None, fiscal_end_month)
    if statement == "balance_sheet":
        period_end = actual_end or (_fiscal_bounds(fy, quarter, fiscal_end_month)[1]
                                    if quarter else annual_end)
        return {"period_type": "point_in_time", "fiscal_year": fy,
                "fiscal_quarter": quarter or "", "period_start": "",
                "period_end": period_end.isoformat()}

    if (actual_end and not quarter and not ytd_q and not half_two
            and actual_end.month == fiscal_end_month
            and actual_end == _month_end(actual_end.year, actual_end.month)
            and "annual report" in document_type.lower()
            and statement in {"profit_loss", "cash_flow"}):
        # A date-only P&L/cash-flow column in an annual report is a FY flow.
        return {"period_type": "annual", "fiscal_year": fy,
                "fiscal_quarter": "", "period_start": start.isoformat(),
                "period_end": actual_end.isoformat()}

    if half_two:
        # H2 is not year-to-date, and must not be confused with H1 or Q4.
        q3_start, _ = _fiscal_bounds(fy, 3, fiscal_end_month)
        return {"period_type": "half_year", "fiscal_year": fy,
                "fiscal_quarter": "", "period_start": q3_start.isoformat(),
                "period_end": annual_end.isoformat()}
    if ytd_q and ytd_q > 1:
        _, q_end = _fiscal_bounds(fy, ytd_q, fiscal_end_month)
        return {"period_type": "cumulative", "fiscal_year": fy,
                "fiscal_quarter": ytd_q, "period_start": start.isoformat(),
                "period_end": (actual_end or q_end).isoformat()}
    if quarter and (is_quarter or ytd_q == 1 or (actual_end and not is_annual)):
        q_start, q_end = _fiscal_bounds(fy, quarter, fiscal_end_month)
        return {"period_type": "quarterly", "fiscal_year": fy,
                "fiscal_quarter": quarter, "period_start": q_start.isoformat(),
                "period_end": (actual_end or q_end).isoformat()}
    if is_annual and not quarter and not ytd_q:
        return {"period_type": "annual", "fiscal_year": fy,
                "fiscal_quarter": "", "period_start": start.isoformat(),
                "period_end": (actual_end or annual_end).isoformat()}
    if ytd_q == 1:
        q_start, q_end = _fiscal_bounds(fy, 1, fiscal_end_month)
        return {"period_type": "quarterly", "fiscal_year": fy,
                "fiscal_quarter": 1, "period_start": q_start.isoformat(),
                "period_end": (actual_end or q_end).isoformat()}
    return {"period_type": "unknown", "fiscal_year": fy, "fiscal_quarter": "",
            "period_start": "", "period_end": ""}


def standardise_unit(value: float, raw_unit: str, default_currency: str) -> dict:
    unit = unicodedata.normalize("NFKC", str(raw_unit or "")).lower().strip()
    unit = unit.replace("₹", " inr ").replace("$", " usd ").replace("€", " eur ")
    unit = re.sub(r"\s+", " ", unit)
    if "%" in unit or re.search(r"\bpercent(?:age)?\b|\bpct\b", unit):
        return {"standard_value": value, "standard_unit": "percent", "currency": ""}
    if re.search(r"\bbps\b|basis points?", unit):
        return {"standard_value": value, "standard_unit": "basis_points", "currency": ""}
    if unit in {"day", "days"}:
        return {"standard_value": value, "standard_unit": "days", "currency": ""}

    currency = ("USD" if re.search(r"\busd\b|\bus dollars?\b", unit) else
                "EUR" if re.search(r"\beur\b|euros?", unit) else
                "GBP" if re.search(r"\bgbp\b|pounds? sterling", unit) else
                "INR" if re.search(r"\binr\b|\brs\.?\b|rupees?", unit) else
                default_currency.upper())
    per_share = bool(re.search(r"per.share|/share|\beps\b", unit))
    scale = None
    if re.search(r"\bcrores?\b|\bcrs?\.?\b", unit):
        scale = 10_000_000
    elif re.search(r"\blakhs?\b|\blacs?\b", unit):
        scale = 100_000
    elif re.search(r"\bbillions?\b|\bbn\b", unit):
        scale = 1_000_000_000
    elif re.search(r"\bmillions?\b|\bmn\b", unit):
        scale = 1_000_000
    elif re.search(r"\bthousands?\b|\b000s\b", unit):
        scale = 1_000
    elif re.search(r"\binr\b|\brs\b|rupees?|\busd\b|\beur\b|\bgbp\b", unit):
        scale = 1

    if per_share:
        # Per-share values are never converted to crores or millions.
        return {"standard_value": value, "standard_unit": f"{currency}_per_share",
                "currency": currency}
    if scale is not None:
        denominator = 10_000_000 if currency == "INR" else 1_000_000
        return {"standard_value": value * scale / denominator,
                "standard_unit": f"{currency}_crore" if currency == "INR" else f"{currency}_million",
                "currency": currency}
    if re.search(r"\btimes\b|\bx\b|ratio", unit):
        return {"standard_value": value, "standard_unit": "times", "currency": ""}
    # A count/volume may be comparable in its original units, but don't
    # pretend that stores, customers, tonnes and individual units are alike.
    if re.search(r"\bunits?\b|\borders?\b|\bstores?\b|\bcustomers?\b|\btonnes?\b|\bkg\b|\blitres?\b", unit):
        cleaned = re.sub(r"[^a-z0-9]+", "_", unit).strip("_")
        return {"standard_value": value, "standard_unit": cleaned, "currency": ""}
    cleaned = re.sub(r"[^a-z0-9]+", "_", unit).strip("_") or "unspecified"
    return {"standard_value": value, "standard_unit": f"unstandardised_{cleaned}",
            "currency": ""}


def _scope(raw: str) -> str:
    text = str(raw or "").lower()
    if "consolidated" in text or "group" == text.strip():
        return "consolidated"
    if "standalone" in text or "stand alone" in text or "separate" in text:
        return "standalone"
    return "unspecified"


def _basis(raw: str, metric: str, evidence: str = "") -> str:
    """Keep explicitly named reporting definitions separate; don't invent one."""
    text = " ".join(str(part or "").lower() for part in (raw, metric, evidence))
    if re.search(r"like[ -]for[ -]like|\blfl\b", text):
        return "like_for_like"
    if re.search(r"pro[ -]?forma", text):
        return "pro_forma"
    if re.search(r"\bconstant[ -]currency\b", text):
        return "constant_currency"
    if re.search(r"\badjusted\b|\bnormalised\b|\bnormalized\b", text):
        return "adjusted"
    if "reported" in text or "statutory" in text:
        return "reported"
    return "unspecified"


def _source_score(record: dict) -> int:
    doc = str(record.get("document_type") or "").lower()
    period_type = record["period_type"]
    score = 0
    if "restated" in str(record.get("evidence") or "").lower():
        score += 8
    if period_type == "annual":
        score += 5 if "annual report" in doc else 3 if "result" in doc else 1
    elif period_type in {"quarterly", "cumulative"}:
        score += 5 if "quarterly" in doc or "result" in doc else 3 if "presentation" in doc else 1
    else:
        score += 3 if "annual report" in doc else 2
    return score


def _reconciliation_tolerance(record: dict) -> float:
    """Allow normal reporting rounding, not materially different definitions."""
    value = abs(float(record["standard_value"]))
    unit = str(record["standard_unit"])
    if unit == "percent":
        return max(0.15, value * 0.002)
    if unit in {"basis_points", "days", "times"}:
        return max(0.5, value * 0.002)
    if unit.endswith("_crore") or unit.endswith("_million"):
        # Large presentation amounts can differ by ~1 crore after rounding;
        # never apply a whole-crore tolerance to very small reported amounts.
        return max(0.1, min(1.05, value * 0.0075), value * 0.002)
    return max(0.01, value * 0.002)


KEY_FIELDS = (
    "metric_key", "period_type", "period_start", "period_end", "standard_unit",
    "standard_scope", "standard_basis",
)


def _reconcile(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for index, record in enumerate(records):
        if record["period_type"] == "unknown":
            # Do not accidentally merge two observations whose periods are unknown.
            key = ("unknown", index)
        else:
            key = tuple(record.get(field, "") for field in KEY_FIELDS)
        groups[key].append(record)

    result = []
    for group in groups.values():
        # Cluster genuinely comparable observations, retaining distinct
        # numeric versions as separate facts if the underlying basis is unclear.
        ranked = [row for _, row in sorted(
            enumerate(group), key=lambda item: (-_source_score(item[1]), item[0])
        )]
        clusters: list[list[dict]] = []
        for row in ranked:
            for cluster in clusters:
                anchor = cluster[0]
                tol = max(_reconciliation_tolerance(anchor), _reconciliation_tolerance(row))
                if abs(float(row["standard_value"]) - float(anchor["standard_value"])) <= tol:
                    cluster.append(row)
                    break
            else:
                clusters.append([row])

        variants = len(clusters)
        for variant_number, cluster in enumerate(clusters, start=1):
            selected = cluster[0].copy()
            selected["source_count"] = len(cluster)
            selected["conflicting_sources"] = len(group) - len(cluster)
            selected["supporting_sources"] = json.dumps([
                {"url": r.get("source_url", ""), "page": r.get("page", ""),
                 "reported_value": r.get("value", ""), "standard_value": r["standard_value"],
                 "unit": r.get("unit", ""), "document_type": r.get("document_type", ""),
                 "period": r.get("period", ""), "basis": r.get("basis", ""),
                 "evidence": r.get("evidence", "")}
                for r in cluster
            ], ensure_ascii=False)
            selected["derivation"] = "reported"
            selected["reporting_variant"] = f"variant_{variant_number}" if variants > 1 else ""
            selected["reporting_variant_count"] = variants
            selected["reporting_definition_status"] = (
                "ambiguous_multiple_versions" if variants > 1 else "consistent_or_rounded"
            )
            selected["reconciliation_status"] = (
                "unresolved_variant" if variants > 1 else "agreed_or_rounded"
            )
            selected["annual_quarter_check"] = ""
            selected["annual_quarter_difference"] = ""
            selected["annual_quarter_details"] = ""
            # Distinct versions cannot be used interchangeably in the
            # deterministic baseline. Both remain available to the LLM.
            selected["usable_for_calculations"] = (selected["period_type"] != "unknown"
                                                    and variants == 1)
            result.append(selected)
    return result


def _derive_quarters(records: list[dict], fiscal_end_month: int) -> list[dict]:
    """Quarterise only additive flows, never balance-sheet snapshots/margins."""
    derived = []
    index: dict[tuple, dict[tuple, dict]] = defaultdict(dict)
    observed_quarters: dict[tuple, set[int]] = defaultdict(set)
    for row in records:
        if row["statement_type"] not in {"profit_loss", "cash_flow"}:
            continue
        if str(row.get("metric_key", "")).endswith("_margin"):
            continue
        if (row["standard_unit"] in {"percent", "basis_points", "times"}
                or str(row["standard_unit"]).endswith("_per_share")):
            continue
        if row["period_type"] not in {"quarterly", "annual", "cumulative"}:
            continue
        fy = row["fiscal_year"]
        if not fy:
            continue
        key = (row["metric_key"], fy, row["standard_unit"],
               row["standard_scope"], row["standard_basis"])
        q = int(row["fiscal_quarter"]) if row["fiscal_quarter"] else None
        if row["period_type"] == "quarterly" and q is not None:
            observed_quarters[key].add(q)
        if not row["usable_for_calculations"]:
            continue
        index[key][(row["period_type"], q)] = row

    for key, values in index.items():
        fy = int(key[1])
        # For valid cumulative YTD statements, take successive differences.
        for quarter in (2, 3, 4):
            if quarter in observed_quarters[key]:
                continue
            current = values.get(("cumulative", quarter)) if quarter < 4 else values.get(("annual", None))
            if current is None or current.get("conflicting_sources"):
                continue
            preceding = (values.get(("cumulative", quarter - 1)) if quarter > 2
                         else values.get(("quarterly", 1)) or values.get(("cumulative", 1)))
            if preceding is None:
                # Sum the earlier individual quarters if all are present.
                earlier = [values.get(("quarterly", q)) for q in range(1, quarter)]
                if any(r is None or r.get("conflicting_sources") for r in earlier):
                    continue
                prior_total = sum(float(r["standard_value"]) for r in earlier)
                sources = [r["supporting_sources"] for r in earlier]
            else:
                if preceding.get("conflicting_sources"):
                    continue
                prior_total = float(preceding["standard_value"])
                sources = [preceding["supporting_sources"]]
            q_start, q_end = _fiscal_bounds(fy, quarter, fiscal_end_month)
            row = current.copy()
            row.update({
                "period": f"Q{quarter} FY{fy % 100:02d}", "period_type": "quarterly",
                "fiscal_quarter": quarter, "period_start": q_start.isoformat(),
                "period_end": q_end.isoformat(),
                "standard_value": float(current["standard_value"]) - prior_total,
                "value": "", "source_url": "", "page": "", "evidence": "",
                "derivation": f"Q{quarter} = {'FY' if quarter == 4 else str(quarter * 3) + 'M'} minus preceding periods",
                "supporting_sources": json.dumps([current["supporting_sources"], *sources]),
                "source_count": 0, "conflicting_sources": 0,
                "reconciliation_status": "derived_from_compatible_periods",
                "reporting_variant": "", "reporting_variant_count": 1,
                "reporting_definition_status": "derived",
                "annual_quarter_check": "", "annual_quarter_difference": "",
                "annual_quarter_details": "",
                "usable_for_calculations": True,
            })
            derived.append(row)
            values[("quarterly", quarter)] = row
    return derived


def _check_annual_quarter_consistency(records: list[dict]) -> None:
    """Record annual/quarter gaps for the future LLM without resolving them.

    When one or more quarters have genuinely different reported variants,
    retain *all* plausible quarter sums instead of silently choosing one.
    """
    groups: dict[tuple, dict[str, object]] = defaultdict(
        lambda: {"annual": [], "quarters": defaultdict(list)}
    )
    for row in records:
        if row["statement_type"] not in {"profit_loss", "cash_flow"}:
            continue
        unit = str(row["standard_unit"])
        if (unit in {"percent", "basis_points", "times"}
                or unit.endswith("_per_share")
                or str(row["metric_key"]).endswith("_margin")
                or not row["fiscal_year"]):
            continue
        key = (row["metric_key"], row["fiscal_year"], unit,
               row["standard_scope"], row["standard_basis"])
        if row["period_type"] == "annual":
            groups[key]["annual"].append(row)
        elif (row["period_type"] == "quarterly"
              and row["derivation"] == "reported"
              and row["fiscal_quarter"]):
            groups[key]["quarters"][int(row["fiscal_quarter"])].append(row)

    for group in groups.values():
        quarters = group["quarters"]
        for annual in group["annual"]:
            if set(quarters) != {1, 2, 3, 4}:
                annual["annual_quarter_check"] = "insufficient_reported_quarters"
                continue
            combinations = math.prod(len(quarters[q]) for q in (1, 2, 3, 4))
            if combinations > 256:
                annual["annual_quarter_check"] = "too_many_quarter_variants"
                continue

            candidates = []
            seen_sums = set()
            for choice in product(*(quarters[q] for q in (1, 2, 3, 4))):
                total = round(sum(float(q["standard_value"]) for q in choice), 6)
                if total in seen_sums:
                    continue
                seen_sums.add(total)
                candidates.append({
                    "quarter_sum": total,
                    "annual_minus_quarters": round(float(annual["standard_value"]) - total, 6),
                    "quarter_values": [float(q["standard_value"]) for q in choice],
                    "quarter_variants": [q["reporting_variant"] for q in choice],
                })
            candidates.sort(key=lambda candidate: candidate["quarter_sum"])
            annual["annual_quarter_details"] = json.dumps({
                "annual_value": float(annual["standard_value"]),
                "standard_unit": annual["standard_unit"],
                "quarter_sum_candidates": candidates,
            }, ensure_ascii=False)

            if len(candidates) == 1:
                annual["annual_quarter_difference"] = candidates[0]["annual_minus_quarters"]
            tolerance = max(2.0, abs(float(annual["standard_value"])) * 0.005)
            matches = any(abs(c["annual_minus_quarters"]) <= tolerance for c in candidates)
            if len(candidates) > 1:
                annual["annual_quarter_check"] = (
                    "possible_match_ambiguous_quarters" if matches
                    else "mismatch_ambiguous_quarters"
                )
            else:
                annual["annual_quarter_check"] = (
                    "match_within_rounding" if matches else "material_mismatch"
                )
            # A discrepancy is information for the eventual explanation,
            # not a reason to change an annual value or force a definition.


def prepare_financial_data(run_dir: Path, default_currency: str = "INR") -> tuple[Path, dict]:
    source = run_dir / "standardised_consolidated_facts.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Standardised facts missing: {source}")
    manifest = run_dir / "extraction_manifest.json"
    meta = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    fiscal_end = int(meta.get("fiscal_end_month", 3))
    if not 1 <= fiscal_end <= 12:
        raise ValueError("Fiscal year-end month must be 1 to 12")

    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        data = list(reader)
    if not data:
        raise ValueError("No standardised financial facts found")

    observations = []
    for row in data:
        try:
            value = float(row["value"])
            if not math.isfinite(value):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        statement = classify_statement(row.get("metric_category", ""), row.get("metric_key", ""))
        record = row.copy()
        record["statement_type"] = statement
        record["standard_scope"] = _scope(row.get("scope", ""))
        record["standard_basis"] = _basis(
            row.get("basis", ""), row.get("metric", ""), row.get("evidence", "")
        )
        record.update(standardise_period(
            row.get("period", ""), statement, fiscal_end,
            row.get("document_type", ""),
        ))
        record.update(standardise_unit(value, row.get("unit", ""), default_currency))
        observations.append(record)

    master = _reconcile(observations)
    derived = _derive_quarters(master, fiscal_end)
    master.extend(derived)
    _check_annual_quarter_consistency(master)
    master.sort(key=lambda row: (
        str(row.get("metric_key", "")), str(row.get("standard_scope", "")),
        str(row.get("period_end", "")), str(row.get("period_type", "")),
        str(row.get("standard_basis", "")), str(row.get("reporting_variant", "")),
    ))
    extra = [
        "statement_type", "standard_scope", "standard_basis", "period_type",
        "fiscal_year", "fiscal_quarter", "period_start", "period_end",
        "standard_value", "standard_unit", "currency", "source_count",
        "conflicting_sources", "supporting_sources", "derivation",
        "reconciliation_status", "reporting_variant", "reporting_variant_count",
        "reporting_definition_status", "annual_quarter_check",
        "annual_quarter_difference", "annual_quarter_details",
        "usable_for_calculations",
    ]
    output = run_dir / "final_prepared_consolidated_facts.csv"
    temp = output.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns + [col for col in extra if col not in columns])
        writer.writeheader()
        writer.writerows(master)
    temp.replace(output)
    ambiguous_groups = {
        tuple(record.get(field, "") for field in KEY_FIELDS)
        for record in master if int(record.get("reporting_variant_count", 1)) > 1
    }
    summary = {
        "source_observations": len(observations),
        "master_observations": len(master),
        "quarters_derived": len(derived),
        "records_without_recognised_period": sum(r["period_type"] == "unknown" for r in master),
        "records_with_conflicting_sources": sum(bool(r["conflicting_sources"]) for r in master),
        "unresolved_source_conflicts": len(ambiguous_groups),
        "annual_quarter_mismatches": sum(
            r["annual_quarter_check"] in {"material_mismatch", "mismatch_ambiguous_quarters"}
            for r in master
        ),
        "annual_quarter_ambiguous_comparisons": sum(
            r["annual_quarter_check"] in {
                "possible_match_ambiguous_quarters", "mismatch_ambiguous_quarters"
            } for r in master
        ),
        "calculation_ready_observations": sum(bool(r["usable_for_calculations"]) for r in master),
        "fiscal_year_end_month": fiscal_end, "default_currency": default_currency,
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Full financial data preparation tool")
    parser.add_argument("--run-dir", type=Path, help="Specific extraction run directory")
    parser.add_argument("--currency", default="INR", help="Default source currency (default INR)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM mapping of unfamiliar metric names")
    args = parser.parse_args()
    run_dir = args.run_dir or _latest_run(Path("data/runs"))
    standardised, _ = standardise_metrics(run_dir, use_llm=not args.no_llm)
    prepared, summary = prepare_financial_data(run_dir, args.currency)
    print(f"Metric-standardised facts: {standardised}")
    print(f"Calculation-ready long-format dataset: {prepared}")
    print(f"Summary: {json.dumps(summary, ensure_ascii=False)}")
    print("No source files were removed or overwritten.")


if __name__ == "__main__":
    main()
