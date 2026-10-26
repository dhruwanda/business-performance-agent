"""Deterministic first-level financial calculations for Business Performance Agent.

Input:  data/runs/<run>/final_prepared_consolidated_facts.csv
Output: data/runs/<run>/calculated_financial_metrics.csv
        data/runs/<run>/financial_calculation_summary.json

Run from the repository root:
    python financial_calculations.py --run-dir data/runs/<run>

Or point directly to a prepared CSV:
    python financial_calculations.py --input path/to/final_prepared_consolidated_facts.csv

This module never re-extracts documents, invokes an LLM, or modifies upstream data.
It only uses rows explicitly marked usable_for_calculations. All comparisons
require compatible metric, scope, basis, unit, and reporting period.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import date
import json
import math
from pathlib import Path
from typing import Any, Iterable


# These are exact canonical-key alternatives, not fuzzy string matching.
# The upstream metric dictionary may use any of these legitimate names.
# Distinct concepts, especially revenue and total income, remain distinct.
KEYS = {
    "sales": ("revenue_from_operations", "revenue", "net_sales", "sales"),
    "gross_profit": ("gross_profit",),
    "ebitda": ("ebitda",),
    "adjusted_ebitda": ("adjusted_ebitda",),
    "ebit": ("ebit", "earnings_before_interest_and_tax"),
    "operating_profit": ("operating_profit",),
    "pbt": ("profit_before_tax", "profit_before_taxes", "pbt"),
    "pat": ("profit_after_tax", "profit_for_the_period", "net_profit", "net_income", "pat"),
    "tax": ("tax_expense", "income_tax_expense", "total_tax_expense"),
    "cogs": ("cost_of_goods_sold", "cost_of_sales", "cost_of_revenue"),
    "total_expenses": ("total_expenses",),
    "employee_expenses": ("employee_expense", "employee_benefit_expenses", "employee_benefits_expense", "employee_expenses"),
    "marketing_expenses": ("advertising_promotion_expense", "advertising_and_promotion_expenses", "advertising_expenses", "marketing_expenses", "selling_and_marketing_expenses"),
    "finance_costs": ("finance_costs", "finance_cost", "interest_expense"),
    "other_expenses": ("other_expenses",),
    "depreciation": ("depreciation_amortisation", "depreciation_and_amortisation", "depreciation_and_amortization", "depreciation_expense"),
    "other_income": ("other_income",),
    "cfo": ("net_cash_from_operating_activities", "net_cash_flow_from_operating_activities", "cash_flow_from_operations", "operating_cash_flow"),
    "capex": ("capital_expenditure", "capex", "purchase_of_property_plant_and_equipment"),
    "current_assets": ("total_current_assets", "current_assets"),
    "current_liabilities": ("total_current_liabilities", "current_liabilities"),
    "inventory": ("inventories", "inventory"),
    "receivables": ("trade_receivables", "accounts_receivable", "accounts_receivables"),
    "payables": ("trade_payables", "accounts_payable", "accounts_payables"),
    "cash": ("cash_and_cash_equivalents", "cash_and_bank_balances", "cash_and_cash_equivalent"),
    "debt": ("total_borrowings", "total_debt"),
    "equity": ("total_equity", "equity_attributable_to_owners", "shareholders_equity", "total_shareholders_equity"),
    "assets": ("total_assets",),
    "liabilities": ("total_liabilities",),
}

PERCENT_UNITS = {"percent", "basis_points"}
NONADDITIVE_UNITS = {"percent", "basis_points", "times", "days"}
OUTPUT_FIELDS = (
    "metric_key", "metric_label", "family", "value", "unit", "formula",
    "period_type", "fiscal_year", "fiscal_quarter", "period_start", "period_end",
    "standard_scope", "standard_basis", "currency", "comparison_period_end",
    "input_facts_json", "notes",
)


def _num(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(str(value).strip()) if str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _safe_div(numerator: float, denominator: float, *, positive: bool = True) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if positive and denominator <= 0:
        return None
    if not positive and denominator == 0:
        return None
    answer = numerator / denominator
    return answer if math.isfinite(answer) else None


def _money(unit: str) -> bool:
    return unit.endswith("_crore") or unit.endswith("_million")


def _additive(fact: "Fact") -> bool:
    """Only accounting flows, not margins, snapshots, EPS or generic KPIs."""
    key = fact.metric_key
    tokens = set(key.split("_"))
    return (fact.statement in {"profit_loss", "cash_flow"} and _money(fact.unit)
            and not (tokens & {"margin", "ratio", "rate", "growth"})
            and "per_share" not in key)


@dataclass(frozen=True)
class Fact:
    metric_key: str
    label: str
    statement: str
    period_type: str
    fy: int
    quarter: int | None
    start: str
    end: str
    scope: str
    basis: str
    unit: str
    currency: str
    value: float
    evidence: dict[str, Any]

    @property
    def dimensions(self) -> tuple[str, str, str, str]:
        return (self.scope, self.basis, self.unit, self.currency)

    @property
    def series(self) -> tuple[str, str, str, str, str]:
        return (self.metric_key, *self.dimensions)

    @property
    def period(self) -> tuple[str, str, str]:
        return (self.period_type, self.start, self.end)

    def reference(self) -> dict[str, Any]:
        return {
            "metric_key": self.metric_key, "value": self.value, "unit": self.unit,
            "period_type": self.period_type, "period_start": self.start,
            "period_end": self.end, "source_url": self.evidence.get("source_url", ""),
            "page": self.evidence.get("page", ""),
            "supporting_sources": self.evidence.get("supporting_sources", ""),
            "derivation": self.evidence.get("derivation", ""),
            "annual_quarter_check": self.evidence.get("annual_quarter_check", ""),
            "annual_quarter_difference": self.evidence.get("annual_quarter_difference", ""),
            "calculation_inputs": self.evidence.get("calculation_inputs", []),
        }


def _fact_from_row(row: dict[str, str]) -> Fact | None:
    if not _truth(row.get("usable_for_calculations", "")):
        return None
    value = _num(row.get("standard_value"))
    fy = _int(row.get("fiscal_year"))
    if value is None or fy is None or not row.get("metric_key") or not row.get("period_end"):
        return None
    if row.get("period_type") not in {"annual", "quarterly", "cumulative", "half_year", "point_in_time"}:
        return None
    if not row.get("standard_unit"):
        return None
    return Fact(
        metric_key=row["metric_key"], label=row.get("metric_label") or row["metric_key"],
        statement=row.get("statement_type", ""), period_type=row["period_type"],
        fy=fy, quarter=_int(row.get("fiscal_quarter")),
        start=row.get("period_start", ""), end=row["period_end"],
        scope=row.get("standard_scope", "unspecified"),
        basis=row.get("standard_basis", "unspecified"),
        unit=row["standard_unit"], currency=row.get("currency", ""),
        value=value, evidence=row,
    )


def read_prepared(path: Path) -> tuple[list[Fact], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "metric_key", "standard_value", "standard_unit", "statement_type",
            "standard_scope", "standard_basis", "period_type", "period_end",
            "fiscal_year", "usable_for_calculations",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Prepared dataset missing: {', '.join(sorted(required - set(reader.fieldnames or [])))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Prepared dataset is empty")
    candidates = [f for row in rows if (f := _fact_from_row(row)) is not None]
    # Do not silently pick one if repeated rows still have the same exact key.
    buckets: dict[tuple, list[Fact]] = defaultdict(list)
    for f in candidates:
        buckets[(f.series, f.period)].append(f)
    duplicates = [key for key, group in buckets.items() if len(group) > 1]
    facts = [group[0] for group in buckets.values() if len(group) == 1]
    diagnostics = {
        "total_rows": len(rows), "upstream_usable_rows": sum(_truth(r.get("usable_for_calculations")) for r in rows),
        "accepted_facts": len(facts), "excluded_duplicate_fact_keys": len(duplicates),
        "excluded_or_invalid_rows": len(rows) - len(candidates),
        "unresolved_variant_rows": sum(r.get("reconciliation_status") == "unresolved_variant" for r in rows),
        "annual_quarter_mismatches": sum(r.get("annual_quarter_check") in {"material_mismatch", "mismatch_ambiguous_quarters"} for r in rows),
        "annual_quarter_ambiguous_checks": sum(r.get("annual_quarter_check") in {"possible_match_ambiguous_quarters", "mismatch_ambiguous_quarters"} for r in rows),
    }
    return facts, diagnostics


class Calculator:
    def __init__(self, facts: list[Fact]):
        self.facts = facts
        self.rows: list[dict[str, Any]] = []
        self.seen: set[tuple] = set()
        self.skips: Counter[str] = Counter()

    def add(self, key: str, label: str, family: str, value: float | None, unit: str,
            period: Fact, formula: str, inputs: Iterable[Fact], *,
            compared: str = "", notes: str = "", start: str | None = None,
            period_type: str | None = None) -> Fact | None:
        if value is None or not math.isfinite(value):
            self.skips["invalid_division_or_value"] += 1
            return None
        ptype = period_type or period.period_type
        pstart = period.start if start is None else start
        identity = (key, ptype, pstart, period.end, period.scope, period.basis, unit, compared)
        if identity in self.seen:
            return None
        self.seen.add(identity)
        references = [f.reference() for f in inputs]
        row = {
            "metric_key": key, "metric_label": label, "family": family,
            "value": round(value, 10), "unit": unit, "formula": formula,
            "period_type": ptype, "fiscal_year": period.fy,
            "fiscal_quarter": period.quarter or "", "period_start": pstart,
            "period_end": period.end, "standard_scope": period.scope,
            "standard_basis": period.basis, "currency": period.currency if _money(unit) else "",
            "comparison_period_end": compared,
            "input_facts_json": json.dumps(references, ensure_ascii=False), "notes": notes,
        }
        self.rows.append(row)
        return Fact(key, label, "calculated", ptype, period.fy, period.quarter,
                    pstart, period.end, period.scope, period.basis, unit,
                    row["currency"], float(value),
                    {"derivation": formula, "calculation_inputs": references})

    def _compare(self, current: Fact, previous: Fact, comparison: str) -> None:
        if current.dimensions != previous.dimensions:
            return
        prefix = f"{current.metric_key}_{comparison}"
        diff = current.value - previous.value
        unit = current.unit
        if unit == "percent":
            self.add(prefix + "_pp_change", current.label + f" {comparison.upper()} change",
                     "changes", diff, "percentage_points", current,
                     "current percentage - comparable previous percentage", (current, previous),
                     compared=previous.end)
        elif unit == "basis_points":
            self.add(prefix + "_bps_change", current.label + f" {comparison.upper()} change",
                     "changes", diff, "basis_points", current,
                     "current basis points - comparable previous basis points", (current, previous),
                     compared=previous.end)
        else:
            self.add(prefix + "_absolute_change", current.label + f" {comparison.upper()} absolute change",
                     "changes", diff, unit, current, "current - comparable previous",
                     (current, previous), compared=previous.end)
            if unit not in {"times", "days"} and not unit.startswith("unstandardised_"):
                # A negative/zero base has no conventional percentage-growth meaning.
                growth = _safe_div(diff * 100, previous.value)
                self.add(prefix + "_growth_pct", current.label + f" {comparison.upper()} growth",
                         "growth", growth, "percent", current,
                         "(current / previous - 1) * 100; previous must be positive",
                         (current, previous), compared=previous.end)

    @staticmethod
    def _quarter_index(f: Fact, fiscal_end_month: int) -> int | None:
        if f.quarter in {1, 2, 3, 4}:
            return f.fy * 4 + f.quarter
        if f.period_type == "point_in_time":
            try:
                d = date.fromisoformat(f.end)
            except ValueError:
                return None
            first_month = fiscal_end_month % 12 + 1
            quarter = ((d.month - first_month) % 12) // 3 + 1
            return f.fy * 4 + quarter
        return None

    def comparisons(self, facts: list[Fact], fiscal_end_month: int) -> None:
        groups: dict[tuple, list[Fact]] = defaultdict(list)
        for f in facts:
            groups[(f.series, f.period_type)].append(f)
        for series in groups.values():
            annual_candidates: dict[tuple, list[Fact]] = defaultdict(list)
            quarter_candidates: dict[int, list[Fact]] = defaultdict(list)
            for f in series:
                if f.period_type in {"annual", "cumulative", "half_year", "ttm"}:
                    # Cumulative comparisons require the same YTD quarter.
                    signature = f.quarter if f.period_type in {"cumulative", "ttm"} else (
                        f.start[5:7] if f.period_type == "half_year" else None)
                    annual_candidates[(f.fy, signature)].append(f)
                elif f.period_type in {"quarterly", "point_in_time"}:
                    qindex = self._quarter_index(f, fiscal_end_month)
                    if qindex is not None:
                        quarter_candidates[qindex].append(f)
            by_year = {key: rows[0] for key, rows in annual_candidates.items() if len(rows) == 1}
            by_qindex = {key: rows[0] for key, rows in quarter_candidates.items() if len(rows) == 1}
            for f in series:
                if f.period_type in {"annual", "cumulative", "half_year", "ttm"}:
                    signature = f.quarter if f.period_type in {"cumulative", "ttm"} else (
                        f.start[5:7] if f.period_type == "half_year" else None)
                    previous = by_year.get((f.fy - 1, signature))
                    if previous and by_year.get((f.fy, signature)) is f:
                        self._compare(f, previous, "yoy")
                else:
                    qindex = self._quarter_index(f, fiscal_end_month)
                    if qindex is None or by_qindex.get(qindex) is not f:
                        continue
                    previous = by_qindex.get(qindex - 1)
                    if previous:
                        self._compare(f, previous, "qoq")
                    previous_year = by_qindex.get(qindex - 4)
                    if previous_year:
                        self._compare(f, previous_year, "yoy")

        # 2-, 3- and 5-year compound growth on annual, positive-base values.
        annual_groups: dict[tuple, dict[int, Fact]] = defaultdict(dict)
        for f in facts:
            if f.period_type == "annual" and f.unit not in NONADDITIVE_UNITS:
                annual_groups[f.series][f.fy] = f
        for by_fy in annual_groups.values():
            for f in by_fy.values():
                for years in (2, 3, 5):
                    previous = by_fy.get(f.fy - years)
                    if not previous or f.value <= 0 or previous.value <= 0:
                        continue
                    answer = ((f.value / previous.value) ** (1 / years) - 1) * 100
                    self.add(f"{f.metric_key}_cagr_{years}y_pct", f"{f.label} {years}-year CAGR",
                             "growth", answer, "percent", f,
                             f"((current / {years}-year-earlier value) ** (1/{years}) - 1) * 100",
                             (f, previous), compared=previous.end)

    def trailing_twelve_months(self) -> list[Fact]:
        candidates: dict[tuple, dict[int, list[Fact]]] = defaultdict(lambda: defaultdict(list))
        for f in self.facts:
            if f.period_type == "quarterly" and f.quarter and _additive(f):
                candidates[f.series][f.fy * 4 + f.quarter].append(f)
        generated = []
        for possible in candidates.values():
            series = {key: rows[0] for key, rows in possible.items() if len(rows) == 1}
            for qindex, current in sorted(series.items()):
                window = [series.get(index) for index in range(qindex - 3, qindex + 1)]
                if any(f is None for f in window):
                    continue
                window = [f for f in window if f is not None]
                generated_fact = self.add(
                    current.metric_key + "_ttm", current.label + " trailing twelve months",
                    "ttm", sum(f.value for f in window), current.unit, current,
                    "sum of four consecutive, compatible standalone quarters",
                    window, start=window[0].start, period_type="ttm",
                    notes="TTM preserves reported/derived quarter values; it does not overwrite annual reporting",
                )
                if generated_fact:
                    generated.append(generated_fact)
        return generated

    @staticmethod
    def _choose(group: dict[str, Fact], concept: str) -> Fact | None:
        for key in KEYS[concept]:
            if key in group:
                return group[key]
        return None

    def _ratio(self, code: str, label: str, numerator: Fact | None, denominator: Fact | None,
               period: Fact, *, as_pct: bool = False, family: str = "ratios",
               numerator_abs: bool = False) -> Fact | None:
        if numerator is None or denominator is None:
            return None
        if (numerator.dimensions != denominator.dimensions or
                not _money(numerator.unit) or not _money(denominator.unit)):
            self.skips["incompatible_ratio_inputs"] += 1
            return None
        n = abs(numerator.value) if numerator_abs else numerator.value
        answer = _safe_div(n * (100 if as_pct else 1), denominator.value)
        return self.add(code, label, family, answer, "percent" if as_pct else "times",
                        period, f"{numerator.metric_key} / {denominator.metric_key}"
                        + (" * 100" if as_pct else "")
                        + (" (numerator cash outflow magnitude)" if numerator_abs else ""),
                        (numerator, denominator))

    def income_and_cashflow(self) -> None:
        """Same-period margins, expense ratios, cash conversion and FCF."""
        groups: dict[tuple, dict[str, Fact]] = defaultdict(dict)
        for f in self.facts:
            if f.period_type in {"annual", "quarterly", "cumulative", "half_year"}:
                groups[(f.period, f.scope, f.basis, f.currency, f.unit)][f.metric_key] = f
        for group in groups.values():
            sales = self._choose(group, "sales")
            cfo = self._choose(group, "cfo")
            pat = self._choose(group, "pat")
            pbt = self._choose(group, "pbt")
            ebit = self._choose(group, "ebit")
            ebitda = self._choose(group, "ebitda")
            if sales:
                # Only infer gross profit where a clearly identified cost-of-sales
                # line exists. Do not approximate it from total company expenses.
                gross_profit = self._choose(group, "gross_profit")
                cogs = self._choose(group, "cogs")
                if gross_profit is None and cogs and cogs.value >= 0 and sales.value >= 0:
                    gross_profit = self.add("calculated_gross_profit", "Gross profit (calculated)",
                                            "profitability", sales.value - cogs.value,
                                            sales.unit, sales,
                                            "revenue - positive reported cost of sales", (sales, cogs))
                for concept, code, label in (
                    ("gross_profit", "gross_margin_pct", "Gross profit margin"),
                    ("ebitda", "ebitda_margin_pct", "EBITDA margin"),
                    ("adjusted_ebitda", "adjusted_ebitda_margin_pct", "Adjusted EBITDA margin"),
                    ("ebit", "ebit_margin_pct", "EBIT margin"),
                    ("operating_profit", "operating_profit_margin_pct", "Operating profit margin"),
                    ("pbt", "pbt_margin_pct", "Profit before tax margin"),
                    ("pat", "pat_margin_pct", "Profit after tax margin"),
                    ("cfo", "operating_cash_flow_margin_pct", "Operating cash flow margin"),
                ):
                    numerator = gross_profit if concept == "gross_profit" else self._choose(group, concept)
                    self._ratio(code, label, numerator, sales, sales, as_pct=True, family="margins")
                for concept, code, label in (
                    ("cogs", "cogs_to_sales_pct", "Cost of sales / revenue"),
                    ("total_expenses", "total_expenses_to_sales_pct", "Total expenses / revenue"),
                    ("employee_expenses", "employee_expenses_to_sales_pct", "Employee expenses / revenue"),
                    ("marketing_expenses", "marketing_expenses_to_sales_pct", "Marketing expenses / revenue"),
                    ("finance_costs", "finance_costs_to_sales_pct", "Finance costs / revenue"),
                    ("other_expenses", "other_expenses_to_sales_pct", "Other expenses / revenue"),
                    ("depreciation", "depreciation_to_sales_pct", "Depreciation / revenue"),
                    ("other_income", "other_income_to_sales_pct", "Other income / revenue"),
                ):
                    self._ratio(code, label, self._choose(group, concept), sales, sales,
                                as_pct=True, family="cost_structure")
            if pbt:
                self._ratio("effective_tax_rate_pct", "Effective tax expense / PBT",
                            self._choose(group, "tax"), pbt, pbt, as_pct=True)
            if ebit:
                self._ratio("interest_coverage_times", "EBIT / finance costs", ebit,
                            self._choose(group, "finance_costs"), ebit)
            if cfo:
                self._ratio("operating_cash_flow_to_pat_times", "Operating cash flow / PAT",
                            cfo, pat, cfo, family="cash_flow")
                self._ratio("operating_cash_flow_to_ebitda_times", "Operating cash flow / EBITDA",
                            cfo, ebitda, cfo, family="cash_flow")
                capex = self._choose(group, "capex")
                if capex and capex.dimensions == cfo.dimensions and _money(capex.unit):
                    # This key explicitly represents capital spending; published
                    # cash-flow statements may represent the outflow as negative.
                    fcf = self.add("free_cash_flow", "Free cash flow", "cash_flow",
                                   cfo.value - abs(capex.value), cfo.unit, cfo,
                                   "operating cash flow - abs(capital expenditure)",
                                   (cfo, capex),
                                   notes="CAPEX uses cash-outflow magnitude; no inference from other investing flows")
                    if fcf and sales:
                        self._ratio("free_cash_flow_margin_pct", "Free cash flow margin",
                                    fcf, sales, sales, as_pct=True, family="cash_flow")

    def balance_sheet(self) -> None:
        groups: dict[tuple, dict[str, Fact]] = defaultdict(dict)
        for f in self.facts:
            if f.period_type == "point_in_time" and _money(f.unit):
                groups[(f.end, f.scope, f.basis, f.currency, f.unit)][f.metric_key] = f
        for group in groups.values():
            anchor = next(iter(group.values()))
            assets = self._choose(group, "current_assets")
            liabilities = self._choose(group, "current_liabilities")
            cash = self._choose(group, "cash")
            inventory = self._choose(group, "inventory")
            debt = self._choose(group, "debt")
            equity = self._choose(group, "equity")
            total_assets = self._choose(group, "assets")
            if assets and liabilities:
                self.add("net_working_capital", "Net working capital", "balance_sheet",
                         assets.value - liabilities.value, assets.unit, anchor,
                         "current assets - current liabilities", (assets, liabilities))
            self._ratio("current_ratio_times", "Current ratio", assets, liabilities,
                        anchor, family="balance_sheet")
            self._ratio("cash_ratio_times", "Cash ratio", cash, liabilities,
                        anchor, family="balance_sheet")
            if assets and inventory and liabilities:
                adjusted = self.add("quick_assets", "Current assets less inventories",
                                    "balance_sheet", assets.value - inventory.value, assets.unit,
                                    anchor, "current assets - inventories", (assets, inventory))
                if adjusted:
                    self._ratio("quick_ratio_times", "Quick ratio", adjusted,
                                liabilities, anchor, family="balance_sheet")
            self._ratio("debt_to_equity_times", "Debt / equity", debt, equity,
                        anchor, family="balance_sheet")
            self._ratio("debt_to_assets_pct", "Debt / total assets", debt,
                        total_assets, anchor, as_pct=True, family="balance_sheet")
            self._ratio("equity_to_assets_pct", "Equity / total assets", equity,
                        total_assets, anchor, as_pct=True, family="balance_sheet")
            if debt and cash:
                net_debt = self.add("net_debt", "Net debt", "balance_sheet",
                                    debt.value - cash.value, debt.unit, anchor,
                                    "total borrowings - cash and cash equivalents", (debt, cash))
                if net_debt:
                    self._ratio("net_debt_to_equity_times", "Net debt / equity",
                                net_debt, equity, anchor, family="balance_sheet")

    def annual_efficiency(self) -> None:
        """Return ratios and turnover use beginning/end snapshots, not closing-only."""
        annual: dict[tuple, dict[str, Fact]] = defaultdict(dict)
        snapshots: dict[tuple, dict[str, Fact]] = defaultdict(dict)
        for f in self.facts:
            if not _money(f.unit):
                continue
            dim = (f.scope, f.basis, f.currency, f.unit)
            if f.period_type == "annual":
                annual[(f.fy, f.end, dim)][f.metric_key] = f
            elif f.period_type == "point_in_time":
                snapshots[(f.end, dim)][f.metric_key] = f

        for (_, year_end, dim), p_and_l in annual.items():
            current_bs = snapshots.get((year_end, dim))
            if not current_bs:
                continue
            anchor = next(iter(p_and_l.values()))
            try:
                prev_end = date.fromisoformat(year_end).replace(year=date.fromisoformat(year_end).year - 1).isoformat()
            except ValueError:
                # Leap-day fiscal closes: match previous FY's last day of month.
                end = date.fromisoformat(year_end)
                from calendar import monthrange
                prev_end = date(end.year - 1, end.month, min(end.day, monthrange(end.year - 1, end.month)[1])).isoformat()
            previous_bs = snapshots.get((prev_end, dim))
            if not previous_bs:
                continue

            def avg_balance(concept: str) -> Fact | None:
                now = self._choose(current_bs, concept)
                before = self._choose(previous_bs, concept)
                if not now or not before:
                    return None
                return self.add(f"average_{concept}", f"Average {concept.replace('_', ' ')}",
                                "efficiency", (now.value + before.value) / 2,
                                now.unit, anchor, "(opening + closing balance) / 2",
                                (before, now))

            pat = self._choose(p_and_l, "pat")
            ebit = self._choose(p_and_l, "ebit")
            sales = self._choose(p_and_l, "sales")
            cogs = self._choose(p_and_l, "cogs")
            avg_equity = avg_balance("equity")
            avg_assets = avg_balance("assets")
            avg_receivables = avg_balance("receivables")
            avg_inventory = avg_balance("inventory")
            avg_payables = avg_balance("payables")
            self._ratio("return_on_equity_pct", "Return on average equity", pat,
                        avg_equity, anchor, as_pct=True, family="efficiency")
            self._ratio("return_on_assets_pct", "Return on average assets", pat,
                        avg_assets, anchor, as_pct=True, family="efficiency")
            self._ratio("asset_turnover_times", "Revenue / average assets", sales,
                        avg_assets, anchor, family="efficiency")

            ca_now = self._choose(current_bs, "assets")
            cl_now = self._choose(current_bs, "current_liabilities")
            ca_prev = self._choose(previous_bs, "assets")
            cl_prev = self._choose(previous_bs, "current_liabilities")
            if ca_now and cl_now and ca_prev and cl_prev:
                avg_cap = self.add("average_capital_employed", "Average capital employed",
                                   "efficiency", ((ca_now.value - cl_now.value)
                                                  + (ca_prev.value - cl_prev.value)) / 2,
                                   ca_now.unit, anchor,
                                   "average of (total assets - current liabilities) at FY beginning/end",
                                   (ca_prev, cl_prev, ca_now, cl_now))
                self._ratio("roce_pct", "Return on capital employed", ebit, avg_cap,
                            anchor, as_pct=True, family="efficiency")

            # Annual days use the actual reported annual period length.
            try:
                days = (date.fromisoformat(anchor.end) - date.fromisoformat(anchor.start)).days + 1
            except ValueError:
                days = 365
            values: dict[str, Fact] = {}
            for name, balance, flow, label in (
                ("receivable_days", avg_receivables, sales, "Receivable days"),
                ("inventory_days", avg_inventory, cogs, "Inventory days"),
                ("payable_days", avg_payables, cogs, "Payable days"),
            ):
                if not balance or not flow or balance.dimensions != flow.dimensions:
                    continue
                answer = _safe_div(balance.value * days, abs(flow.value))
                made = self.add(name, label, "efficiency", answer, "days", anchor,
                                f"average balance / abs(annual flow) * {days}",
                                (balance, flow), notes="Uses reported fiscal-year day count")
                if made:
                    values[name] = made
            if all(name in values for name in ("receivable_days", "inventory_days", "payable_days")):
                recv, inv, pay = (values[k] for k in ("receivable_days", "inventory_days", "payable_days"))
                self.add("cash_conversion_cycle_days", "Cash conversion cycle", "efficiency",
                         recv.value + inv.value - pay.value, "days", anchor,
                         "receivable days + inventory days - payable days", (recv, inv, pay))

    def run(self, fiscal_end_month: int = 3) -> list[dict[str, Any]]:
        self.comparisons(self.facts, fiscal_end_month)
        ttm = self.trailing_twelve_months()
        self.comparisons(ttm, fiscal_end_month)
        before = len(self.rows)
        self.income_and_cashflow()
        self.balance_sheet()
        self.annual_efficiency()
        # Percentage-point changes in computed margins/expense ratios/returns.
        derived = []
        for row in self.rows[before:]:
            if row["unit"] != "percent":
                continue
            derived.append(Fact(
                metric_key=row["metric_key"], label=row["metric_label"], statement="calculated",
                period_type=row["period_type"], fy=int(row["fiscal_year"]),
                quarter=_int(row["fiscal_quarter"]), start=row["period_start"],
                end=row["period_end"], scope=row["standard_scope"],
                basis=row["standard_basis"], unit="percent", currency="",
                value=float(row["value"]),
                evidence={"derivation": row["formula"],
                          "calculation_inputs": json.loads(row["input_facts_json"])},
            ))
        self.comparisons(derived, fiscal_end_month)
        self.rows.sort(key=lambda row: (row["period_end"], row["metric_key"],
                                        row["standard_scope"], row["standard_basis"],
                                        row["period_type"]))
        return self.rows


def calculate(input_path: Path, output_path: Path | None = None,
              summary_path: Path | None = None, fiscal_end_month: int = 3) -> tuple[Path, Path, dict]:
    if fiscal_end_month not in range(1, 13):
        raise ValueError("Fiscal year-end month must be between 1 and 12")
    facts, diagnostics = read_prepared(input_path)
    calculator = Calculator(facts)
    rows = calculator.run(fiscal_end_month)
    output_path = output_path or input_path.with_name("calculated_financial_metrics.csv")
    summary_path = summary_path or input_path.with_name("financial_calculation_summary.json")
    if output_path.resolve() == input_path.resolve() or summary_path.resolve() == input_path.resolve():
        raise ValueError("Calculation outputs must not overwrite the prepared input")
    if output_path.resolve() == summary_path.resolve():
        raise ValueError("CSV output and summary output must use different filenames")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.with_suffix(".tmp").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output_path.with_suffix(".tmp").replace(output_path)
    summary = {
        "source": str(input_path), **diagnostics, "calculations_created": len(rows),
        "calculations_by_family": dict(sorted(Counter(r["family"] for r in rows).items())),
        "skipped_operations": dict(calculator.skips),
        "distinct_source_metric_keys": len({f.metric_key for f in facts}),
        "distinct_calculated_metric_keys": len({r["metric_key"] for r in rows}),
        "fiscal_year_end_month": fiscal_end_month,
        "cautions": [
            "Annual and quarterly facts remain separate; source reconciliation discrepancies are not resolved here.",
            "Unusable or ambiguous upstream variants are excluded, rather than arbitrarily selected.",
            "Unavailable results are omitted; zero or negative denominators are not used for ratios/growth.",
            "Metrics with different reported/adjusted bases, scopes, currencies or units are not combined.",
            "Unstandardised unit labels can support exact-unit absolute changes, but not percentage growth or financial ratios.",
            "Custom operating metric growth is calculated where comparable; segment contribution requires explicit segment metadata.",
        ],
    }
    temp_summary = summary_path.with_suffix(".tmp")
    temp_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_summary.replace(summary_path)
    return output_path, summary_path, summary


def _latest_prepared_run(root: Path) -> Path:
    candidates = list(root.glob("*/final_prepared_consolidated_facts.csv"))
    if not candidates:
        raise FileNotFoundError("No data/runs/<run>/final_prepared_consolidated_facts.csv found")
    return max(candidates, key=lambda p: p.stat().st_mtime).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic financial calculation engine")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--run-dir", type=Path, help="Extraction run containing prepared facts")
    selection.add_argument("--input", type=Path, help="Exact prepared CSV path")
    parser.add_argument("--output", type=Path, help="Optional calculated CSV output path")
    parser.add_argument("--summary", type=Path, help="Optional calculation summary JSON path")
    parser.add_argument("--fiscal-end-month", type=int, default=None,
                        help="Override manifest fiscal-year end month (1-12)")
    args = parser.parse_args()
    run_dir = args.run_dir or (args.input.parent if args.input else _latest_prepared_run(Path("data/runs")))
    input_path = args.input or run_dir / "final_prepared_consolidated_facts.csv"
    manifest = run_dir / "extraction_manifest.json"
    fiscal_end = 3
    if manifest.is_file():
        fiscal_end = int(json.loads(manifest.read_text(encoding="utf-8")).get("fiscal_end_month", 3))
    if args.fiscal_end_month is not None:
        fiscal_end = args.fiscal_end_month
    output, summary_path, report = calculate(input_path, args.output, args.summary, fiscal_end)
    print(f"Prepared input: {input_path}")
    print(f"Calculated financial metrics: {output}")
    print(f"Calculation summary: {summary_path}")
    print(f"Summary: {json.dumps({k: report[k] for k in ('accepted_facts', 'calculations_created', 'calculations_by_family', 'annual_quarter_mismatches')})}")
    print("Original extracted, standardised and prepared facts were not modified.")


if __name__ == "__main__":
    main()
