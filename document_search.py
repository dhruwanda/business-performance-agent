
"""Read-only retrieval of passages from PDFs already downloaded for a company run.

Uses the run's extraction_manifest.json to identify the documents and preserve
their original URLs. No vector database, embeddings, new downloads or LLM calls.
Optional --documents-dir enables searching additional local PDFs/text files.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha1
import argparse
import json
import math
from pathlib import Path
import re
from typing import Any


_STOPWORDS = {
    "a", "about", "after", "all", "and", "are", "as", "at", "be", "been",
    "between", "by", "can", "company", "compared", "did", "do", "does", "during",
    "explain", "for", "from", "had", "has", "have", "how", "in", "into", "is",
    "it", "its", "of", "on", "or", "our", "over", "the", "their", "there", "these",
    "this", "those", "to", "was", "were", "what", "when", "which", "while",
    "why", "with", "year", "years", "fy", "q", "that", "trend", "trends",
}

# Small, explicit synonym groups help recall without an embedding service.
_RELATED = (
    {"advertising", "advertisement", "marketing", "promotion", "promotional", "brand", "campaign", "media"},
    {"profit", "profits", "profitability", "ebitda", "margin", "margins", "earnings"},
    {"revenue", "revenues", "sales", "turnover", "topline"},
    {"distribution", "distributor", "channel", "channels", "retail", "offline"},
    {"inventory", "stock", "stocks", "destocking"},
    {"premiumisation", "premiumization", "premium", "pricing", "price"},
    {"commerce", "ecommerce", "online", "quickcommerce"},
)


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z][a-z0-9]+", value.lower())
            if token not in _STOPWORDS and len(token) > 2}


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _chunks(text: str, size: int = 950, overlap: int = 160):
    """Overlapping page-local windows, preserving each PDF page number."""
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer a sentence boundary when possible.
            boundary = max(text.rfind(". ", start + 450, end),
                           text.rfind("; ", start + 450, end))
            if boundary > start:
                end = boundary + 1
        yield text[start:end].strip()
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


@dataclass(frozen=True)
class DocumentInfo:
    id: str
    title: str
    kind: str
    period: str
    url: str
    path: Path
    page_cache: Path | None = None


@dataclass(frozen=True)
class Passage:
    document_id: str
    page: int
    text: str
    terms: frozenset[str]


class DocumentSearchStore:
    """Search only a chosen run's locally saved, identified source documents."""

    def __init__(self, run_dir: str | Path, documents_dir: str | Path | None = None,
                 project_root: str | Path | None = None):
        self.run_dir = Path(run_dir).resolve()
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.documents_dir = Path(documents_dir).resolve() if documents_dir else None
        self._loaded = False
        self.documents: dict[str, DocumentInfo] = {}
        self.passages: list[Passage] = []
        self.page_text: dict[tuple[str, int], str] = {}
        self.warnings: list[str] = []
        self.pages_without_text = 0

    def _allowed(self, candidate: Path) -> bool:
        allowed = [self.project_root, self.run_dir]
        if self.documents_dir:
            allowed.append(self.documents_dir)
        return any(candidate.is_relative_to(base) for base in allowed)

    def _resolve_saved(self, saved: str) -> Path | None:
        if not saved:
            return None
        name = saved.replace("\\", "/")
        path = Path(name)
        candidates = [path, self.project_root / path, self.run_dir / path,
                      self.project_root / "data" / "cache" / "pdfs" / path.name,
                      self.run_dir / path.name]
        if self.documents_dir:
            candidates.append(self.documents_dir / path.name)
        for item in candidates:
            resolved = item.resolve()
            if self._allowed(resolved) and resolved.is_file() and resolved.suffix.lower() in (".pdf", ".txt", ".md"):
                return resolved
        return None

    def _locate_documents(self) -> list[DocumentInfo]:
        found: list[DocumentInfo] = []
        seen_paths: set[Path] = set()
        manifest = self.run_dir / "extraction_manifest.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                self.warnings.append(f"Could not read extraction manifest: {exc}")
                data = {}
            for original_url, result in data.get("completed", {}).items():
                if not isinstance(result, dict):
                    continue
                source = result.get("document") or {}
                if not isinstance(source, dict):
                    source = {}
                saved_pdf = str(result.get("saved_pdf") or "")
                saved_text = str(result.get("saved_text") or "")
                saved = saved_pdf or saved_text
                path = self._resolve_saved(saved_pdf) or self._resolve_saved(saved_text)
                page_cache = None
                if saved_pdf and Path(saved_pdf.replace("\\", "/")).suffix.lower() == ".pdf":
                    pdf_name = Path(saved_pdf.replace("\\", "/")).name
                    page_cache = self.run_dir / "text" / f"{Path(pdf_name).stem}.json"
                if path is None and page_cache is not None and page_cache.is_file():
                    path = self.run_dir / "pdfs" / pdf_name
                if path is None:
                    self.warnings.append(f"Local file missing: {source.get('title') or saved or original_url}")
                    continue
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                url = str(source.get("url") or original_url)
                doc_id = sha1((url + str(path)).encode("utf-8")).hexdigest()[:12]
                found.append(DocumentInfo(
                    doc_id, str(source.get("title") or path.stem),
                    str(source.get("document_type") or "document"),
                    str(source.get("period") or ""), url, path, page_cache,
                ))

        # Additional files are opt-in. Never silently mix another company's PDFs
        # from a shared global cache with the selected financial run.
        if self.documents_dir and self.documents_dir.is_dir():
            for path in sorted(self.documents_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in (".pdf", ".txt", ".md"):
                    continue
                path = path.resolve()
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                found.append(DocumentInfo(sha1(str(path).encode()).hexdigest()[:12],
                                          path.stem, "local document", "", "", path))
        return found

    def _load(self) -> None:
        if self._loaded:
            return
        located = self._locate_documents()
        if located:
            print(f"Indexing searchable text from {len(located)} local document(s)...", flush=True)
        for info in located:
            self.documents[info.id] = info
            if info.page_cache is not None and info.page_cache.is_file():
                try:
                    raw_pages = json.loads(info.page_cache.read_text(encoding="utf-8"))
                    if not isinstance(raw_pages, list) or not raw_pages:
                        raise ValueError("Expected a nonempty list of pages")
                    page_texts = []
                    seen_pages = set()
                    for entry in raw_pages:
                        if not isinstance(entry, dict):
                            raise ValueError("Invalid page entry")
                        number, content = entry.get("page"), entry.get("text")
                        if (type(number) is not int or number < 1 or number in seen_pages
                                or not isinstance(content, str)):
                            raise ValueError("Invalid or duplicate page number or text")
                        seen_pages.add(number)
                        page_texts.append((number, content))
                    self._index_pages(info, page_texts)
                    continue
                except (OSError, UnicodeError, ValueError, TypeError) as exc:
                    self.warnings.append(f"Unable to use cached text for {info.title}: {exc}")
            try:
                if info.path.suffix.lower() == ".pdf":
                    try:
                        import pymupdf
                    except ImportError:
                        try:
                            import fitz as pymupdf
                        except ImportError as exc:
                            raise RuntimeError("PDF fallback requires pymupdf.") from exc
                    with pymupdf.open(info.path) as pdf:
                        page_texts = ((n + 1, page.get_text("text", sort=True))
                                      for n, page in enumerate(pdf))
                        self._index_pages(info, page_texts)
                else:
                    text = info.path.read_text(encoding="utf-8", errors="replace")
                    self._index_pages(info, [(1, text)])
            except (OSError, ValueError, RuntimeError) as exc:
                self.warnings.append(f"Unable to read {info.title}: {exc}")
        self._loaded = True
        if located:
            print(f"Document index ready: {len(self.passages)} passages; "
                  f"{self.pages_without_text} pages without extractable text.", flush=True)

    def _index_pages(self, doc: DocumentInfo, pages) -> None:
        for number, raw in pages:
            cleaned = _normalise_text(raw)
            if len(cleaned) < 30:
                self.pages_without_text += 1
                continue
            self.page_text[(doc.id, number)] = cleaned
            for part in _chunks(cleaned):
                self.passages.append(Passage(doc.id, number, part, frozenset(_tokens(part))))

    def catalog(self, limit: int = 30) -> dict[str, Any]:
        """List searchable local documents and report missing or image-only PDFs."""
        self._load()
        rows = [{"document_id": d.id, "title": d.title, "type": d.kind,
                 "period": d.period, "source_url": d.url,
                 "pages_with_text": sum(key[0] == d.id for key in self.page_text)}
                for d in self.documents.values()]
        return {"documents": rows[:max(1, min(limit, 80))], "documents_found": len(rows),
                "searchable_passages": len(self.passages),
                "pages_without_extractable_text": self.pages_without_text,
                "warnings": self.warnings[:12],
                "note": "Scanned/image-only pages need OCR or prior text extraction. Unlisted documents are not searched."}

    def search(self, query: str, document_type: str = "", limit: int = 5) -> dict[str, Any]:
        """Return a few relevant passages, each tied to an original document and page."""
        self._load()
        requested = _tokens(query)
        if not requested:
            return {"error": "Please provide specific search terms."}
        expanded = set(requested)
        for group in _RELATED:
            if requested & group:
                expanded.update(group)
        candidates = [(p, self.documents[p.document_id]) for p in self.passages
                      if not document_type or document_type.lower() in self.documents[p.document_id].kind.lower()]
        document_frequency: Counter[str] = Counter()
        for p, _ in candidates:
            document_frequency.update(p.terms & expanded)
        n = len(candidates)
        ranked: list[tuple[float, Passage, DocumentInfo]] = []
        for passage, document in candidates:
            matched = passage.terms & expanded
            if not matched:
                continue
            direct_matches = len(passage.terms & requested)
            # Give rare exact query terms more weight than broad synonyms.
            score = sum(math.log(1 + (n - document_frequency[t] + 0.5) /
                                 (document_frequency[t] + 0.5)) *
                        (2.0 if t in requested else 0.65) for t in matched)
            score += 1.4 * direct_matches / max(1, len(requested))
            if direct_matches >= 2:
                score *= 1.25
            if any(word in query.lower() for word in ("why", "reason", "driver", "explain")):
                if any(word in document.kind.lower() for word in ("presentation", "transcript", "earnings", "investor")):
                    score *= 1.12
            ranked.append((score, passage, document))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = []
        per_document: Counter[str] = Counter()
        used_pages: set[tuple[str, int]] = set()
        for score, passage, doc in ranked:
            if per_document[doc.id] >= 2 or (doc.id, passage.page) in used_pages:
                continue
            per_document[doc.id] += 1
            used_pages.add((doc.id, passage.page))
            selected.append({"document_id": doc.id, "title": doc.title,
                             "document_type": doc.kind, "period": doc.period,
                             "source_url": doc.url, "pdf_page": passage.page,
                             "passage": passage.text})
            if len(selected) >= max(1, min(limit, 8)):
                break
        result = {"query": query, "results": selected, "matches": len(ranked),
                "documents_searched": len({d.id for _, d in candidates}),
                "note": ("No matching passage was found in searchable text. "
                         "This does not prove the topic is absent from image-only pages."
                         if not selected else "PDF pages are 1-indexed; quote sources with their URLs and page numbers.")}
        if not self.documents:
            result["warning"] = ("No local documents were indexed for this run. "
                                 "Check extraction_manifest.json and cached PDFs, "
                                 "or supply --documents-dir for additional files.")
        elif self.warnings:
            result["file_warnings"] = self.warnings[:5]
        return result

    def page(self, document_id: str, page: int, max_chars: int = 5000) -> dict[str, Any]:
        """Read a complete indexed page when a snippet needs more context."""
        self._load()
        doc = self.documents.get(document_id)
        if not doc:
            return {"error": "Unknown document_id. Use list_local_documents first."}
        content = self.page_text.get((document_id, page))
        if not content:
            return {"error": "This page has no searchable text or is out of range.",
                    "title": doc.title, "pdf_page": page}
        maximum = max(500, min(max_chars, 7000))
        return {"title": doc.title, "source_url": doc.url, "pdf_page": page,
                "page_text": content[:maximum], "truncated": len(content) > maximum}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test local document search without an LLM or API calls.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--documents-dir", type=Path, default=None)
    parser.add_argument("--query", default="advertising marketing profitability")
    args = parser.parse_args()

    store = DocumentSearchStore(args.run_dir, documents_dir=args.documents_dir)
    catalog = store.catalog()
    print(f"Found {catalog['documents_found']} local documents, "
          f"{catalog['searchable_passages']} searchable passages, "
          f"{catalog['pages_without_extractable_text']} image-only/empty pages.")
    if catalog["warnings"]:
        print("Warnings:", *catalog["warnings"], sep="\n  - ")
    print(json.dumps(store.search(args.query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
