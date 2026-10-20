"""Inventory local literature and extract searchable evidence; never edit sources."""

import hashlib
import io
import json
import re
import sys
import zipfile
from difflib import SequenceMatcher
from pathlib import Path

import openpyxl
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "working"


def normalized(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    book = openpyxl.load_workbook(ROOT / "Literature_Analysis.xlsx", read_only=True, data_only=True)
    rows = list(book["Master Literature Database"].values)
    records = [dict(zip(rows[0], row)) for row in rows[1:]]
    register = openpyxl.load_workbook(ROOT / "Analyzed_Papers.xlsx", read_only=True, data_only=True)
    register_rows = list(register["Analyzed Papers"].values)
    (OUT / "workbook_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT / "register_rows.json").write_text(json.dumps(register_rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    items, seen = [], {}

    def inspect(name, data, source):
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            seen[digest]["duplicate_locations"].append(source)
            return
        stem = normalized(re.sub(r"^\d+\s*-\s*", "", Path(name).stem))
        matches = [(SequenceMatcher(None, stem, normalized(r["Title"])).ratio(), r) for r in records]
        score, match = max(matches, key=lambda v: v[0])
        item = {"key": f"pdf_{len(items)+1:03}", "filename": name, "source": source, "sha256": digest,
                "literature_ids": [], "matched_title": None, "match_score": round(score, 3), "duplicate_locations": []}
        if score >= .82:
            item["matched_title"] = match["Title"]
            item["literature_ids"] = [r["Paper No"] for r in records if normalized(r["Title"]) == normalized(match["Title"])]
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [p.extract_text() or "" for p in reader.pages]
            item["pages"] = len(pages)
            item["first_page"] = pages[0][:1600] if pages else ""
            item["dfdc_pages"] = [i+1 for i,p in enumerate(pages) if re.search(r"\bDFDC\b|deep\s*fake\s*detection\s*challenge", p, re.I)]
            (OUT / (item["key"] + ".json")).write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        seen[digest] = item
        items.append(item)

    for path in sorted((ROOT / "papers").glob("*.pdf")):
        inspect(path.name, path.read_bytes(), str(path))
    for path in [ROOT / "sources" / "Additional_Papers.zip", *sorted((ROOT / "archive").rglob("*.zip"))]:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.filename.lower().endswith(".pdf"):
                    inspect(info.filename, archive.read(info), str(path) + "!" + info.filename)
    (OUT / "paper_inventory.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"unique_pdfs": len(items), "dfdc_mentions": sum(bool(x.get("dfdc_pages")) for x in items), "errors": [x["filename"] for x in items if "error" in x]}))
    for x in items:
        if x.get("dfdc_pages"):
            print(x["key"], x["literature_ids"], x["filename"], "DFDC pages", x["dfdc_pages"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
