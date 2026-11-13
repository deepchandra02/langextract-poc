"""python -m form_extract samples/*.pdf --out output/ [--no-llm]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .acroform import match_labels, read_widgets
from .hints import build_blocks
from .layout import build_layout
from .lx_extract import DEFAULT_MODEL, run_langextract
from .reconcile import reconcile
from .schema import FormStructure, Item, Section

MARK = {"agree": "", "disagree": "  [!layout/LLM disagree]", "llm_missing": "  [layout only: LLM missed]", "layout_only": ""}


def _line(it: Item) -> str:
    text = " ".join(it.text.split())
    fields = f"  <{', '.join(f.type + ':' + f.name for f in it.fields)}>" if it.fields else ""
    attrs = f"  {it.attributes}" if it.attributes else ""
    return f"{it.type}: {text}{attrs}{fields}{MARK[it.agreement]}"


def outline(fs: FormStructure) -> str:
    out = [f"{fs.source_file}  ({fs.mode}{', ' + fs.model_id if fs.model_id else ''}, {fs.page_count} pages)"]
    if fs.issuer:
        out.append("Issuer: " + " | ".join(" ".join(i.text.split()) for i in fs.issuer))

    def walk(s: Section, depth: int) -> None:
        pad = "  " * depth
        t = s.title
        fields = f"  <{', '.join(f.type + ':' + f.name for f in t.fields)}>" if t.fields else ""
        out.append(f"{pad}{'#' * (s.level + 1)} {' '.join(t.text.split())}{fields}{MARK[t.agreement]}")
        out.extend(f"{pad}  - {_line(i)}" for i in s.items)
        for c in s.subsections:
            walk(c, depth + 1)

    for p in fs.parts:
        title = " ".join(p.title.text.split()) if p.title else "(untitled)"
        out.append(f"\n# {title}   [{p.kind}, form {p.form_number} of {p.version_date}, pages {p.pages}]")
        out.extend(f"  - {_line(i)}" for i in p.preamble)
        for s in p.sections:
            walk(s, 1)
    if fs.footnotes:
        out.append("\nFootnotes:")
        out.extend(f"  - {' '.join(f.text.split())}" for f in fs.footnotes)
    d = fs.diagnostics
    out.append(
        f"\nDiagnostics: llm accepted {d.llm_extractions_accepted}/{d.llm_extractions_total}, "
        f"rejected {len(d.rejected)}, disagreements {len(d.layout_disagreements)}, "
        f"missed_by_llm {len(d.missed_by_llm)}, unlabelled fields {len(d.unlabelled_fields)}"
    )
    return "\n".join(out)


def process(pdf: Path, out_dir: Path, *, use_llm: bool, model_id: str, passes: int, model=None) -> FormStructure:
    layout = build_layout(pdf)
    widgets = read_widgets(pdf)
    match_labels(widgets, layout)
    blocks = build_blocks(layout, widgets)
    extractions = None
    if use_llm:
        doc = None
        for candidate in model_id.split(","):  # comma-separated fallback list
            try:
                doc = run_langextract(layout.text, model_id=candidate.strip(), passes=passes, model=model)
                model_id = candidate.strip()
                break
            except Exception as e:  # noqa: BLE001 - provider raises various transient errors
                print(f"[{pdf.name}] {candidate.strip()} failed: {str(e)[:160]}")
        if doc is None:
            raise RuntimeError(f"all models failed for {pdf.name}")
        extractions = list(doc.extractions or [])
        import langextract as lx

        lx.io.save_annotated_documents([doc], output_dir=str(out_dir), output_name=f"{pdf.stem}.lx.jsonl",
                                       show_progress=False)
        html = lx.visualize(str(out_dir / f"{pdf.stem}.lx.jsonl"))
        (out_dir / f"{pdf.stem}.lx.html").write_text(getattr(html, "data", html), encoding="utf-8")
    fs = reconcile(layout, widgets, blocks, extractions, model_id=model_id)
    (out_dir / f"{pdf.stem}.json").write_text(fs.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / f"{pdf.stem}.outline.txt").write_text(outline(fs) + "\n", encoding="utf-8")
    return fs


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (KEY=VALUE lines); real environment variables win."""
    import os

    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
    ap = argparse.ArgumentParser(description="Extract grounded form structure from PDFs.")
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--no-llm", action="store_true", help="layout-only mode (no API key needed)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="model id, or comma-separated fallback list")
    ap.add_argument("--passes", type=int, default=2, help="LangExtract extraction passes (recall)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for pdf in args.pdfs:
        try:
            fs = process(pdf, args.out, use_llm=not args.no_llm, model_id=args.model, passes=args.passes)
        except RuntimeError as e:
            failed.append(pdf.name)
            print(f"SKIPPED {pdf.name}: {e}\n")
            continue
        print(outline(fs), end="\n\n")
    if failed:
        raise SystemExit(f"failed: {failed}")
    print(json.dumps({"written_to": str(args.out)}))


if __name__ == "__main__":
    main()
