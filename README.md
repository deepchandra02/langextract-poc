# LangExtract POC: grounded structure extraction from PDF forms

Turns a PDF form into structured JSON: **title, issuer, sections, sub-sections, field labels (mapped to the
PDF's fillable fields), checkbox options, instructions, required documents, legal text and footnotes**.
Every output string is traceable to an exact character span, page and bounding box in the PDF.

Samples: three UBS Vested Benefits Foundation forms in `samples/`.

## Why hybrid, and what LangExtract does and doesn't buy you

[LangExtract](https://github.com/google/langextract) aligns every LLM extraction back to a character interval in
the source text. That makes hallucination **detectable**, not impossible:

| Risk | Does LangExtract alone catch it? | What this POC adds |
|---|---|---|
| Invented text ("IBAN of the spouse") | Yes: `char_interval = None` | Rejected and reported |
| Paraphrased text ("Details of the account holder") | **No.** Fuzzy alignment maps it onto a real span and returns `match_fuzzy` | Re-slice the source at the span and require equality (whitespace-normalised). Output text is always the source slice, never the model's string |
| Invented attribute values | No: attributes are not grounded | Attribute values must be substrings of their extraction |
| Real text, wrong role (a label called a section) | No: it only sees plain text | Cross-check against PDF layout (font size, bold, indentation) |
| Text silently skipped | No | Every block not fully covered by the LLM is kept as a `llm_missing` item |

Note that `match_fuzzy` doesn't always mean a hallucination: the model emitting real text out of order also
produces it. That's why the gate checks the slice instead of trusting the status.

These PDFs carry strong visual structure that plain text loses. Titles are 17–18pt, section headings are bold at
the left margin, and payout options are bold and indented with a checkbox. They also have AcroForm fields. So the
pipeline is:

```
PDF ─┬─ layout.py    pdfplumber: lines + font/size/bold/bbox → ordered source text (noise removed)
     ├─ acroform.py  pypdf widgets → matched to their label line (checkbox→right, text box→below/left)
     ├─ hints.py     deterministic roles: title, section_heading, option_heading, field_label, choice, ...
     ├─ lx_extract.py LangExtract (Gemini) classifies the same text → grounded extractions
     └─ reconcile.py  grounding gate + layout cross-check + tree → FormStructure JSON
```

**Rule for structure:** a heading shapes the tree only with visual evidence. The LLM can *demote* a bold line it
judges to be an instruction, but it cannot *promote* a plain line to a heading. Every disagreement is listed
in `diagnostics.layout_disagreements`.

## Setup and run

```bash
cd langextract-poc
pip install -e '.[test]'

# Layout-only: deterministic, no API key
python -m form_extract samples/*.pdf --out output/layout-only --no-llm

# Hybrid: LangExtract + Gemini
cp .env.example .env                  # then put your key in .env (or export LANGEXTRACT_API_KEY)
python -m form_extract samples/authorisation-to-inform-en.pdf --out output/hybrid \
    --model gemini-3.5-flash,gemini-3.8-flash,gemini-flash-latest   # comma-separated fallbacks

pytest                                # 16 tests, no API key needed
```

Per PDF it writes `<name>.json` (full structure), `<name>.outline.txt` (human-readable tree), and in hybrid mode
`<name>.lx.jsonl` plus `<name>.lx.html` (LangExtract's highlighted-span viewer).

## Output shape (abridged)

```jsonc
{
  "title": "Payout UBS Vested benefits account",
  "issuer": [{"type": "issuer", "text": "Vested Benefits Foundation of UBS AG", ...}],
  "parts": [
    {"kind": "form", "form_number": "61073", "version_date": "20.03.2026", "pages": [1,2,3],
     "title": {...},
     "sections": [
       {"title": {"text": "Reason for payout (specifying the documents that must be submitted)", ...}, "level": 1,
        "subsections": [
          {"title": {"text": "Disability", "fields": [{"name": "Invalidität", "type": "checkbox"}]}, "level": 2,
           "items": [{"type": "instruction", "text": "Condition: Recipient of a full disability pension ..."}]}]}]},
    {"kind": "attachment", "form_number": "63652", "title": {"text": "Information sheet \"Payout UBS vested benefits account\""}, ...}
  ],
  "footnotes": [...],
  "diagnostics": {"rejected": [...], "layout_disagreements": [...], "missed_by_llm": [...], "dropped_noise": [...]},
  "source_text": "..."   // all char_spans index into this
}
```

Every `Item` has `type`, `text`, `layout_role`, `llm_class`, `agreement`
(`agree | disagree | llm_missing | layout_only`), `attributes`, `fields` (AcroForm), and `provenance`
(`char_span`, `page`, `bbox`, `font_size`, `bold`).

## Status

- Layout-only output for all three samples is committed in `output/layout-only/`.
- Hybrid output (gemini-3.5-flash, 1 pass) for the authorisation form is in `output/hybrid/`: 38/38 extractions
  grounded, 0 rejected, 2 layout disagreements. Free-tier Gemini returns 503 "high demand" in bursts, so the run
  took ~6 minutes with retries. Run one PDF at a time.
- The hybrid pipeline has also been run end to end with a **stub model** (see `tests/test_pipeline.py`). The stub
  hallucinates, paraphrases, adds unsupported attributes, and tries to promote and demote headings, and every case
  is caught.

## Known limitations

- Payout form, page 1: an odd glyph layout in the PDF merges "Copy of an official, signed identity document of
  the pension account holder" into the preceding list item.
- Signature lines without a fillable field (e.g. "Signature of the authorised representative") are `paragraph` in
  layout-only mode. The LLM is expected to classify them as `field_label`.
- Heuristics (column gap, margins, value-over-label captions) are tuned on the UBS forms. Other form families will
  need checking.
- Text in widget appearance streams is only recovered for read-only fields with a `/V` value.
