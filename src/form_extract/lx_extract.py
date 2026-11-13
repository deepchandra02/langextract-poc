"""LangExtract step: classify the form text into structural roles, grounded to char spans."""

from __future__ import annotations

import os
from typing import Any

import langextract as lx

DEFAULT_MODEL = "gemini-3.5-flash"

CLASSES = {
    "form_title": "the main title of the form or of an attached information sheet",
    "section_title": "a heading that opens a section of the form",
    "subsection_title": "a heading inside a section, e.g. one selectable reason/option that has its own conditions",
    "field_label": "the caption of something the user fills in, ticks or signs (incl. signature/place/date lines)",
    "checkbox_option": "a short answer choice such as Yes / No (with any parenthetical consequence)",
    "prefilled_value": "a value already printed on the form, e.g. a fixed bank name or IBAN",
    "instruction": "guidance telling the reader what to do or what applies (incl. 'Condition: ...' lines)",
    "required_document": "a document the user must enclose or submit",
    "legal_text": "terms, declarations, authorisations, governing-law or liability wording",
    "footnote": "footnote text",
}

PROMPT = f"""\
You are extracting the STRUCTURE of a printed PDF form whose text is given line by line, in reading order.
Extract each structural element in order of appearance, using one of these classes:
{chr(10).join(f"- {k}: {v}" for k, v in CLASSES.items())}

Strict rules:
- extraction_text MUST be copied character-for-character from the input. Never paraphrase, translate,
  normalise, fix typos, or invent text. If an element is not literally in the input, do not output it.
- Extract whole lines (or whole wrapped sentences/paragraphs); do not cut a label in half.
- Do not extract the sender letterhead or the postal address block.
- Attributes are optional and their values MUST also be verbatim substrings of the extraction_text:
  - format_hint (field_label): a format pattern in the label, e.g. "dd.mm.yyyy".
  - consequence (checkbox_option): parenthetical consequence, e.g. "payout not possible".
  - validity (required_document): an age limit, e.g. "not older than three months".
"""

_EX_TEXT = """\
Acme Pension Fund
Application for change of address
Details of insured person¹
Last name
First name(s)
Date of birth (dd.mm.yyyy)
Reason for change
Move within Switzerland
Condition: The new address is in Switzerland.
– Enclose a copy of the residence certificate (not older than three months)
– I still work in Switzerland.
Yes
No (change not possible)
Payment details
CH00 1234 5678 9012 3456 7
IBAN
Declaration
The insured person confirms that the information provided is correct and complete.
Place
Date
Signature of insured person
¹ The singular form also includes the plural."""


def _ex(cls: str, text: str, **attrs: str) -> lx.data.Extraction:
    assert text in _EX_TEXT, text  # keep the few-shot example itself grounded
    for v in attrs.values():
        assert v in text, v
    return lx.data.Extraction(extraction_class=cls, extraction_text=text, attributes=attrs or None)


EXAMPLES = [
    lx.data.ExampleData(
        text=_EX_TEXT,
        extractions=[
            _ex("form_title", "Application for change of address"),
            _ex("section_title", "Details of insured person¹"),
            _ex("field_label", "Last name"),
            _ex("field_label", "First name(s)"),
            _ex("field_label", "Date of birth (dd.mm.yyyy)", format_hint="dd.mm.yyyy"),
            _ex("section_title", "Reason for change"),
            _ex("subsection_title", "Move within Switzerland"),
            _ex("instruction", "Condition: The new address is in Switzerland."),
            _ex(
                "required_document",
                "– Enclose a copy of the residence certificate (not older than three months)",
                validity="not older than three months",
            ),
            _ex("field_label", "– I still work in Switzerland."),
            _ex("checkbox_option", "Yes"),
            _ex("checkbox_option", "No (change not possible)", consequence="change not possible"),
            _ex("section_title", "Payment details"),
            _ex("prefilled_value", "CH00 1234 5678 9012 3456 7"),
            _ex("field_label", "IBAN"),
            _ex("section_title", "Declaration"),
            _ex("legal_text", "The insured person confirms that the information provided is correct and complete."),
            _ex("field_label", "Place"),
            _ex("field_label", "Date"),
            _ex("field_label", "Signature of insured person"),
            _ex("footnote", "¹ The singular form also includes the plural."),
        ],
    )
]


def run_langextract(
    text: str,
    *,
    model_id: str = DEFAULT_MODEL,
    api_key: str | None = None,
    model: Any | None = None,
    passes: int = 2,
    max_char_buffer: int = 1500,
    max_workers: int = 2,
) -> lx.data.AnnotatedDocument:
    """Run LangExtract over the layout text. `model` lets tests inject a stub."""
    kwargs: dict[str, Any] = dict(
        text_or_documents=text,
        prompt_description=PROMPT,
        examples=EXAMPLES,
        extraction_passes=passes,
        max_char_buffer=max_char_buffer,
        max_workers=max_workers,
        show_progress=False,
    )
    if model is not None:
        kwargs["model"] = model
    else:
        key = api_key or os.environ.get("LANGEXTRACT_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit(
                "No Gemini API key: set LANGEXTRACT_API_KEY (or GEMINI_API_KEY), or run with --no-llm."
            )
        kwargs.update(
            model_id=model_id,
            api_key=key,
            temperature=0.0,
            # free-tier Gemini returns 503 "high demand" in bursts; back off patiently
            language_model_params={"max_retries": 8, "retry_delay": 5.0, "max_retry_delay": 60.0},
        )
    return lx.extract(**kwargs)
