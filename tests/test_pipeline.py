"""Tests that run without an API key.

* Layout tests check deterministic structure on the real sample PDFs.
* Grounding tests push the *real* LangExtract pipeline (prompting, chunking,
  alignment) through a stub model that deliberately misbehaves, and check the
  reconcile gate catches every kind of hallucination.
"""

import json
import re
from pathlib import Path

import pytest
from langextract.core.base_model import BaseLanguageModel
from langextract.core.types import ScoredOutput

from form_extract.acroform import match_labels, read_widgets
from form_extract.hints import build_blocks
from form_extract.layout import build_layout
from form_extract.lx_extract import run_langextract
from form_extract.reconcile import norm, reconcile

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
AUTH = SAMPLES / "authorisation-to-inform-en.pdf"
ORDER = SAMPLES / "order-to-transfer-retirement-assets-to-ubs-en.pdf"
PAYOUT = SAMPLES / "payout-vested-benefits-account-en.pdf"


def structure(pdf, extractions=None):
    layout = build_layout(pdf)
    widgets = read_widgets(pdf)
    match_labels(widgets, layout)
    return reconcile(layout, widgets, build_blocks(layout, widgets), extractions, model_id="stub")


def all_items(fs):
    out = list(fs.issuer) + list(fs.recipient_address) + list(fs.footnotes)
    for p in fs.parts:
        out += ([p.title] if p.title else []) + p.preamble
        stack = list(p.sections)
        while stack:
            s = stack.pop()
            out += [s.title, *s.items]
            stack += s.subsections
    return out


def section_titles(fs):
    titles = []

    def walk(s):
        titles.append(norm(s.title.text))
        for c in s.subsections:
            walk(c)

    for p in fs.parts:
        for s in p.sections:
            walk(s)
    return titles


# ---------------------------------------------------------------- layout-only


@pytest.mark.parametrize(
    "pdf,title",
    [
        (AUTH, "Authorisation to give information"),
        (ORDER, "Order to transfer retirement assets to the Vested Benefits Foundation of UBS AG"),
        (PAYOUT, "Payout UBS Vested benefits account"),
    ],
)
def test_titles(pdf, title):
    assert norm(structure(pdf).title) == title


def test_sections_auth():
    assert section_titles(structure(AUTH)) == [
        "Details of pension account holder¹",
        "The authorised representative",
        "Authorisation",
        "Signature(s)",
        "For internal bank use only",
    ]


def test_sections_order():
    assert section_titles(structure(ORDER)) == ["Details of pension account holder", "Order", "Payment instruction", "Signature"]


def test_payout_options_and_attachment():
    fs = structure(PAYOUT)
    titles = section_titles(fs)
    for t in ["Personal data of pension account holder", "Date of payment", "Reason for payout (specifying the documents that must be submitted)",
              "Definitive emigration from Switzerland", "Disability", "Payment instruction (please specify exclusively an account in your name)"]:
        assert t in titles
    reason = next(s for s in fs.parts[0].sections if s.title.text.startswith("Reason for payout"))
    assert len(reason.subsections) == 6  # the six payout reasons, each a checkbox
    assert all(s.title.fields and s.title.fields[0].type == "checkbox" for s in reason.subsections)
    assert [p.kind for p in fs.parts] == ["form", "attachment"]
    assert fs.parts[1].form_number == "63652" and fs.parts[0].form_number == "61073"


def test_field_labels_mapped():
    fs = structure(AUTH)
    labelled = {it.text: [f.name for f in it.fields] for it in all_items(fs) if it.fields}
    assert labelled["Date of birth (dd.mm.yyyy)"] == ["Geburtsdatum_af_date"]
    assert labelled["AHV number / Social security number (756.xxxx.xxxx.xx)"] == ["AHV"]


def test_noise_removed():
    text = build_layout(PAYOUT).text
    assert "Page 1/3" not in text and "61073 E" not in text
    assert text.count("ubs.com/vb") == 1  # running header only once
    assert "Address of previous pension scheme:" in build_layout(ORDER).text  # read-only field text recovered


@pytest.mark.parametrize("pdf", [AUTH, ORDER, PAYOUT])
def test_every_item_is_a_source_slice(pdf):
    fs = structure(pdf)
    for it in all_items(fs):
        s, e = it.provenance.char_span
        assert fs.source_text[s:e] == it.text


# ----------------------------------------------------------- grounding gate

HALLUCINATED = "IBAN of the spouse"
PARAPHRASE = "Details of the account holder"


class MisbehavingModel(BaseLanguageModel):
    """Answers each chunk like an LLM would, plus deliberate errors."""

    def infer(self, batch_prompts, **kwargs):
        for prompt in batch_prompts:
            chunk = prompt.rsplit("\nQ: ", 1)[1].rsplit("\nA:", 1)[0]
            out = []
            for line in (ln.strip() for ln in chunk.splitlines()):
                if not line:
                    continue
                if line in ("Details of pension account holder¹", "The authorised representative", "Signature(s)",
                            "For internal bank use only"):
                    out.append({"section_title": line})
                elif line == "Authorisation":
                    out.append({"legal_text": line})  # demotes a real heading
                elif line == "Phone no.":
                    out.append({"section_title": line})  # tries to promote a label
                elif line == "Last name":
                    out.append({"field_label": line, "field_label_attributes": {"format_hint": "dd.mm.yyyy"}})
                elif line == "Authorisation to give information":
                    out.append({"form_title": line})
                elif len(line) < 60:
                    out.append({"field_label": line})
            out.append({"field_label": HALLUCINATED})
            out.append({"section_title": PARAPHRASE})
            yield [ScoredOutput(score=1.0, output=json.dumps({"extractions": out}))]


@pytest.fixture(scope="module")
def hybrid():
    layout = build_layout(AUTH)
    doc = run_langextract(layout.text, model=MisbehavingModel(), passes=1, max_workers=1)
    return structure(AUTH, list(doc.extractions))


def test_hallucination_rejected(hybrid):
    texts = [norm(it.text) for it in all_items(hybrid)]
    assert HALLUCINATED not in texts and PARAPHRASE not in texts
    rejected = {r.text for r in hybrid.diagnostics.rejected}
    assert HALLUCINATED in rejected and PARAPHRASE in rejected


def test_unsupported_attribute_dropped(hybrid):
    last = next(it for it in all_items(hybrid) if it.text == "Last name")
    assert "format_hint" not in last.attributes
    assert any(d["attribute"] == "format_hint" for d in hybrid.diagnostics.dropped_attributes)


def test_llm_cannot_promote_but_can_demote(hybrid):
    titles = section_titles(hybrid)
    assert "Phone no." not in titles  # promotion refused: no visual heading evidence
    assert "Authorisation" not in titles  # demotion accepted, but reported
    res = {d.text: d.resolution for d in hybrid.diagnostics.layout_disagreements}
    assert res["Phone no."].startswith("kept layout type")
    assert res["Authorisation"].startswith("heading demoted")


def test_hybrid_items_are_source_slices(hybrid):
    for it in all_items(hybrid):
        s, e = it.provenance.char_span
        assert hybrid.source_text[s:e] == it.text
    assert re.search(r"\bstub\b", hybrid.model_id)


def test_no_text_silently_lost(hybrid):
    """Every alphanumeric char of every block ends up in some output item."""
    layout = build_layout(AUTH)
    covered = set()
    for it in all_items(hybrid):
        covered.update(range(*it.provenance.char_span))
    lost = [i for i, ch in enumerate(layout.text) if ch.isalnum() and i not in covered]
    assert lost == []
    assert any(m.get("partial_block") for m in hybrid.diagnostics.missed_by_llm)
