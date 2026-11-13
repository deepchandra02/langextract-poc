"""Grounding gate + layout cross-check + tree building.

Anti-hallucination contract
---------------------------
1. Every LLM extraction must have a char span, and the source slice at that
   span must equal the extraction text (whitespace-normalised). LangExtract's
   alignment status is recorded but not trusted on its own: MATCH_FUZZY can be
   a perfectly real span emitted out of order, and conversely a fuzzy match can
   point at text that differs from what the model wrote.
2. Attribute values must be verbatim substrings of their extraction's text.
3. Output text is always the *source slice*, never the model's string.
4. Headings shape the tree only with visual evidence: the LLM can demote a
   bold line it judges to be an instruction, but it cannot promote a plain
   line to a heading. Every disagreement is reported in diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import hints as H
from .acroform import Widget
from .layout import Layout
from .schema import (
    Diagnostics,
    Disagreement,
    FormField,
    FormStructure,
    Item,
    Part,
    Provenance,
    Rejected,
    Section,
)

LAYOUT_DEFAULT_TYPE = {
    H.TITLE: "form_title",
    H.ATTACHMENT_TITLE: "form_title",
    H.SECTION: "section_title",
    H.OPTION: "subsection_title",
    H.CHOICE: "checkbox_option",
    H.LABEL: "field_label",
    H.PREFILLED: "prefilled_value",
    H.LIST_ITEM: "list_item",
    H.PARAGRAPH: "paragraph",
    H.EMPHASIS: "instruction",
    H.FOOTNOTE: "footnote",
    H.CONNECTOR: "connector",
    H.ISSUER: "issuer",
    H.ADDRESS: "address",
}

# Which layout roles are visually consistent with each LLM class.
COMPATIBLE: dict[str, set[str]] = {
    "form_title": {H.TITLE, H.ATTACHMENT_TITLE},
    "section_title": {H.SECTION},
    "subsection_title": {H.OPTION, H.SECTION},
    "field_label": {H.LABEL, H.PARAGRAPH, H.LIST_ITEM},
    "checkbox_option": {H.CHOICE, H.OPTION},
    "prefilled_value": {H.PREFILLED},
    "instruction": {H.PARAGRAPH, H.EMPHASIS, H.LIST_ITEM, H.CONNECTOR},
    "required_document": {H.LIST_ITEM, H.PARAGRAPH},
    "legal_text": {H.PARAGRAPH, H.EMPHASIS, H.LIST_ITEM},
    "footnote": {H.FOOTNOTE},
}
HEADING_CLASSES = {"form_title", "section_title", "subsection_title", "checkbox_option"}
HEADING_ROLES = {H.TITLE, H.ATTACHMENT_TITLE, H.SECTION, H.OPTION}
STRUCTURAL_ROLES = HEADING_ROLES | {H.LABEL, H.CHOICE, H.PREFILLED}


def norm(s: str) -> str:
    return " ".join(s.split())


@dataclass
class Grounded:
    cls: str
    start: int
    end: int
    attributes: dict[str, str]
    alignment: str | None


def ground(extractions: list[Any], source: str, diag: Diagnostics) -> list[Grounded]:
    """Apply rules 1 and 2. Returns accepted extractions sorted by position."""
    accepted: dict[tuple[str, int, int], Grounded] = {}
    for ex in extractions:
        diag.llm_extractions_total += 1
        status = getattr(ex.alignment_status, "value", None)
        ci = ex.char_interval
        if ci is None or ci.start_pos is None or ci.end_pos is None:
            diag.rejected.append(Rejected(llm_class=ex.extraction_class, text=ex.extraction_text,
                                          reason="not found in source text", alignment_status=status))
            continue
        span = (ci.start_pos, ci.end_pos)
        if norm(source[span[0]:span[1]]) != norm(ex.extraction_text):
            diag.rejected.append(Rejected(llm_class=ex.extraction_class, text=ex.extraction_text,
                                          reason=f"source at span reads {source[span[0]:span[1]]!r}",
                                          alignment_status=status, char_span=span))
            continue
        attrs: dict[str, str] = {}
        for k, v in (ex.attributes or {}).items():
            if isinstance(v, str) and v and norm(v) in norm(ex.extraction_text):
                attrs[k] = v
            elif v:
                diag.dropped_attributes.append({"text": ex.extraction_text, "attribute": k, "value": v,
                                                "reason": "value is not a substring of the extraction text"})
        key = (ex.extraction_class, *span)
        if key not in accepted:  # multiple passes can repeat an extraction
            accepted[key] = Grounded(ex.extraction_class, span[0], span[1], attrs, status)
    diag.llm_extractions_accepted = len(accepted)
    return sorted(accepted.values(), key=lambda g: (g.start, -g.end))


@dataclass
class _BlockResult:
    block: H.Block
    items: list[Item] = field(default_factory=list)
    is_heading: bool = False


def _provenance(layout: Layout, start: int, end: int) -> Provenance:
    segs = layout.segments_in_span(start, end)
    s = segs[0]
    return Provenance(char_span=(start, end), page=s.page, bbox=s.bbox, font_size=s.size, bold=s.bold)


def _layout_item(layout: Layout, block: H.Block, agreement: str) -> Item:
    return Item(
        type=LAYOUT_DEFAULT_TYPE[block.role],
        text=block.text(layout.text),
        layout_role=block.role,
        agreement=agreement,
        provenance=_provenance(layout, block.start, block.end),
    )


def _classify_blocks(layout: Layout, blocks: list[H.Block], grounded: list[Grounded] | None,
                     diag: Diagnostics) -> list[_BlockResult]:
    results = [_BlockResult(b) for b in blocks]
    if grounded is None:  # layout-only mode
        for r in results:
            r.items = [_layout_item(layout, r.block, "layout_only")]
            r.is_heading = r.block.role in HEADING_ROLES
        return results

    owner: dict[int, list[Grounded]] = {}
    covered: set[int] = set()
    for g in grounded:
        hit = [b for b in blocks if b.start < g.end and g.start < b.end]
        if not hit:  # e.g. span lies only on a page separator
            diag.rejected.append(Rejected(llm_class=g.cls, text=layout.text[g.start:g.end],
                                          reason="span does not overlap any layout block", char_span=(g.start, g.end)))
            continue
        owner.setdefault(hit[0].id, []).append(g)
        covered.update(b.id for b in hit[1:])

    for r in results:
        b = r.block
        gs = owner.get(b.id, [])
        if b.role in (H.ISSUER, H.ADDRESS):  # letterhead/address: layout owns these regions
            gs = []
        if not gs:
            if b.id in covered:
                continue
            r.items = [_layout_item(layout, b, "llm_missing")]
            r.is_heading = b.role in HEADING_ROLES
            if b.role in STRUCTURAL_ROLES:
                diag.missed_by_llm.append({"text": b.text(layout.text), "page": b.page, "layout_role": b.role})
            continue
        for g in gs:
            text = layout.text[g.start:g.end]
            ok = g.cls in COMPATIBLE and b.role in COMPATIBLE[g.cls]
            final_type, resolution = g.cls, None
            if not ok:
                if g.cls in HEADING_CLASSES and b.role not in HEADING_ROLES:
                    final_type = LAYOUT_DEFAULT_TYPE[b.role]
                    resolution = f"kept layout type '{final_type}': no visual heading evidence"
                elif b.role in HEADING_ROLES:
                    resolution = f"heading demoted: LLM classified it as '{g.cls}'"
                else:
                    resolution = f"kept LLM class '{g.cls}' (content-level difference)"
                diag.layout_disagreements.append(Disagreement(text=text, page=b.page, layout_role=b.role,
                                                              llm_class=g.cls, resolution=resolution))
            if ok and g.cls in HEADING_CLASSES and b.role in HEADING_ROLES:
                r.is_heading = True
            r.items.append(Item(type=final_type, text=text, layout_role=b.role, llm_class=g.cls,
                                agreement="agree" if ok else "disagree", attributes=g.attributes,
                                provenance=_provenance(layout, g.start, g.end)))
        # No silent loss: block text the LLM left uncovered is kept as layout-only items.
        for s, e in _uncovered(layout.text, b.start, b.end, [(g.start, g.end) for g in gs]):
            r.items.append(Item(type=LAYOUT_DEFAULT_TYPE[b.role], text=layout.text[s:e], layout_role=b.role,
                                agreement="llm_missing", provenance=_provenance(layout, s, e)))
            diag.missed_by_llm.append({"text": layout.text[s:e], "page": b.page, "layout_role": b.role,
                                       "partial_block": True})
        r.items.sort(key=lambda it: it.provenance.char_span)
    return results


def _uncovered(text: str, start: int, end: int, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sub-ranges of [start, end) not covered by `spans`, trimmed, ignoring whitespace/punctuation-only gaps."""
    out, pos = [], start
    for s, e in sorted(spans) + [(end, end)]:
        s, e = max(s, start), min(e, end)
        if s > pos:
            piece = text[pos:s]
            if any(ch.isalnum() for ch in piece):
                lead = len(piece) - len(piece.lstrip())
                out.append((pos + lead, pos + len(piece.rstrip())))
        pos = max(pos, e)
    return out


def _attach_fields(results: list[_BlockResult], widgets: list[Widget], layout: Layout, diag: Diagnostics) -> None:
    items = [it for r in results for it in r.items]
    for w in widgets:
        if w.label_segment is None:
            diag.unlabelled_fields.append(f"{w.name} (page {w.page}, {w.type})")
            continue
        seg = layout.segments[w.label_segment]
        hits = [it for it in items if it.provenance.char_span[0] < seg.end and seg.start < it.provenance.char_span[1]]
        if not hits:
            diag.unlabelled_fields.append(f"{w.name} (page {w.page}, {w.type}) - label text not kept")
            continue
        best = min(hits, key=lambda it: it.provenance.char_span[1] - it.provenance.char_span[0])
        best.fields.append(FormField(name=w.name, type=w.type, page=w.page, bbox=w.bbox, label_position=w.label_position))


def _build_tree(results: list[_BlockResult], layout: Layout, fs: FormStructure) -> None:
    part = Part(kind="form")
    fs.parts.append(part)
    sec: Section | None = None
    sub: Section | None = None

    for r in results:
        if not r.items:
            continue
        role = r.block.role
        if role == H.ISSUER:
            fs.issuer.extend(r.items)
            continue
        if role == H.ADDRESS:
            fs.recipient_address.extend(r.items)
            continue
        if all(it.type == "footnote" for it in r.items):
            fs.footnotes.extend(r.items)
            continue
        if r.is_heading:
            head, rest = r.items[0], r.items[1:]
            if role == H.TITLE and part.title is None:
                part.title = head
                part.preamble.extend(rest)
                continue
            if role == H.ATTACHMENT_TITLE or (role == H.TITLE and part.title is not None):
                part = Part(kind="attachment", title=head, preamble=list(rest))
                fs.parts.append(part)
                sec = sub = None
                continue
            if role == H.SECTION or sec is None:
                sec = Section(title=head, level=1, items=list(rest))
                part.sections.append(sec)
                sub = None
            else:
                sub = Section(title=head, level=2, items=list(rest))
                sec.subsections.append(sub)
            continue
        target = sub.items if sub else sec.items if sec else part.preamble
        target.extend(r.items)

    footers = {f.page: f for f in layout.footers}

    def pages_of(p: Part) -> set[int]:
        items = [p.title] if p.title else []
        items += p.preamble
        stack = list(p.sections)
        while stack:
            s = stack.pop()
            items += [s.title, *s.items]
            stack += s.subsections
        return {it.provenance.page for it in items}

    for p in fs.parts:
        p.pages = sorted(pages_of(p))
        f = next((footers[pg] for pg in p.pages if pg in footers), None)
        if f:
            p.form_number, p.version_date = f.form_no, f.date
    fs.title = fs.parts[0].title.text if fs.parts and fs.parts[0].title else None


def reconcile(layout: Layout, widgets: list[Widget], blocks: list[H.Block], extractions: list[Any] | None,
              model_id: str | None = None) -> FormStructure:
    diag = Diagnostics(dropped_noise=layout.dropped)
    grounded = ground(extractions, layout.text, diag) if extractions is not None else None
    results = _classify_blocks(layout, blocks, grounded, diag)
    _attach_fields(results, widgets, layout, diag)
    fs = FormStructure(
        source_file=layout.source_file,
        mode="hybrid" if extractions is not None else "layout-only",
        model_id=model_id if extractions is not None else None,
        page_count=layout.page_count,
        diagnostics=diag,
        source_text=layout.text,
    )
    _build_tree(results, layout, fs)
    return fs
