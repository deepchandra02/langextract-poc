"""Deterministic structure hints from layout alone (no LLM).

Groups segments into blocks (a heading, a label, a wrapped paragraph...) and
assigns each block a layout role. These roles are (a) the output in
`--no-llm` mode and (b) the cross-check that LLM classifications must agree
with before they are allowed to shape the document tree.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .acroform import Widget
from .layout import Layout, Segment

TITLE_MIN_SIZE = 14.0
SECTION_MAX_CHARS = 90
FOOTNOTE_MAX_SIZE = 7.0
CHOICE_WORDS = ("Yes", "No")
DASHES = ("–", "-", "•")

# Layout roles
TITLE = "title"
ATTACHMENT_TITLE = "attachment_title"
ISSUER = "issuer"
ADDRESS = "address_block"
SECTION = "section_heading"
OPTION = "option_heading"  # indented bold line, usually a checkbox choice with sub-content
CONNECTOR = "connector"  # bold "or"
CHOICE = "choice"  # Yes / No style checkbox option
LABEL = "field_label"
LIST_ITEM = "list_item"
EMPHASIS = "emphasised_text"  # bold sentence, e.g. "Send original to ..."
PARAGRAPH = "paragraph"
PREFILLED = "prefilled_value"  # printed value sitting above its label (e.g. a fixed IBAN)
FOOTNOTE = "footnote"


@dataclass
class Block:
    id: int
    role: str
    segments: list[Segment]
    widgets: list[Widget] = field(default_factory=list)

    @property
    def page(self) -> int:
        return self.segments[0].page

    @property
    def start(self) -> int:
        return self.segments[0].start

    @property
    def end(self) -> int:
        return self.segments[-1].end

    @property
    def x0(self) -> float:
        return self.segments[0].x0

    def text(self, source: str) -> str:
        return source[self.start : self.end]


def _margin(layout: Layout) -> float:
    """Left text margin = most common x0 of body-size text in the document
    (per-page modes fail on pages that are mostly indented sub-options)."""
    xs = Counter(round(s.x0) for s in layout.segments if 7 <= s.size < TITLE_MIN_SIZE)
    if not xs:
        return 50.0
    top = max(xs.values())
    return float(min(x for x, n in xs.items() if n == top))


def _prefilled_pairs(layout: Layout) -> dict[int, int]:
    """value segment id -> label segment id, for larger printed values sitting
    directly above a small caption at the same x (UBS 'value over label' style)."""
    pairs = {}
    segs = layout.segments
    for s in segs:
        for t in segs:
            if (t.page == s.page and t.size < s.size - 1 and abs(t.x0 - s.x0) < 3
                    and 0 <= t.top - s.bottom < 8 and len(t.text) < 40):
                pairs[s.id] = t.id
    return pairs


def _is_heading_like(seg: Segment) -> bool:
    if seg.text.rstrip().endswith("."):
        return False
    return seg.bold or (seg.first_word_bold and seg.bold_ratio >= 0.2)


def _segment_role(seg: Segment, layout: Layout, title_seen: bool, margin: float, labelled: set[int],
                  prefilled: dict[int, int]) -> str:
    text = seg.text.strip()
    if seg.size >= TITLE_MIN_SIZE:
        return ATTACHMENT_TITLE if title_seen else TITLE
    if title_seen and seg.id in prefilled:
        return PREFILLED
    if not title_seen and seg.page == 1:
        return ISSUER if seg.x0 > layout.page_widths[1] / 2 else ADDRESS
    if seg.size <= FOOTNOTE_MAX_SIZE or seg.leading_marker or text.startswith(("¹", "²", "³")):
        return FOOTNOTE
    if seg.bold and text.lower() == "or":
        return CONNECTOR
    if text in CHOICE_WORDS or text.startswith(tuple(w + " (" for w in CHOICE_WORDS)):
        return CHOICE
    if _is_heading_like(seg):
        at_margin = seg.x0 - margin < 5
        if at_margin and len(text) <= SECTION_MAX_CHARS:
            return SECTION
        if not at_margin:
            return OPTION
    if seg.bold:
        return EMPHASIS
    if seg.id in labelled:
        return LABEL
    if text.startswith(DASHES):
        return LIST_ITEM
    return PARAGRAPH


def _continues(prev: Segment, seg: Segment, prev_role: str, role: str, page_width: float) -> bool:
    """Does `seg` continue the paragraph/list item that `prev` belongs to?"""
    if role in (TITLE, ATTACHMENT_TITLE) and prev_role == role and seg.page == prev.page:
        return seg.top - prev.bottom < 0.8 * seg.size  # multi-line title
    if seg.page != prev.page or role not in (PARAGRAPH, LIST_ITEM, FOOTNOTE, EMPHASIS):
        return False
    if prev_role not in (PARAGRAPH, LIST_ITEM, FOOTNOTE, EMPHASIS):
        return False
    if role == LIST_ITEM:  # a new dash starts a new item
        return False
    if seg.text.strip().startswith(("Condition:",)):
        return False
    if prev.x1 < 0.75 * page_width:  # previous line did not run to the margin -> no wrap
        return False
    line_gap = seg.top - prev.bottom
    if line_gap > 0.6 * max(seg.size, prev.size):  # blank line between them
        return False
    if prev.text.rstrip().endswith((".", ":")) and prev_role != LIST_ITEM:
        return False
    return abs(seg.x0 - prev.x0) < 10 or (prev_role == LIST_ITEM and seg.x0 > prev.x0)


def build_blocks(layout: Layout, widgets: list[Widget]) -> list[Block]:
    prefilled = _prefilled_pairs(layout)
    labelled = {w.label_segment for w in widgets if w.label_segment is not None} | set(prefilled.values())
    by_label: dict[int, list[Widget]] = {}
    for w in widgets:
        if w.label_segment is not None:
            by_label.setdefault(w.label_segment, []).append(w)

    blocks: list[Block] = []
    title_seen = False
    margin = _margin(layout)
    prev_seg: Segment | None = None
    prev_role = None
    for seg in layout.segments:
        role = _segment_role(seg, layout, title_seen, margin, labelled, prefilled)
        if role == ATTACHMENT_TITLE and prev_role == TITLE and prev_seg is not None and prev_seg.page == seg.page:
            role = TITLE  # second line of a wrapped title
        if role == TITLE:
            title_seen = True
        if blocks and prev_seg is not None and _continues(prev_seg, seg, prev_role, role, layout.page_widths[seg.page]):
            blocks[-1].segments.append(seg)
        else:
            blocks.append(Block(id=len(blocks), role=role, segments=[seg]))
        blocks[-1].widgets.extend(by_label.get(seg.id, []))
        prev_seg, prev_role = seg, blocks[-1].role
    return blocks
