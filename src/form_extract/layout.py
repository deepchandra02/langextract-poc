"""PDF layout pass: turns a PDF into ordered text segments with font metadata.

Everything downstream (LangExtract, reconciliation, the output tree) works on
`Layout.text`, which is built here deterministically from the PDF. Each
segment records its [start, end) character offsets in that text, so any
character span can be traced back to a page, a bounding box and a font.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
import pypdf

# Two words further apart than this (relative to font size) belong to
# different columns / cells, e.g. "Last name" ... "First name(s)".
COLUMN_GAP_FACTOR = 2.5
LINE_TOP_TOLERANCE = 2.5
MIN_FONT_SIZE = 4.0  # below this: barcodes / scanner marks
SUPERSCRIPT_RATIO = 0.75  # chars smaller than this * line size are footnote markers

FOOTER_RE = re.compile(r"^(?P<form_no>\d{5})\s+E\b.*Page\s*\d+/\d+$")
DATE_RE = re.compile(r"\d{2}\.\d{2}\.\d{4}")
ROW_TOLERANCE = 3.0
DIGITS_ONLY_RE = re.compile(r"^[\d\s]+$")


@dataclass
class Segment:
    """One run of text on one visual line (a line may hold several segments)."""

    id: int
    page: int  # 1-based
    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    size: float
    bold_ratio: float
    first_word_bold: bool
    has_footnote_marker: bool = False
    leading_marker: bool = False  # line starts with a small footnote number -> footnote text
    start: int = -1  # offsets into Layout.text
    end: int = -1

    @property
    def bold(self) -> bool:
        return self.bold_ratio >= 0.6

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (round(self.x0, 1), round(self.top, 1), round(self.x1, 1), round(self.bottom, 1))


@dataclass
class Footer:
    page: int
    text: str
    form_no: str
    date: str


@dataclass
class Layout:
    source_file: str
    page_count: int
    page_widths: dict[int, float]
    segments: list[Segment]
    text: str
    footers: list[Footer] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)  # noise we removed, for transparency

    def segments_in_span(self, start: int, end: int) -> list[Segment]:
        return [s for s in self.segments if s.start < end and start < s.end]


def _is_bold(fontname: str) -> bool:
    return "bold" in fontname.lower()


def _line_size(chars: list[dict]) -> float:
    sizes = sorted(round(c["size"], 1) for c in chars if c["text"].strip())
    return sizes[len(sizes) // 2] if sizes else 0.0


def _words_to_lines(words: list[dict]) -> list[list[dict]]:
    words = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
    lines: list[list[dict]] = []
    for w in words:
        # Superscripts sit higher than the baseline; match on bottom as well.
        for line in lines:
            ref = line[0]
            if abs(w["top"] - ref["top"]) <= LINE_TOP_TOLERANCE or abs(w["bottom"] - ref["bottom"]) <= LINE_TOP_TOLERANCE:
                line.append(w)
                break
        else:
            lines.append([w])
    for line in lines:
        line.sort(key=lambda w: w["x0"])
    lines.sort(key=lambda ln: min(w["top"] for w in ln))
    return lines


def _split_columns(line: list[dict]) -> list[list[dict]]:
    size = _line_size([c for w in line for c in w["chars"]]) or 8.0
    groups = [[line[0]]]
    for prev, w in zip(line, line[1:]):
        if w["x0"] - prev["x1"] > COLUMN_GAP_FACTOR * size:
            groups.append([w])
        else:
            groups[-1].append(w)
    return groups


def _render(words: list[dict]) -> tuple[str, float, float, bool, bool, bool]:
    chars = [c for w in words for c in w["chars"]]
    size = _line_size(chars)
    parts, marker = [], False
    leading = bool(chars) and size and chars[0]["size"] < SUPERSCRIPT_RATIO * size and chars[0]["text"] in "¹²³1234567890"
    for w in words:
        kept = []
        for c in w["chars"]:
            if size and c["size"] < SUPERSCRIPT_RATIO * size and c["text"] in "¹²³1234567890":
                marker = True  # footnote reference like "holder¹" -> keep text clean
                continue
            kept.append(c["text"])
        if kept:
            parts.append("".join(kept))
    text = " ".join(parts).strip()
    visible = [c for c in chars if c["text"].strip()]
    bold_ratio = sum(_is_bold(c["fontname"]) for c in visible) / len(visible) if visible else 0.0
    first_word_bold = bool(words[0]["chars"]) and all(_is_bold(c["fontname"]) for c in words[0]["chars"])
    return text, size, bold_ratio, first_word_bold, marker and not leading, bool(leading)


def _readonly_widget_text(pdf_path: Path) -> list[dict]:
    """Visible text stored as read-only form-field values (e.g. a printed
    "Address of previous pension scheme:" caption). pdfplumber does not render
    widget appearance streams, so without this such text would be lost."""
    out = []
    reader = pypdf.PdfReader(str(pdf_path))
    for pno, page in enumerate(reader.pages, start=1):
        height = float(page.mediabox.height)
        for ref in page.get("/Annots") or []:
            a = ref.get_object()
            value, flags = a.get("/V"), int(a.get("/Ff", 0))
            if a.get("/Subtype") != "/Widget" or not isinstance(value, str) or not value.strip() or not flags & 1:
                continue
            x0, y0, x1, y1 = (float(v) for v in a["/Rect"])
            m = re.search(r"([\d.]+)\s+Tf", str(a.get("/DA", "")))
            out.append({"page": pno, "text": value.strip(), "x0": min(x0, x1), "x1": max(x0, x1),
                        "top": height - max(y0, y1), "bottom": height - min(y0, y1),
                        "size": float(m.group(1)) if m else 8.0, "field": str(a.get("/T"))})
    return out


def build_layout(pdf_path: str | Path) -> Layout:
    pdf_path = Path(pdf_path)
    segments: list[Segment] = []
    footers: list[Footer] = []
    dropped: list[dict] = []
    page_widths: dict[int, float] = {}
    page1_header: set[str] = set()

    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            page_widths[pno] = float(page.width)
            n_tiny = sum(1 for c in page.chars if c["size"] < MIN_FONT_SIZE)
            if n_tiny:
                dropped.append({"page": pno, "text": f"{n_tiny} chars", "top": None, "reason": "tiny font (barcode)"})
            page = page.filter(lambda o: not (o.get("object_type") == "char" and o.get("size", 99) < MIN_FONT_SIZE))
            words = page.extract_words(extra_attrs=["fontname", "size"], return_chars=True, keep_blank_chars=False)
            for line in _words_to_lines(words):
                line_text = " ".join(w["text"] for w in line)
                m = FOOTER_RE.match(line_text)
                if m:
                    d = DATE_RE.search(line_text.replace(" ", ""))
                    footers.append(Footer(pno, line_text, m["form_no"], d.group(0) if d else ""))
                    dropped.append({"page": pno, "text": line_text, "top": round(line[0]["top"], 1), "reason": "footer"})
                    continue
                for group in _split_columns(line):
                    text, size, bold_ratio, fwb, marker, leading = _render(group)
                    x0 = min(w["x0"] for w in group)
                    top = min(w["top"] for w in group)
                    info = {"page": pno, "text": text, "top": round(top, 1)}
                    if not text:
                        continue
                    if text in ("–", "-", "•"):
                        dropped.append({**info, "reason": "orphan bullet glyph"})
                        continue
                    if DIGITS_ONLY_RE.match(text):
                        dropped.append({**info, "reason": "digits-only placeholder"})
                        continue
                    # Running header repeated from page 1's letterhead (top band, right half).
                    right_half = x0 > page.width / 2
                    if pno == 1 and right_half and top < 150:
                        page1_header.add(text)
                    elif pno > 1 and right_half and top < 100 and text in page1_header:
                        dropped.append({**info, "reason": "running header"})
                        continue
                    segments.append(
                        Segment(
                            id=len(segments),
                            page=pno,
                            text=text,
                            x0=x0,
                            top=top,
                            x1=max(w["x1"] for w in group),
                            bottom=max(w["bottom"] for w in group),
                            size=size,
                            bold_ratio=round(bold_ratio, 2),
                            first_word_bold=fwb,
                            has_footnote_marker=marker,
                            leading_marker=leading,
                        )
                    )
        page_count = len(pdf.pages)

    for w in _readonly_widget_text(pdf_path):
        segments.append(Segment(id=-1, page=w["page"], text=w["text"], x0=w["x0"], top=w["top"], x1=w["x1"],
                                bottom=w["bottom"], size=w["size"], bold_ratio=0.0, first_word_bold=False))

    # Reading order: page, then top, then x. (Two-column cells on one row stay left->right.)
    segments.sort(key=lambda s: (s.page, s.top))
    row, row_top, prev_pg = -1, None, None
    rows: dict[int, int] = {}
    for seg in segments:
        if seg.page != prev_pg or seg.top - row_top > ROW_TOLERANCE:
            row, row_top, prev_pg = row + 1, seg.top, seg.page
        rows[id(seg)] = row
    segments.sort(key=lambda s: (rows[id(s)], s.x0))
    pieces: list[str] = []
    pos = 0
    prev_page = None
    for i, seg in enumerate(segments):
        seg.id = i
        sep = "" if prev_page is None else ("\n\n" if seg.page != prev_page else "\n")
        pos += len(sep)
        pieces.append(sep + seg.text)
        seg.start, seg.end = pos, pos + len(seg.text)
        pos = seg.end
        prev_page = seg.page

    return Layout(
        source_file=pdf_path.name,
        page_count=page_count,
        page_widths=page_widths,
        segments=segments,
        text="".join(pieces),
        footers=footers,
        dropped=dropped,
    )
