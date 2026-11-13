"""Fillable-field (AcroForm widget) extraction and widget -> label matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pypdf

from .layout import Layout, Segment

FIELD_TYPES = {"/Tx": "text", "/Btn": "checkbox", "/Ch": "choice", "/Sig": "signature"}


@dataclass
class Widget:
    name: str
    type: str
    page: int
    x0: float
    top: float
    x1: float
    bottom: float
    label_segment: int | None = None
    label_position: str | None = None  # below | right | left | above

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (round(self.x0, 1), round(self.top, 1), round(self.x1, 1), round(self.bottom, 1))


def _inherited(annot, key):
    node = annot
    while node is not None:
        if key in node:
            return node[key]
        parent = node.get("/Parent")
        node = parent.get_object() if parent is not None else None
    return None


def read_widgets(pdf_path: str | Path) -> list[Widget]:
    reader = pypdf.PdfReader(str(pdf_path))
    widgets: list[Widget] = []
    for pno, page in enumerate(reader.pages, start=1):
        height = float(page.mediabox.height)
        for ref in page.get("/Annots") or []:
            annot = ref.get_object()
            if annot.get("/Subtype") != "/Widget":
                continue
            name = _inherited(annot, "/T")
            ftype = _inherited(annot, "/FT")
            if name is None or "/Rect" not in annot:
                continue
            x0, y0, x1, y1 = (float(v) for v in annot["/Rect"])
            widgets.append(
                Widget(
                    name=str(name),
                    type=FIELD_TYPES.get(str(ftype), str(ftype)),
                    page=pno,
                    x0=min(x0, x1),
                    top=height - max(y0, y1),  # PDF y-up -> pdfplumber top-down
                    x1=max(x0, x1),
                    bottom=height - min(y0, y1),
                )
            )
    return widgets


def _x_overlap(a: Widget, s: Segment) -> float:
    return max(0.0, min(a.x1, s.x1) - max(a.x0, s.x0))


def _same_row(a: Widget, s: Segment) -> bool:
    return min(a.bottom, s.bottom) - max(a.top, s.top) > 0.3 * (s.bottom - s.top)


def match_labels(widgets: list[Widget], layout: Layout) -> None:
    """Attach each widget to the text segment that labels it (in place).

    Conventions in these forms: a checkbox's label sits to its right on the
    same row; a text box's label sits just below it (UBS style) or, for inline
    boxes, to its left; a few header boxes have their label just above.
    """
    by_page: dict[int, list[Segment]] = {}
    for seg in layout.segments:
        by_page.setdefault(seg.page, []).append(seg)

    for w in widgets:
        segs = by_page.get(w.page, [])
        candidates: list[tuple[float, str, Segment]] = []
        for s in segs:
            if _same_row(w, s):
                if 0 <= s.x0 - w.x1 <= 20:
                    candidates.append((s.x0 - w.x1 + (0 if w.type == "checkbox" else 30), "right", s))
                elif 0 <= w.x0 - s.x1 <= 80 and w.type != "checkbox":
                    candidates.append((w.x0 - s.x1 + 5, "left", s))
            elif _x_overlap(w, s) > 0.3 * min(w.x1 - w.x0, s.x1 - s.x0):
                gap_below = s.top - w.bottom
                gap_above = w.top - s.bottom
                if -1 <= gap_below <= 8:
                    candidates.append((gap_below, "below", s))
                elif -1 <= gap_above <= 12:
                    # weakest convention; any same-row label should win
                    candidates.append((gap_above + 100, "above", s))
        if candidates:
            _, pos, seg = min(candidates, key=lambda c: c[0])
            w.label_segment, w.label_position = seg.id, pos
