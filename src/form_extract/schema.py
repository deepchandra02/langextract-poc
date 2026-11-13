"""Output schema: one FormStructure JSON per PDF."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Agreement = Literal["agree", "disagree", "llm_missing", "layout_only"]


class Provenance(BaseModel):
    """Where the text came from: every item can be traced back to the PDF."""

    char_span: tuple[int, int] = Field(description="[start, end) in source_text")
    page: int
    bbox: tuple[float, float, float, float] = Field(description="x0, top, x1, bottom in PDF points (first line)")
    font_size: float
    bold: bool


class FormField(BaseModel):
    name: str = Field(description="AcroForm field name as stored in the PDF")
    type: str
    page: int
    bbox: tuple[float, float, float, float]
    label_position: str | None = Field(None, description="where the label sits relative to the field")


class Item(BaseModel):
    type: str
    text: str
    layout_role: str
    llm_class: str | None = None
    agreement: Agreement
    attributes: dict[str, str] = {}
    fields: list[FormField] = []
    provenance: Provenance


class Section(BaseModel):
    title: Item
    level: int
    items: list[Item] = []
    subsections: list[Section] = []


class Part(BaseModel):
    kind: Literal["form", "attachment"]
    title: Item | None = None
    form_number: str | None = None
    version_date: str | None = None
    pages: list[int] = []
    preamble: list[Item] = []
    sections: list[Section] = []


class Rejected(BaseModel):
    llm_class: str
    text: str
    reason: str
    alignment_status: str | None = None
    char_span: tuple[int, int] | None = None


class Disagreement(BaseModel):
    text: str
    page: int
    layout_role: str
    llm_class: str
    resolution: str


class Diagnostics(BaseModel):
    llm_extractions_total: int = 0
    llm_extractions_accepted: int = 0
    rejected: list[Rejected] = []
    dropped_attributes: list[dict] = []
    layout_disagreements: list[Disagreement] = []
    missed_by_llm: list[dict] = []
    unlabelled_fields: list[str] = []
    dropped_noise: list[dict] = []


class FormStructure(BaseModel):
    source_file: str
    mode: Literal["hybrid", "layout-only"]
    model_id: str | None = None
    page_count: int
    title: str | None = Field(None, description="convenience copy of parts[0].title.text")
    issuer: list[Item] = []
    recipient_address: list[Item] = []
    parts: list[Part] = []
    footnotes: list[Item] = []
    diagnostics: Diagnostics = Diagnostics()
    source_text: str = Field(description="the exact text all char_spans index into")
