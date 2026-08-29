from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class OCRSpan(StrictModel):
    id: str = Field(pattern=r"^ocr_\d{4,}$")
    text: str = Field(min_length=1)
    polygon: list[float] = Field(min_length=8, max_length=8)
    confidence: float | None = Field(default=None, ge=0, le=1)


class OCRDocument(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    provider: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    spans: list[OCRSpan]

    @field_validator("spans")
    @classmethod
    def spans_are_unique(cls, value: list[OCRSpan]) -> list[OCRSpan]:
        if len({span.id for span in value}) != len(value):
            raise ValueError("OCR span IDs must be unique")
        return value


class SourceText(StrictModel):
    text: str = Field(min_length=1)
    source_ids: list[str]


class Price(StrictModel):
    raw: str = Field(min_length=1)
    amount: str | int | float | None = None
    currency: str | None = None
    label: str | None = None
    source_ids: list[str]


class Calories(StrictModel):
    raw: str = Field(min_length=1)
    value: int | None = Field(default=None, ge=0)
    source_ids: list[str]


class MenuVariant(StrictModel):
    name: SourceText | None = None
    prices: list[Price] = Field(default_factory=list)


class MenuItem(StrictModel):
    name: SourceText
    description: SourceText | None = None
    prices: list[Price] = Field(default_factory=list)
    calories: Calories | None = None
    variants: list[MenuVariant] = Field(default_factory=list)
    notes: list[SourceText] = Field(default_factory=list)


class MenuSection(StrictModel):
    name: SourceText
    items: list[MenuItem] = Field(default_factory=list)
    notes: list[SourceText] = Field(default_factory=list)


class MenuDocument(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    currency: str | None = None
    sections: list[MenuSection] = Field(default_factory=list)
    unsectioned_items: list[MenuItem] = Field(default_factory=list)


class ImageRef(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_name: str


class Metadata(StrictModel):
    restaurant_id: str
    language: list[str] = Field(default_factory=list)
    source: str = "local"
    synthetic: bool = False
    tags: list[str] = Field(default_factory=list)


class ReleaseRecord(StrictModel):
    document_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    kind: Literal["source_reference", "silver_augmented"]
    split: Literal["train", "validation", "test"] | None = None
    image: ImageRef
    ocr: OCRDocument
    metadata: Metadata
    annotation: MenuDocument
    provenance: dict[str, Any]


class PrintedText(StrictModel):
    text: str = Field(alias="t")
    line_indices: list[int] = Field(alias="l")


class CompactSection(StrictModel):
    id: str
    heading: PrintedText = Field(alias="h")
    notes: list[PrintedText]


class CompactItem(StrictModel):
    line_indices: list[int] = Field(alias="l", min_length=1)
    name: str = Field(alias="n", min_length=1)
    anchor: int = Field(alias="a", ge=1)
    confidence: float = Field(alias="c", ge=0, le=1)
    section_ref: str | None = Field(alias="s")
    notes: list[PrintedText]

    @model_validator(mode="after")
    def anchor_is_an_item_line(self) -> "CompactItem":
        if self.anchor not in self.line_indices:
            raise ValueError("Item anchor must be included in item OCR lines")
        return self


class CompactOutput(StrictModel):
    sections: list[CompactSection] = Field(alias="s")
    items: list[CompactItem] = Field(alias="i")

    @model_validator(mode="after")
    def section_references_exist(self) -> "CompactOutput":
        ids = [section.id for section in self.sections]
        if len(ids) != len(set(ids)):
            raise ValueError("Section IDs must be unique")
        if any(item.section_ref is not None and item.section_ref not in ids for item in self.items):
            raise ValueError("Item references an unknown section")
        return self


def validate_compact_output(value: Any, *, ocr_line_count: int) -> CompactOutput:
    output = CompactOutput.model_validate(value)
    references = (
        [
            line
            for section in output.sections
            for text in [section.heading, *section.notes]
            for line in text.line_indices
        ]
        + [line for item in output.items for line in item.line_indices]
        + [line for item in output.items for note in item.notes for line in note.line_indices]
    )
    if any(line < 1 or line > ocr_line_count for line in references):
        raise ValueError("Compact output contains an out-of-range OCR reference")
    return output
