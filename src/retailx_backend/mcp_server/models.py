"""Pydantic input/output schemas for the MCP tools (these become the tool JSON schemas)."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..domain import MAX_AMOUNT_PAISE, parse_date, validate_range

GroupBy = Literal["none", "day", "week", "month"]


class Selector(BaseModel):
    """Targets sales records. Provide exactly one of: record_ids, date, or start_date + end_date."""

    record_ids: list[str] | None = Field(default=None, min_length=1, max_length=100)
    date: str | None = Field(default=None, description="Single day, YYYY-MM-DD")
    start_date: str | None = Field(default=None, description="Range start, YYYY-MM-DD")
    end_date: str | None = Field(default=None, description="Range end (inclusive), YYYY-MM-DD")

    @model_validator(mode="after")
    def _exactly_one_mode(self) -> "Selector":
        has_range = self.start_date is not None or self.end_date is not None
        if sum([self.record_ids is not None, self.date is not None, has_range]) != 1:
            raise ValueError("Provide exactly one of: record_ids, date, or start_date+end_date")
        if self.date is not None:
            parse_date(self.date)
        if has_range:
            if self.start_date is None or self.end_date is None:
                raise ValueError("start_date and end_date are both required for a range")
            validate_range(self.start_date, self.end_date)
        return self

    @property
    def mode(self) -> Literal["ids", "date", "range"]:
        if self.record_ids is not None:
            return "ids"
        return "date" if self.date is not None else "range"


class SaleChanges(BaseModel):
    """Fields to change. ``new_date`` is only allowed when selecting by record_ids or one date."""

    amount_paise: int | None = Field(default=None, gt=0, le=MAX_AMOUNT_PAISE)
    note: str | None = Field(default=None, max_length=200)
    new_date: str | None = Field(default=None, description="YYYY-MM-DD")

    @model_validator(mode="after")
    def _at_least_one(self) -> "SaleChanges":
        if self.amount_paise is None and self.note is None and self.new_date is None:
            raise ValueError("At least one of amount_paise, note, new_date is required")
        if self.new_date is not None:
            parse_date(self.new_date)
        return self


class SaleRecord(BaseModel):
    id: str
    date: str
    amount: float = Field(description="Rupees")
    amount_paise: int
    note: str = ""
    created_at: str


class SalesList(BaseModel):
    records: list[SaleRecord]
    returned: int
    total_count: int
    total_paise: int
    total: float
    truncated: bool


class AggregateRow(BaseModel):
    label: str
    total_paise: int
    total: float
    count: int


class AggregateResult(BaseModel):
    start_date: str
    end_date: str
    group_by: GroupBy
    rows: list[AggregateRow]
    total_paise: int
    total: float
    count: int
    average_paise: int


class TrendPoint(BaseModel):
    label: str
    total_paise: int
    total: float
    count: int
    change_pct: float | None = Field(default=None, description="vs previous period")


class TrendResult(BaseModel):
    start_date: str
    end_date: str
    granularity: Literal["day", "week", "month"]
    points: list[TrendPoint]
    total_paise: int
    total: float
    overall_change_pct: float | None = Field(default=None, description="first vs last period")
    peak_label: str | None = None
    lowest_label: str | None = None


class SelectionPreview(BaseModel):
    count: int
    total_paise: int
    total: float
    min_date: str | None = None
    max_date: str | None = None
    span_days: int = 0
    sample: list[SaleRecord]


class WriteResult(BaseModel):
    op_key: str
    tool: str
    replayed: bool = False
    affected: int
    total_paise: int
    total: float
    record: SaleRecord | None = None
