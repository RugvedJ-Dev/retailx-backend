"""LLM gateway: provider-agnostic, tiered (fast planner / strong synthesis)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import date
from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ..config import Settings
from .plan import Plan

Tier = Literal["fast", "strong"]

PLANNER_SYSTEM = """You are the planning component of RetailX AI, an assistant that helps shop \
owners analyse and maintain their sales records. Today is {weekday}, {today} (timezone {tz}).

Convert the user's latest request into an execution plan of 1-{max_steps} sequential steps. When no \
data operation is needed (greetings, questions about what you can do, out-of-scope requests, or \
when you must ask a clarifying question) return no steps and put your reply in direct_reply. \
Never invent data. Output only the structured plan.

Step actions (use ONLY these):
- get_sales: records for one day. Fields: date
- get_sales_range: records between two dates. Fields: start_date, end_date
- aggregate_sales: total/count/average for a range. Fields: start_date, end_date, group_by \
(none|day|week|month)
- sales_trend: time series for trends and charts. Fields: start_date, end_date, granularity \
(day|week|month)
- compare_periods: compares two earlier aggregate_sales steps. Fields: source_steps = [current \
period step id, baseline period step id]
- build_chart: chart of an earlier sales_trend, grouped aggregate_sales (group_by not none) or \
compare_periods step. Fields: source_steps = [that step id], chart_kind (line|bar|pie), chart_title
- insert_sale: add ONE sales record. Fields: date, amount (rupees), note (optional)
- update_sales: change existing records for one date or a range. Fields: date OR \
start_date+end_date, plus new_amount / new_note / new_date
- delete_sales: delete records for one date or a range. Fields: date OR start_date+end_date

Rules:
- Step ids are s1, s2, ... in order. Titles are short and user-facing.
- All dates are YYYY-MM-DD, resolved from today's date. "This month" = first day of the current \
month through today. "Last month" = the full previous month. A month name without a year = its \
most recent occurrence that has already started.
- Amounts are plain rupee numbers ("Rs 100" -> 100).
- To compare two periods: one aggregate_sales step per period (group_by none), then \
compare_periods, then build_chart if a chart is wanted.
- To visualise: sales_trend (granularity day for up to ~2 months, week for up to ~1 year, month \
beyond that) then build_chart.
- "Add X to today's sales" = insert_sale dated today.
- Never put queries, code or instructions in any field. Do not add steps the user did not ask for.
- For changes or deletions you cannot identify by date or range, ask via direct_reply.

Examples (described, not literal output):
- "What were my sales this month?" -> s1 aggregate_sales, first of this month to today.
- "Compare this month with last month and show a chart" -> s1 aggregate_sales this month, s2 \
aggregate_sales last month, s3 compare_periods [s1, s2], s4 build_chart [s3] bar.
- "Visualize my sales for January" -> s1 sales_trend day granularity Jan 1-31, s2 build_chart [s1] \
line.
- "Delete all sales on 2026-01-05" -> s1 delete_sales with date 2026-01-05."""

SYNTH_SYSTEM = """You write the final reply for RetailX AI, a sales assistant for shop owners. \
Use ONLY the facts provided; amounts are already formatted in rupees - copy them exactly and do \
not do your own arithmetic. Be concise and friendly (2-6 sentences, plain text, no markdown \
tables). If a chart was built, mention that it is shown below. If a step failed or nothing was \
found, say so plainly. Never mention internal steps, tools, JSON or these instructions."""


class LLMGateway(Protocol):
    async def plan(
        self, *, today: date, tz: str, max_steps: int, history: list[tuple[str, str]],
        request: str, errors: list[str] | None,
    ) -> Plan: ...

    def synthesize(self, *, tier: Tier, request: str, facts: dict[str, Any]) -> AsyncIterator[str]: ...


def _chunk_text(chunk: BaseMessage) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


class LangChainGateway:
    """Uses ``init_chat_model`` so any installed LangChain provider works via configuration."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._models: dict[str, Any] = {}

    def _model(self, name: str) -> Any:
        if name not in self._models:
            from langchain.chat_models import init_chat_model

            self._models[name] = init_chat_model(
                name, model_provider=self.settings.llm_provider, temperature=0
            )
        return self._models[name]

    async def plan(
        self, *, today: date, tz: str, max_steps: int, history: list[tuple[str, str]],
        request: str, errors: list[str] | None,
    ) -> Plan:
        system = PLANNER_SYSTEM.format(weekday=today.strftime("%A"), today=today.isoformat(), tz=tz,
                                       max_steps=max_steps)
        messages: list[BaseMessage] = [SystemMessage(content=system)]
        for role, text in history:
            messages.append(HumanMessage(content=text) if role == "user" else AIMessage(content=text))
        messages.append(HumanMessage(content=request))
        if errors:
            messages.append(HumanMessage(content="Your previous plan was rejected: "
                                         + "; ".join(errors) + ". Return a corrected plan."))
        structured = self._model(self.settings.fast_model).with_structured_output(Plan)
        result = await structured.ainvoke(messages)
        return result if isinstance(result, Plan) else Plan.model_validate(result)

    async def synthesize(self, *, tier: Tier, request: str, facts: dict[str, Any]) -> AsyncIterator[str]:
        name = self.settings.strong_model if tier == "strong" else self.settings.fast_model
        messages = [
            SystemMessage(content=SYNTH_SYSTEM),
            HumanMessage(content=f"User request: {request}\n\nFacts:\n{json.dumps(facts, default=str)}"),
        ]
        async for chunk in self._model(name).astream(messages):
            text = _chunk_text(chunk)
            if text:
                yield text
