"""Agent error types."""


class PlanError(Exception):
    """The planner produced a plan that failed deterministic validation."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class StepError(Exception):
    """A plan step failed. The message is safe to show to the user."""


class ToolFailure(StepError):
    """An MCP tool call failed (validation, authorization, timeout or outage)."""
