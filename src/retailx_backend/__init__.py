"""RetailX AI backend package."""

import os


def main() -> None:
    """Run the FastAPI gateway (``retailx-backend`` console script)."""
    import uvicorn

    uvicorn.run(
        "retailx_backend.api.app:create_app",
        factory=True,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
