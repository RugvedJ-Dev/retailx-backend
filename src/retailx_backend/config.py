"""Application settings loaded from environment variables / .env."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: Literal["development", "production", "test"] = "development"
    cors_origins: str = "http://localhost:3000"
    timezone: str = "Asia/Kolkata"

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "retailx"

    # API session tokens
    jwt_secret: SecretStr
    jwt_issuer: str = "retailx-api"
    access_token_ttl_minutes: int = 720

    # MCP server access (tokens are minted by the API and verified by the MCP server)
    mcp_url: str = "http://localhost:8001/mcp"
    mcp_jwt_secret: SecretStr
    mcp_jwt_issuer: str = "retailx-api"
    mcp_jwt_audience: str = "retailx-mcp"
    mcp_token_ttl_seconds: int = 300
    mcp_timeout_seconds: float = 15.0

    # LLM tiers (provider-agnostic: any langchain `init_chat_model` provider)
    llm_provider: str = "groq"
    fast_model: str = "llama-3.1-8b-instant"
    strong_model: str = "llama-3.3-70b-versatile"

    # Agent policy
    approval_policy: Literal["all_writes", "destructive_only"] = "all_writes"
    approval_ttl_hours: int = 24
    max_plan_steps: int = 5
    history_window: int = 10
    read_cache_ttl_seconds: int = 60
    run_lock_seconds: int = 180

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @model_validator(mode="after")
    def _check_secrets(self) -> "Settings":
        api, mcp = self.jwt_secret.get_secret_value(), self.mcp_jwt_secret.get_secret_value()
        if self.env == "production":
            if len(api) < 32 or len(mcp) < 32:
                raise ValueError("JWT_SECRET and MCP_JWT_SECRET must be at least 32 characters")
            if api == mcp:
                raise ValueError("JWT_SECRET and MCP_JWT_SECRET must be different")
            if "change-me" in api or "change-me" in mcp:
                raise ValueError("Replace the placeholder secrets from .env.example")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
