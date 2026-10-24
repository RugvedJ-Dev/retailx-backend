"""Settings for the standalone MCP server (independent of the API settings)."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class McpSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "retailx"
    mcp_jwt_secret: SecretStr
    mcp_jwt_issuer: str = "retailx-api"
    mcp_jwt_audience: str = "retailx-mcp"
    max_bulk_affected: int = 500
    host: str = "0.0.0.0"
    port: int = 8001
