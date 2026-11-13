import os
from typing import Optional
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized 12-factor configuration using Pydantic Settings.
    Eliminates scattered os.getenv calls across the application.
    """
    # Environment & Server
    APP_ENV: str = Field(default="development", description="Runtime environment (development, staging, production)")
    HOST: str = Field(default="0.0.0.0", description="Host address for ASGI server")
    PORT: int = Field(default=8000, description="Listening port for ASGI server")
    LOG_LEVEL: str = Field(default="INFO", description="Standard logging level")

    # Database & Cache
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgrespassword@localhost:5433/resolveops",
        description="PostgreSQL connection string (supports Neon, Supabase, Render, AWS RDS)"
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string (supports rediss:// for Upstash / Redis Cloud)"
    )

    # Security & Webhook Signatures
    ADMIN_WEBHOOK_SECRET: str = Field(
        default="resolveops_default_development_secret_change_in_production",
        description="Shared secret key for HMAC-SHA256 signature verification"
    )

    # AI Model Providers
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API key")
    GEMINI_MODEL: str = Field(default="gemini-2.0-flash", description="Primary agent LLM model identifier")
    GEMINI_FALLBACK_MODEL: str = Field(default="gemini-1.5-flash", description="Fallback LLM for resilience")

    # Business Rules Defaults (Overridden dynamically by PostgreSQL policies)
    MAX_AUTO_REFUND_LIMIT_DEFAULT: float = Field(default=100.0, description="Fallback auto-approval limit")
    RETURN_WINDOW_DAYS_DEFAULT: int = Field(default=30, description="Fallback return eligibility window in days")

    # Concurrency & Resilience
    REDIS_LOCK_TTL_SECONDS: int = Field(default=10, description="Distributed lock timeout in seconds")
    REDIS_LOCK_RETRY_TIMES: int = Field(default=3, description="Lock acquisition retry attempts")
    IDEMPOTENCY_TTL_SECONDS: int = Field(default=86400, description="Idempotency cache retention (24h)")
    LLM_TIMEOUT_SECONDS: int = Field(default=12, description="Maximum wait time for LLM response")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @field_validator("DATABASE_URL")
    @classmethod
    def normalize_database_url(cls, v: str) -> str:
        # Normalize postgres:// to postgresql:// if passed by some cloud hosts (e.g. Render/Heroku)
        if v.startswith("postgres://"):
            v = "postgresql://" + v[len("postgres://"):]
        return v


@lru_cache()
def get_settings() -> Settings:
    """Returns singleton cached instance of application settings."""
    return Settings()
