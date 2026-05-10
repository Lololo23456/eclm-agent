"""Pydantic models for the ECLM API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    command: str = Field(..., description="Natural language command in French")
    target_file: str | None = Field(None, description="Target file path (optional)")
    behavior_tests: list[str] = Field(default_factory=list, description="Pytest test bodies")


class ProjectRequest(BaseModel):
    brief: str = Field(..., description="Project description in French")


class CompleteRequest(BaseModel):
    """Continue.dev inline completion format."""
    prefix: str = Field(..., description="Code before cursor")
    suffix: str = Field("", description="Code after cursor")
    file_path: str | None = None
    language: str = "python"
    max_tokens: int = 512


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    """Continue.dev chat format."""
    messages: list[ChatMessage]
    model: str = "eclm"
    stream: bool = False
    max_tokens: int = 2048


# ── Responses ─────────────────────────────────────────────────────────────────

class GenerateResponse(BaseModel):
    success: bool
    code: str
    score: float
    message: str
    written_to: str | None = None
    retries_used: int = 0


class TaskStatus(BaseModel):
    index: int
    label: str
    status: str
    score: float
    files_created: list[str]
    error: str | None = None


class ProjectResponse(BaseModel):
    session_id: str
    brief: str
    total: int
    done: int
    failed: int
    output_dir: str
    tasks: list[TaskStatus]
    test_score: float | None = None
    critic_issues: list[dict[str, str]] = Field(default_factory=list)


class ProjectListItem(BaseModel):
    id: str
    brief: str
    created_at: str
    tasks: int
    done: int
    tech_stack: list[str]


class CompleteResponse(BaseModel):
    completion: str


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: list[dict[str, Any]]
    model: str = "eclm"


class HealthResponse(BaseModel):
    status: str
    ollama_url: str
    fast_model: str
    strong_model: str
    dpo_pairs: int
