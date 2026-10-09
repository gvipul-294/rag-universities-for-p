from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class UniversitySource(BaseModel):
    country: str
    university: str
    url: HttpUrl
    focus: str = Field(
        default="graduate research / PhD admissions",
        description="The type of official page selected for ingestion.",
    )


class CandidateProfile(BaseModel):
    field: str
    degree: str
    research_interests: list[str]
    citizenship_or_residency: Optional[str] = None
    preferred_countries: list[str] = []
    budget_or_funding_need: Optional[str] = None
    test_scores: Optional[str] = None
    constraints: Optional[str] = None
