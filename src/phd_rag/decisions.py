"""Deterministic quality gates used by the PhD recommendation agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceDecision:
    approved: bool
    reason: str
    top_score: float | None
    mean_top_scores: float | None
    unique_universities: int
    result_count: int
    min_top_score: float
    min_unique_universities: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_retrieval_evidence(
    results: list[dict[str, Any]],
    min_top_score: float = 0.30,
    min_unique_universities: int = 2,
) -> EvidenceDecision:
    """Decide whether retrieved cosine-similarity results support a recommendation.

    Thresholds are guardrails, not an accuracy guarantee; calibrate them with labelled
    questions before production use. The agent never interprets a weak result set as a
    positive university match.
    """
    valid = [item for item in results if isinstance(item, dict) and isinstance(item.get("score"), (int, float))]
    if not valid:
        return EvidenceDecision(
            approved=False,
            reason="No retrievable official source excerpts were returned.",
            top_score=None,
            mean_top_scores=None,
            unique_universities=0,
            result_count=0,
            min_top_score=min_top_score,
            min_unique_universities=min_unique_universities,
        )
    ordered = sorted(valid, key=lambda item: float(item["score"]), reverse=True)
    top_score = float(ordered[0]["score"])
    mean_top_scores = sum(float(item["score"]) for item in ordered[:3]) / min(3, len(ordered))
    universities = {str(item.get("university", "")).strip() for item in ordered if item.get("university")}
    if top_score < min_top_score:
        reason = f"Top cosine score {top_score:.3f} is below the configured threshold {min_top_score:.3f}."
        approved = False
    elif len(universities) < min_unique_universities:
        reason = (
            f"Only {len(universities)} unique university source(s) were retrieved; "
            f"need at least {min_unique_universities}."
        )
        approved = False
    else:
        reason = "Retrieved sources passed score and source-diversity guardrails."
        approved = True
    return EvidenceDecision(
        approved=approved,
        reason=reason,
        top_score=top_score,
        mean_top_scores=mean_top_scores,
        unique_universities=len(universities),
        result_count=len(ordered),
        min_top_score=min_top_score,
        min_unique_universities=min_unique_universities,
    )
