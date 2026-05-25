from typing import Literal, List
from pydantic import BaseModel, Field


class ComplexityClassification(BaseModel):
    complexity: Literal["SIMPLE", "ADVANCED"]


class QueryDecomposition(BaseModel):
    """Sub-queries for multi-angle retrieval (theory, architecture, experiments)."""
    main_query: str
    theory_query: str = ""
    architecture_query: str = ""
    experiments_query: str = ""


class CriticChecklist(BaseModel):
    addresses_all_parts: bool = Field(
        description="True if every explicit part of the user question is addressed",
    )
    has_experimental_evidence: bool = Field(
        description="True if numeric results, metrics, or table references appear when required",
    )
    has_architectural_detail: bool = Field(
        description="True if named architectural components are discussed when required",
    )
    has_theoretical_comparison: bool = Field(
        description="True if theoretical/complexity comparisons are present when required",
    )
    claims_supported_by_sources: bool = Field(
        description="True if major claims are grounded in provided source excerpts",
    )


class CriticVerdict(BaseModel):
    verdict: Literal["COMPLETE", "INCOMPLETE", "PARTIAL"]
    missing_points: List[str] = Field(
        default_factory=list,
        description="Specific gaps; required when verdict is INCOMPLETE or PARTIAL",
    )
    checklist: CriticChecklist


class FinalCriticVerdict(BaseModel):
    verdict: Literal["COMPLETE", "INCOMPLETE", "PARTIAL"]
    missing_points: List[str] = Field(default_factory=list)
    checklist: CriticChecklist
