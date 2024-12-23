from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class FeatureResponse(BaseModel):
    """Response object for a feature."""

    id: UUID
    label: str
    index_in_sae: int


class SearchFeatureResponseItem(FeatureResponse):
    """Response object for a feature with relevance score."""

    relevance: float = 1


class SearchFeatureResponse(BaseModel):
    """Response object for a list of features."""

    features: list[SearchFeatureResponseItem]


class FeatureDetailsResponse(BaseModel):
    """Response object for a feature with additional details."""

    features: list[FeatureResponse]


class ClusteringConfig(BaseModel):
    min_cluster_size: int = Field(default=5, ge=2)
    max_cluster_size: Optional[int] = Field(default=None)
    min_samples: Optional[int] = Field(default=None, ge=1)
    cluster_selection_method: str = Field(default="leaf", pattern="^(eom|leaf)$")
    cluster_selection_epsilon: float = Field(default=0.0, ge=0.0)
