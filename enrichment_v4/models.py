from __future__ import annotations
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, AliasChoices, field_validator, model_validator


def company_number(value: str) -> str:
    value = re.sub(r"\s+", "", value).upper()
    if value.isdigit() and 1 <= len(value) <= 8:
        return value.zfill(8)
    if re.fullmatch(r"[A-Z]{2}\d{6}", value):
        return value
    raise ValueError("Expected a UK company number, e.g. 01234567 or SC123456")


class EnrichRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, str_strip_whitespace=True)
    company_number: str | None = None
    company_name: str | None = Field(None, max_length=200, validation_alias=AliasChoices("company_name", "company"))
    query: str | None = Field(None, max_length=300, description="Single business name; use /discover-businesses for sector searches")
    domain: str | None = Field(None, max_length=2048)
    phone_number: str | None = Field(None, max_length=50, validation_alias=AliasChoices("phone_number", "phone"))
    director_name: str | None = Field(None, max_length=200)
    postcode: str | None = Field(None, max_length=12)
    location: str | None = Field(None, max_length=200)
    country: Literal["GB", "UK"] = "GB"
    corroboration_urls: list[str] = Field(default_factory=list, max_length=8)
    refresh: bool = False

    @field_validator("company_number")
    @classmethod
    def number(cls, value):
        return company_number(value) if value else None

    @field_validator("corroboration_urls")
    @classmethod
    def urls(cls, values):
        if any(len(v) > 2048 for v in values):
            raise ValueError("URL too long")
        return values

    @model_validator(mode="after")
    def identifiers(self):
        if not any((self.company_number, self.company_name, self.query, self.domain)):
            raise ValueError("Provide company_number, company_name, query or domain")
        return self


class DiscoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=2, max_length=300)
    location: str = Field(default="United Kingdom", max_length=200)
    max_results: int = Field(default=5, ge=1, le=20)
    offset: int = Field(default=0, ge=0, le=9)
    enrich: bool = False


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[EnrichRequest] = Field(min_length=1, max_length=20)
