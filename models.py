from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str = ""
    description: str = ""

    @classmethod
    def from_api_item(cls, item: object) -> "ModelInfo":
        if isinstance(item, dict):
            return cls(
                id=str(item.get("id", "")).strip(),
                name=str(item.get("name", "") or "").strip(),
                description=str(item.get("description", "") or "").strip(),
            )

        return cls(
            id=str(getattr(item, "id", "")).strip(),
            name=str(getattr(item, "name", "") or "").strip(),
            description=str(getattr(item, "description", "") or "").strip(),
        )

    def matches(self, text_filter: str | None) -> bool:
        if not text_filter:
            return True

        normalized_filter = text_filter.lower()
        return normalized_filter in self.id.lower() or normalized_filter in self.name.lower()
