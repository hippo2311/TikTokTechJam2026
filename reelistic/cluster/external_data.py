"""Shared safeguards for staging SID-Set and WildFake training subsets."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping


SID_LABEL_NAMES = {0: "real", 1: "full_synthetic", 2: "tampered"}
_SAFE_COMPONENT = re.compile(r"[^a-zA-Z0-9._-]+")


def sid_binary_label(label: Any) -> int:
    """Map SID's real/synthetic/tampered labels to this project's binary task."""
    value = int(label)
    if value not in SID_LABEL_NAMES:
        raise ValueError(f"Unexpected SID label: {label!r}")
    return 0 if value == 0 else 1


def wildfake_exclusion_reason(row: Mapping[str, Any]) -> str | None:
    """Reject the competition demonstration subset before any split is made."""
    is_fake = int(row.get("IsFake", 0)) == 1
    is_advanced = int(row.get("IsAdvanced", 0)) == 1
    searchable = " ".join(
        str(row.get(key, ""))
        for key in ("Generator", "Architecture", "Weight", "Category", "Image_path")
    ).lower()
    compact = searchable.replace("·", "").replace("-", "").replace("_", "")

    if not is_fake and "coco" in compact and ("val2017" in compact or "val" in compact):
        return "reserved_demo_coco_val2017"
    if is_fake and is_advanced and "dalle" in compact:
        return "reserved_demo_dalle_advanced"
    return None


def stable_score(seed: int, *parts: Any) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def safe_component(value: Any, fallback: str = "sample") -> str:
    cleaned = _SAFE_COMPONENT.sub("_", str(value)).strip("._")
    return cleaned[:120] or fallback
