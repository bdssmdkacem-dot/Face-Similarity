#!/usr/bin/env python3
"""Build a deterministic 1000-person celebrity catalog from the curated list plus a public face-recognition celebrity list."""
from __future__ import annotations

from pathlib import Path
import re
import requests

SOURCE_URL = "https://raw.githubusercontent.com/prateekmehta59/Celebrity-Face-Recognition-Dataset/master/List%20of%20Celebrities"
TARGET = Path("data/celebrity_names.txt")
TEMP = Path("data/celebrity_names.generated.txt")
TARGET_COUNT = 1000


def clean_name(value: str) -> str | None:
    value = re.sub(r"\s+", " ", value.strip())
    if not value or len(value) < 3 or len(value) > 80:
        return None
    if any(ch in value for ch in "\t\r\n<>{}[]|"):
        return None
    if sum(ch.isalpha() for ch in value) < 3:
        return None
    return value


def unique_names(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in values:
        name = clean_name(raw)
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(name)
    return result


def main() -> int:
    existing = TARGET.read_text(encoding="utf-8").splitlines() if TARGET.exists() else []
    response = requests.get(SOURCE_URL, timeout=60, headers={"User-Agent": "ShabahGalleryCatalog/1.0"})
    response.raise_for_status()
    external = response.text.splitlines()

    names = unique_names(existing + external)
    if len(names) < TARGET_COUNT:
        raise RuntimeError(f"Celebrity source produced only {len(names)} unique names; need {TARGET_COUNT}")

    names = names[:TARGET_COUNT]
    TEMP.write_text("\n".join(names) + "\n", encoding="utf-8")
    TARGET.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"[catalog] wrote {len(names)} unique celebrity names to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
