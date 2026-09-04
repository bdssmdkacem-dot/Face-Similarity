#!/usr/bin/env python3
"""Import licensed celebrity portraits from Wikimedia Commons into Supabase."""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import requests

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
EMBED_URL = "https://huggingface.co/LibreYOLO/librefacerec-l/resolve/main/librefacerec-l.onnx?download=true"
EMBED_SHA256 = "a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60"
DET_URL = "https://huggingface.co/LibreYOLO/librefacerec-det/resolve/main/librefacerec-det.onnx?download=true"
DET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
USER_AGENT = "ShabahGalleryImporter/1.4 (https://github.com/bdssmdkacem-dot/Face-Similarity)"

DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().strip().lower()


def download_verified(url: str, path: Path, expected_sha256: str) -> None:
    expected = expected_sha256.strip().lower()
    if path.exists() and sha256_file(path) == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(1, 4):
        try:
            r = requests.get(url, timeout=180, stream=True, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
            actual = sha256_file(tmp)
            if actual == expected:
                tmp.replace(path)
                return
            tmp.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"SHA256 mismatch for {url}: expected={expected} actual={actual}")
            time.sleep(attempt)
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(attempt)


def _retry_after(response: requests.Response, fallback: float) -> float:
    value = response.headers.get("Retry-After", "").strip()
    try:
        return max(1.0, min(float(value), 3600.0))
    except ValueError:
        return fallback


def commons_candidates(name: str, limit: int = 12) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f'File:"{name}"',
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 960,
    }
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    for attempt in range(1, 5):
        try:
            r = requests.get(COMMONS_API, params=params, timeout=30, headers=headers)
            if r.status_code in (429, 503):
                wait = _retry_after(r, min(60.0, 2.0 ** attempt))
                print(f"  Commons rate limit/server busy ({r.status_code}); waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", {})
            out = []
            for page in pages.values():
                info = (page.get("imageinfo") or [{}])[0]
                meta = info.get("extmetadata") or {}
                license_name = (meta.get("LicenseShortName", {}).get("value") or "").strip()
                normalized = license_name.casefold()
                allowed_base = normalized.startswith("cc by") or normalized in {"cc0", "public domain", "pdm"}
                forbidden = (
                    "noncommercial" in normalized
                    or "no derivatives" in normalized
                    or normalized.endswith("-nc")
                    or normalized.endswith("-nd")
                    or " cc nc" in f" {normalized}"
                    or " cc nd" in f" {normalized}"
                )
                if not allowed_base or forbidden:
                    continue
                image_url = info.get("thumburl") or info.get("url")
                if image_url:
                    out.append({
                        "title": page.get("title", ""),
                        "image_url": image_url,
                        "source_url": "https://commons.wikimedia.org/wiki/" + quote(page.get("title", "").replace(" ", "_")),
                        "license_type": license_name,
                    })
            return out
        except requests.RequestException as exc:
            if attempt == 4:
                print(f"  Commons lookup failed after retries: {exc}")
                return []
            wait = min(30.0, 2.0 ** attempt)
            print(f"  Commons lookup error; retrying in {wait:.0f}s: {exc}")
            time.sleep(wait)
    return []


def slugify(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-").replace("--", "-")


def rest_headers(key: str) -> dict[str, str]:
    return {"apikey": key, "Content-Type": "application/json", "Prefer": "return=representation"}


def verify_admin_key(base: str, key: str) -> None:
    r = requests.get(f"{base}/rest/v1/", headers={"apikey": key}, timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(
            f"Supabase admin key verification failed ({r.status_code}). "
            "Ensure SHABAH_SUPABASE_SERVICE_ROLE_KEY contains the Face-Similarity sb_secret_* key. "
            f"Response: {r.text[:500]}"
        )


def get_existing_celebrity(base: str, key: str, slug: str) -> dict | None:
    r = requests.get(f"{base}/rest/v1/celebs", headers=rest_headers(key), params={"slug": f"eq.{slug}", "select": "id,name,slug,is_active", "limit": "1"}, timeout=30)
    r.raise_for_status()
    rows = r.json()
    return rows[0] if rows else None


def detect_landmarks(detector: cv2.FaceDetectorYN, image: np.ndarray) -> tuple[np.ndarray, float] | None:
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None
    face = max(faces, key=lambda row: float(row[14]))
    kps = np.array([[face[4], face[5]], [face[6], face[7]], [face[8], face[9]], [face[12], face[13]], [face[10], face[11]]], dtype=np.float32)
    return kps[[1, 0, 2, 3, 4]], float(face[14])


def embed_image(path: Path, detector: cv2.FaceDetectorYN, session: cv2.dnn.Net) -> tuple[list[float], float] | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    detected = detect_landmarks(detector, image)
    if detected is None:
        return None
    kps, detection_score = detected
    matrix, _ = cv2.estimateAffinePartial2D(kps, DST, method=cv2.LMEDS)
    if matrix is None:
        return None
    aligned = cv2.warpAffine(image, matrix, (112, 112), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None, ...]
    session.setInput(tensor)
    output = session.forward().reshape(-1).astype(np.float32)
    if output.size != 512:
        raise RuntimeError(f"Expected 512 embedding values, got {output.size}")
    norm = float(np.linalg.norm(output))
    if norm <= 1e-12:
        return None
    output /= norm
    return output.astype(float).tolist(), detection_score


def rest_insert(base: str, key: str, table: str, payload: dict) -> dict:
    r = requests.post(f"{base}/rest/v1/{table}", headers=rest_headers(key), json=payload, timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase {table} insert failed ({r.status_code}): {r.text[:500]}")
    return r.json()[0]


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", default="data/celebrity_names.txt")
    parser.add_argument("--cache", default=".gallery-cache")
    parser.add_argument("--max-per-celebrity", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="0 = all names")
    args = parser.parse_args()
    base = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not base or not key:
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the environment; never commit the secret key.")
    verify_admin_key(base, key)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    embed_model = cache / "librefacerec-l.onnx"
    det_model = cache / "librefacerec-det.onnx"
    download_verified(EMBED_URL, embed_model, EMBED_SHA256)
    download_verified(DET_URL, det_model, DET_SHA256)
    detector = cv2.FaceDetectorYN.create(str(det_model), "", (640, 640), 0.55, 0.3, 5000)
    embedder = cv2.dnn.readNetFromONNX(str(embed_model))
    names = read_names(Path(args.names))
    if args.limit:
        names = names[: args.limit]
    session = requests.Session()
    imported = 0
    for name in names:
        print(f"[gallery] {name}")
        candidates = commons_candidates(name)
        if not candidates:
            print("  no acceptable licensed Commons portrait found")
            time.sleep(1.5)
            continue
        slug = slugify(name)
        celeb = get_existing_celebrity(base, key, slug)
        if celeb is None:
            celeb = rest_insert(base, key, "celebs", {"name": name, "slug": slug, "is_active": False, "popularity_score": 0})
        celeb_id = celeb["id"]
        existing = requests.get(f"{base}/rest/v1/celebrity_images", headers=rest_headers(key), params={"celebrity_id": f"eq.{celeb_id}", "select": "source_url"}, timeout=30)
        existing.raise_for_status()
        existing_sources = {row["source_url"] for row in existing.json()}
        inserted_embeddings = 0
        for index, candidate in enumerate(candidates[: args.max_per_celebrity]):
            if candidate["source_url"] in existing_sources:
                continue
            image_path = cache / f"{slug}-{index}.jpg"
            try:
                response = session.get(candidate["image_url"], timeout=60, headers={"User-Agent": USER_AGENT})
                if response.status_code in (429, 503):
                    wait = _retry_after(response, 10.0)
                    print(f"  image server busy ({response.status_code}); waiting {wait:.0f}s")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                image_path.write_bytes(response.content)
                result = embed_image(image_path, detector, embedder)
                if result is None:
                    continue
                embedding, quality = result
                image_row = rest_insert(base, key, "celebrity_images", {"celebrity_id": celeb_id, "image_url": candidate["image_url"], "source_name": "Wikimedia Commons", "source_url": candidate["source_url"], "license_type": candidate["license_type"], "is_primary": not existing_sources and inserted_embeddings == 0})
                rest_insert(base, key, "celebrity_embeddings", {"celebrity_id": celeb_id, "image_id": image_row["id"], "embedding": embedding, "quality_score": quality})
                existing_sources.add(candidate["source_url"])
                inserted_embeddings += 1
            except Exception as exc:
                print(f"  skipped candidate: {exc}")
            finally:
                image_path.unlink(missing_ok=True)
        if inserted_embeddings or celeb.get("is_active"):
            patch = requests.patch(f"{base}/rest/v1/celebs?id=eq.{celeb_id}", headers=rest_headers(key), json={"is_active": True}, timeout=30)
            patch.raise_for_status()
            imported += 1
            print(f"  imported {inserted_embeddings} new embedding(s)")
        else:
            print("  no usable face embedding; celebrity remains inactive")
        time.sleep(1.5)
    print(f"[gallery] active celebrities processed: {imported}/{len(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
