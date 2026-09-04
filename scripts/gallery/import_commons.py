#!/usr/bin/env python3
"""Import licensed celebrity portraits from Wikimedia Commons into Supabase.

Required environment variables:
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY

The service-role key is intentionally required only by this offline/admin
pipeline and must never be placed in Flutter or committed to GitHub.

Pipeline:
  names -> Commons search -> license check -> image download -> YuNet 5-point
  detection -> ArcFace 112x112 alignment -> LibreFaceRec/AuraFace 512-d
  embedding -> Supabase celebs/images/embeddings.
"""
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
EMBED_URL = "https://huggingface.co/LibreYOLO/librefacerec-l/resolve/main/librefacerec-l.onnx"
EMBED_SHA256 = "a7933ea5330113b01c9b60351d8f4c33003f145d847ac5f0e52ee2effe25c60"
DET_URL = "https://huggingface.co/LibreYOLO/librefacerec-det/resolve/main/librefacerec-det.onnx"
DET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

# Standard ArcFace 112x112 five-point template.
DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

ALLOWED_LICENSE_PREFIXES = (
    "CC BY",
    "CC BY-SA",
    "CC0",
    "Public domain",
    "PDM",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_verified(url: str, path: Path, expected_sha256: str) -> None:
    if path.exists() and sha256_file(path) == expected_sha256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120, stream=True, headers={"User-Agent": "ShabahGalleryImporter/1.0"})
    r.raise_for_status()
    with path.open("wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
    actual = sha256_file(path)
    if actual != expected_sha256:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"SHA256 mismatch for {url}: {actual}")


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
        "iiurlwidth": 1200,
    }
    r = requests.get(COMMONS_API, params=params, timeout=30, headers={"User-Agent": "ShabahGalleryImporter/1.0"})
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    out = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        license_name = (meta.get("LicenseShortName", {}).get("value") or "").strip()
        if not license_name or not license_name.startswith(ALLOWED_LICENSE_PREFIXES):
            continue
        image_url = info.get("thumburl") or info.get("url")
        if image_url:
            out.append(
                {
                    "title": page.get("title", ""),
                    "image_url": image_url,
                    "source_url": "https://commons.wikimedia.org/wiki/" + quote(page.get("title", "").replace(" ", "_")),
                    "license_type": license_name,
                    "artist": (meta.get("Artist", {}).get("value") or "").strip(),
                }
            )
    return out


def slugify(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-").replace("--", "-")


def detect_landmarks(detector: cv2.FaceDetectorYN, image: np.ndarray) -> tuple[np.ndarray, float] | None:
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None
    # YuNet: x,y,w,h, 5 landmarks, score. Choose highest confidence.
    face = max(faces, key=lambda row: float(row[14]))
    kps = np.array(
        [
            [face[4], face[5]],
            [face[6], face[7]],
            [face[8], face[9]],
            [face[12], face[13]],
            [face[10], face[11]],
        ],
        dtype=np.float32,
    )
    # OpenCV YuNet order is right eye, left eye, nose, right mouth, left mouth;
    # reorder to ArcFace convention: left eye, right eye, nose, left mouth, right mouth.
    kps = kps[[1, 0, 2, 4, 3]]
    return kps, float(face[14])


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
    input_name = session.getLayerNames()[0] if False else None
    # cv2.dnn.Net uses the single input blob without requiring a named input.
    session.setInput(tensor)
    output = session.forward().reshape(-1).astype(np.float32)
    if output.size != 512:
        raise RuntimeError(f"Expected 512 embedding values, got {output.size}")
    norm = float(np.linalg.norm(output))
    if norm <= 1e-12:
        return None
    output /= norm
    return output.astype(float).tolist(), detection_score


def rest_headers(key: str) -> dict[str, str]:
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json", "Prefer": "return=representation"}


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
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the environment; never commit the service-role key.")

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
            continue

        celeb = rest_insert(base, key, "celebs", {"name": name, "slug": slugify(name), "is_active": False, "popularity_score": 0})
        celeb_id = celeb["id"]
        inserted_embeddings = 0
        for index, candidate in enumerate(candidates[: args.max_per_celebrity]):
            image_path = cache / f"{slugify(name)}-{index}.jpg"
            try:
                response = session.get(candidate["image_url"], timeout=60, headers={"User-Agent": "ShabahGalleryImporter/1.0"})
                response.raise_for_status()
                image_path.write_bytes(response.content)
                result = embed_image(image_path, detector, embedder)
                if result is None:
                    continue
                embedding, quality = result
                image_row = rest_insert(
                    base,
                    key,
                    "celebrity_images",
                    {
                        "celebrity_id": celeb_id,
                        "image_url": candidate["image_url"],
                        "source_name": "Wikimedia Commons",
                        "source_url": candidate["source_url"],
                        "license_type": candidate["license_type"],
                        "is_primary": inserted_embeddings == 0,
                    },
                )
                rest_insert(
                    base,
                    key,
                    "celebrity_embeddings",
                    {
                        "celebrity_id": celeb_id,
                        "image_id": image_row["id"],
                        "embedding": embedding,
                        "quality_score": quality,
                        "model_name": "librefacerec-l",
                    },
                )
                inserted_embeddings += 1
            except Exception as exc:
                print(f"  skipped candidate: {exc}")
            finally:
                image_path.unlink(missing_ok=True)

        if inserted_embeddings:
            # Activate only after at least one verified embedding exists.
            patch = requests.patch(
                f"{base}/rest/v1/celebs?id=eq.{celeb_id}",
                headers=rest_headers(key),
                json={"is_active": True},
                timeout=30,
            )
            patch.raise_for_status()
            imported += 1
            print(f"  imported {inserted_embeddings} embedding(s)")
        else:
            # Keep the metadata row inactive if no usable face was extracted.
            print("  no usable face embedding; celebrity remains inactive")
        time.sleep(0.2)

    print(f"[gallery] active celebrities imported: {imported}/{len(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
