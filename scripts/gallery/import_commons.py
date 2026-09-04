#!/usr/bin/env python3
"""Import licensed celebrity portraits and matching 512-d embeddings."""
from __future__ import annotations
import argparse, hashlib, os, time
from pathlib import Path
from urllib.parse import quote
import cv2, numpy as np, requests

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
EMBED_URL = "https://huggingface.co/LibreYOLO/librefacerec-l/resolve/main/librefacerec-l.onnx"
EMBED_SHA256 = "a7933ea5330113b01c9b60351d8f4c33003f145d847ac5f0e52ee2effe25c60"
DET_URL = "https://huggingface.co/LibreYOLO/librefacerec-det/resolve/main/librefacerec-det.onnx"
DET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DST = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]], dtype=np.float32)
ALLOWED = ("CC BY", "CC BY-SA", "CC0", "Public domain", "PDM")


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def download_verified(url: str, path: Path, expected: str) -> None:
    if path.exists() and sha256_file(path)==expected: return
    path.parent.mkdir(parents=True, exist_ok=True)
    r=requests.get(url, timeout=120, stream=True, headers={"User-Agent":"ShabahGalleryImporter/1.0"}); r.raise_for_status()
    with path.open("wb") as f:
        for chunk in r.iter_content(1024*1024):
            if chunk: f.write(chunk)
    actual=sha256_file(path)
    if actual!=expected:
        path.unlink(missing_ok=True); raise RuntimeError(f"SHA256 mismatch: {actual}")


def candidates(name: str) -> list[dict]:
    p={"action":"query","format":"json","generator":"search","gsrsearch":f'File:"{name}"',"gsrnamespace":6,"gsrlimit":12,"prop":"imageinfo","iiprop":"url|extmetadata","iiurlwidth":1200}
    r=requests.get(COMMONS_API,params=p,timeout=30,headers={"User-Agent":"ShabahGalleryImporter/1.0"}); r.raise_for_status()
    out=[]
    for page in r.json().get("query",{}).get("pages",{}).values():
        info=(page.get("imageinfo") or [{}])[0]; meta=info.get("extmetadata") or {}
        license_name=(meta.get("LicenseShortName",{}).get("value") or "").strip()
        url=info.get("thumburl") or info.get("url")
        if url and license_name.startswith(ALLOWED):
            title=page.get("title","")
            out.append({"image_url":url,"source_url":"https://commons.wikimedia.org/wiki/"+quote(title.replace(" ","_")),"license_type":license_name})
    return out


def slugify(name:str)->str:
    return "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-").replace("--","-")


def detect(detector: cv2.FaceDetectorYN, image: np.ndarray):
    h,w=image.shape[:2]; detector.setInputSize((w,h)); _,faces=detector.detect(image)
    if faces is None or len(faces)==0: return None
    f=max(faces,key=lambda row:float(row[14]))
    # YuNet: right eye, left eye, nose, right mouth, left mouth -> ArcFace semantic order.
    pts=np.array([[f[6],f[7]],[f[4],f[5]],[f[8],f[9]],[f[12],f[13]],[f[10],f[11]]],dtype=np.float32)
    return pts,float(f[14])


def embed(path:Path, detector:cv2.FaceDetectorYN, net:cv2.dnn.Net):
    image=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if image is None:return None
    found=detect(detector,image)
    if found is None:return None
    pts,score=found; matrix,_=cv2.estimateAffinePartial2D(pts,DST,method=cv2.LMEDS)
    if matrix is None:return None
    aligned=cv2.warpAffine(image,matrix,(112,112),flags=cv2.INTER_LINEAR,borderValue=(0,0,0))
    rgb=cv2.cvtColor(aligned,cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor=((rgb-127.5)/127.5).transpose(2,0,1)[None,...]
    net.setInput(tensor); out=net.forward().reshape(-1).astype(np.float32)
    if out.size!=512: raise RuntimeError(f"Expected 512, got {out.size}")
    norm=float(np.linalg.norm(out))
    if norm<=1e-12:return None
    return (out/norm).astype(float).tolist(),score


def headers(key): return {"apikey":key,"Authorization":f"Bearer {key}","Content-Type":"application/json","Prefer":"return=representation"}

def insert(base,key,table,payload):
    r=requests.post(f"{base}/rest/v1/{table}",headers=headers(key),json=payload,timeout=30)
    if r.status_code>=300: raise RuntimeError(f"{table}: {r.status_code} {r.text[:500]}")
    return r.json()[0]


def names(path): return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--names",default="data/celebrity_names.txt"); ap.add_argument("--cache",default=".gallery-cache"); ap.add_argument("--max-per-celebrity",type=int,default=3); ap.add_argument("--limit",type=int,default=0); a=ap.parse_args()
    base=os.environ.get("SUPABASE_URL","").rstrip("/"); key=os.environ.get("SUPABASE_SERVICE_ROLE_KEY","")
    if not base or not key: raise SystemExit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
    cache=Path(a.cache); cache.mkdir(parents=True,exist_ok=True)
    em=cache/"librefacerec-l.onnx"; de=cache/"librefacerec-det.onnx"
    download_verified(EMBED_URL,em,EMBED_SHA256); download_verified(DET_URL,de,DET_SHA256)
    detector=cv2.FaceDetectorYN.create(str(de),"",(640,640),0.55,0.3,5000); net=cv2.dnn.readNetFromONNX(str(em)); ns=names(a.names); ns=ns[:a.limit] if a.limit else ns
    ok=0
    for name in ns:
        print(f"[gallery] {name}"); cs=candidates(name)
        if not cs: print("  no licensed Commons candidate"); continue
        celeb=insert(base,key,"celebs",{"name":name,"slug":slugify(name),"is_active":False,"popularity_score":0}); cid=celeb["id"]; count=0
        for i,c in enumerate(cs[:a.max_per_celebrity]):
            p=cache/f"{slugify(name)}-{i}.jpg"
            try:
                r=requests.get(c["image_url"],timeout=60,headers={"User-Agent":"ShabahGalleryImporter/1.0"}); r.raise_for_status(); p.write_bytes(r.content)
                result=embed(p,detector,net)
                if result is None: continue
                vector,quality=result; image=insert(base,key,"celebrity_images",{"celebrity_id":cid,"image_url":c["image_url"],"source_name":"Wikimedia Commons","source_url":c["source_url"],"license_type":c["license_type"],"is_primary":count==0})
                insert(base,key,"celebrity_embeddings",{"celebrity_id":cid,"image_id":image["id"],"embedding":vector,"quality_score":quality,"model_name":"librefacerec-l"}); count+=1
            except Exception as e: print(f"  skipped: {e}")
            finally: p.unlink(missing_ok=True)
        if count:
            requests.patch(f"{base}/rest/v1/celebs?id=eq.{cid}",headers=headers(key),json={"is_active":True},timeout=30).raise_for_status(); ok+=1; print(f"  imported {count}")
        time.sleep(.2)
    print(f"[gallery] active={ok}/{len(ns)}")


if __name__=="__main__": main()
