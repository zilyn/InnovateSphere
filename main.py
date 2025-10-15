from flask import Flask, request, jsonify
import requests, os, re, hashlib
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag, urlparse

app = Flask(__name__)

# ===== Weaviate (inlined for your test) =====
WEAVIATE_URL = os.environ.get(
    "WEAVIATE_URL",
    "https://54he2t8ht0e239j2vi4eow.c0.us-west3.gcp.weaviate.cloud"
)
WEAVIATE_API_KEY = os.environ.get(
    "WEAVIATE_API_KEY",
    "aXk0MEhaUVJldmVnT0JCb19MUnB4OWtsUDVaRTBnOVJmSTh3Z090WS84cmgzYW9tdmsvN1l5VEFVWmFFPV92MjAw"
)

# Use Collections API (v2) – no manual schema creation needed
COLLECTION_NAME = os.environ.get("WEAVIATE_COLLECTION", "DocChunk")
W_HEADERS = {
    "Authorization": f"Bearer {WEAVIATE_API_KEY}",
    "Content-Type": "application/json"
}
HEADERS = {"User-Agent": "InnovateSphere-RAG-Indexer/1.0"}

# ---------- helpers ----------
def same_site(seed, url):
    a, b = urlparse(seed), urlparse(url)
    return (a.scheme, a.netloc) == (b.scheme, b.netloc)

def clean(t): 
    return re.sub(r"\s+", " ", t or "").strip()

def extract(html, base_url):
    soup = BeautifulSoup(html, "lxml")
    for sel in ["nav","footer","script","style","noscript","aside"]:
        for tag in soup.select(sel): tag.decompose()
    main = soup.select_one("main") or soup.select_one("article") or soup.body
    if not main: return "", ""
    title = soup.title.string.strip() if soup.title and soup.title.string else urlparse(base_url).path
    text = clean(main.get_text(separator=" "))
    return title, text

def chunk(text, size=4000, overlap=600):
    out, n, i = [], len(text), 0
    while i < n:
        j = min(i+size, n)
        c = clean(text[i:j])
        if c: out.append(c)
        if j == n: break
        i = max(0, j-overlap)
    return out

def crawl(start_urls, max_pages=10):
    visited, q, pages = set(), list(start_urls), []
    while q and len(pages) < max_pages:
        url, _ = urldefrag(q.pop(0))
        if url in visited: continue
        visited.add(url)
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            if "text/html" not in r.headers.get("Content-Type",""): continue
            title, text = extract(r.text, url)
            if len(text) < 200: continue
            pages.append({"url": url, "title": title, "text": text})
            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.select("a[href]"):
                href = a.get("href") or ""
                absu = urljoin(url, href)
                if same_site(start_urls[0], absu) and absu not in visited:
                    q.append(absu)
        except Exception:
            continue
    return pages

# ---------- Weaviate v2 upsert (auto schema) ----------
def upsert_object_v2(props: dict):
    """
    Auto-schema: Weaviate will create the collection on first insert
    when using the v2 Objects API if it's not there yet.
    """
    url = f"{WEAVIATE_URL}/v2/objects"
    payload = {
        "collection": COLLECTION_NAME,
        "properties": props
    }
    r = requests.post(url, headers=W_HEADERS, json=payload, timeout=30)
    # 201 = created, 200 = OK (depends on cluster)
    if r.status_code not in (200, 201):
        # Surface helpful error
        try:
            return False, r.status_code, r.json()
        except Exception:
            return False, r.status_code, r.text
    return True, r.status_code, None

# ---------- routes ----------
@app.route("/", methods=["GET"])
def health():
    return "OK", 200

@app.route("/", methods=["POST"])
def indexer():
    body = request.get_json(force=True) or {}
    source = body.get("source", "default")
    start_urls = body.get("start_urls", [])
    max_pages = int(body.get("max_pages", 10))

    if not start_urls:
        return jsonify({"error": "start_urls required"}), 400
    if not WEAVIATE_URL or not WEAVIATE_API_KEY:
        return jsonify({"error": "Missing Weaviate credentials"}), 500

    pages = crawl(start_urls, max_pages=max_pages)

    upserted = 0
    errors = []
    for p in pages:
        for c in chunk(p["text"]):
            ok, code, detail = upsert_object_v2({
                "text": c,
                "url": p["url"],
                "title": p["title"],
                "source": source
            })
            if ok:
                upserted += 1
            else:
                errors.append({"status": code, "detail": detail})

    resp = {"upserted": upserted, "namespace": source, "pages": len(pages)}
    if errors:
        resp["errors"] = errors[:3]  # show first few if any
    return jsonify(resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
