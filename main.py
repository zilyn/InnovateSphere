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
CLASS_NAME = os.environ.get("WEAVIATE_CLASS", "DocChunk")

HEADERS = {"User-Agent": "InnovateSphere-RAG-Indexer/1.0"}
W_HEADERS = {
    "Authorization": f"Bearer {WEAVIATE_API_KEY}",
    "Content-Type": "application/json"
}

def same_site(seed, url):
    a, b = urlparse(seed), urlparse(url)
    return (a.scheme, a.netloc) == (b.scheme, b.netloc)

def clean(t): 
    return re.sub(r"\s+", " ", t or "").strip()

def extract(html, base_url):
    soup = BeautifulSoup(html, "lxml")
    for sel in ["nav", "footer", "script", "style", "noscript", "aside"]:
        for tag in soup.select(sel):
            tag.decompose()
    main = soup.select_one("main") or soup.select_one("article") or soup.body
    if not main:
        return "", ""
    title = soup.title.string.strip() if soup.title and soup.title.string else urlparse(base_url).path
    text = clean(main.get_text(separator=" "))
    return title, text

def chunk(text, size=4000, overlap=600):
    out, n, i = [], len(text), 0
    while i < n:
        j = min(i + size, n)
        c = clean(text[i:j])
        if c:
            out.append(c)
        if j == n:
            break
        i = max(0, j - overlap)
    return out

def crawl(start_urls, max_pages=10):
    visited, q, pages = set(), list(start_urls), []
    while q and len(pages) < max_pages:
        url, _ = urldefrag(q.pop(0))
        if url in visited:
            continue
        visited.add(url)
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            if "text/html" not in r.headers.get("Content-Type", ""):
                continue
            title, text = extract(r.text, url)
            if len(text) < 200:
                continue
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

def ensure_class():
    # Creates a class using Weaviate-managed embeddings (text2vec-weaviate)
    try:
        schema = requests.get(f"{WEAVIATE_URL}/v1/schema", headers=W_HEADERS, timeout=20).json()
        if any(c.get('class') == CLASS_NAME for c in schema.get('classes', [])):
            return
    except Exception:
        pass

    body = {
        "class": CLASS_NAME,
        "vectorizer": "text2vec-weaviate",
        "properties": [
            {"name": "text", "dataType": ["text"]},
            {"name": "url", "dataType": ["text"]},
            {"name": "title", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]}
        ]
    }
    requests.post(
        f"{WEAVIATE_URL}/v1/schema/classes", 
        headers=W_HEADERS, json=body, timeout=30
    ).raise_for_status()

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

    ensure_class()

    pages = crawl(start_urls, max_pages=max_pages)

    upserted = 0
    for p in pages:
        for i, c in enumerate(chunk(p["text"])):
            obj = {
                "class": CLASS_NAME,
                "properties": {
                    "text": c,
                    "url": p["url"],
                    "title": p["title"],
                    "source": source
                }
            }
            try:
                r = requests.post(
                    f"{WEAVIATE_URL}/v1/objects",
                    headers=W_HEADERS,
                    json=obj,
                    timeout=30
                )
                if r.status_code in (200, 201):
                    upserted += 1
            except Exception:
                continue

    return jsonify({"upserted": upserted, "namespace": source, "pages": len(pages)})

if __name__ == "__main__":
    # Local run (Render will use the Procfile/gunicorn)
    app.run(host="0.0.0.0", port=8080)
