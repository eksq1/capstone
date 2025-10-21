"""
api_fastapi.py — Tupick RAG (FastAPI 백엔드)
===========================================
Streamlit 단일앱(main.py)을 **FastAPI API 서버**로 변환한 버전입니다.

기능 요약
- POST /ingest : URL 다건 수집 → 청크 → 임베딩 → FAISS 인덱스(증분) → 저장
- POST /query  : 벡터검색 + BM25 재순위 → LLM 리포트(A~H 템플릿, 성향 반영)
- GET  /health : 상태/백엔드 확인
- GET  /stats  : 인덱스 문서 수
- (선택) GET /docs : 자동 API 문서 (uvicorn 실행 시)


실행
-----
    pip install fastapi uvicorn[standard] requests beautifulsoup4 sentence-transformers faiss-cpu rank-bm25 pydantic python-dotenv
    # (선택) OpenAI 사용 시
    pip install openai

    # (선택) SPA(JS) 렌더링을 쓰려면
    pip install playwright && playwright install chromium

    uvicorn api_fastapi:app --reload
    # → http://127.0.0.1:8000 /docs
    
    터미널에 입력하면 실행
    .\venv\Scripts\python.exe -m uvicorn api_fastapi:app --reload --host 127.0.0.1 --port 8000 

환경 변수(.env)
---------------
- TUPICK_DATA_DIR (기본: Windows면 C:\\tupick_data, 그 외: ./tupick_data)
- EMB_MODEL_NAME (기본: BAAI/bge-m3)
- OPENAI_API_KEY / OPENAI_MODEL (기본: gpt-4o-mini)
- OLLAMA_HOST (기본: http://localhost:11434) / OLLAMA_MODEL (기본: llama3.1:8b-instruct)
- TOP_K (기본: 5)

주의
----
- 각 사이트의 약관/robots.txt/저작권을 준수하세요.
- 본 백엔드는 교육/연구용 데모이며, 투자 권유가 아닙니다.
"""

from __future__ import annotations
import os
import re
import json
import uuid
import datetime
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from fastapi.responses import HTMLResponse, Response

_profile_cache = {}

# === [ADD] User Profile 모델/스토리지 ===
from pydantic import BaseModel
import json, os

class UserProfile(BaseModel):
    risk: str = "중간"
    budget: int = 1_000_000
    goal: str = "단기현금흐름"
    interest_field: str = "부동산, 예술, 음악"
    experience_level: str = "초보자"

DATA_DIR = os.path.join(os.getcwd(), "tupick_data")
PROFILE_PATH = os.path.join(DATA_DIR, "profile.json")

def load_profile():
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return UserProfile().model_dump()

def save_profile(p):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)

profile_cache = load_profile()

# ----------------------------
# 환경설정 & 경로
# ----------------------------
load_dotenv()

EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "BAAI/bge-m3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", 5))
CHUNK_SIZE = 550
CHUNK_OVERLAP = 60


def _default_data_dir() -> str:
    env = os.getenv("TUPICK_DATA_DIR")
    if env:
        return env
    if os.name == "nt":
        return r"C:\\tupick_data"  # Windows ASCII-safe 경로 (FAISS 유니코드 버그 회피)
    return os.path.join(os.getcwd(), "tupick_data")

DATA_DIR = _default_data_dir()
INDEX_DIR = os.path.join(DATA_DIR, "index")
DOCS_PATH = os.path.join(DATA_DIR, "docs.jsonl")

os.makedirs(INDEX_DIR, exist_ok=True)

# ----------------------------
# 데이터 스키마
# ----------------------------
class DocChunk(BaseModel):
    id: str
    source: str
    category: str
    title: str
    section: str
    url: str
    as_of_date: str
    text: str
    language: str = "ko-KR"


class IngestReq(BaseModel):
    urls: List[str]
    source: str = Field("generic")
    use_js: bool = Field(False, description="SPA/차단 페이지 JS 렌더 사용")


class QueryReq(BaseModel):
    question: str
    risk: str = "중간"
    budget: int = 1_000_000
    goal: str = "단기현금흐름"
    k: int = 5
    use_profile: bool = True   


class QueryRes(BaseModel):
    answer: str
    backend: str
    k: int
    sources: List[Dict[str, Any]]

# ----------------------------
# 유틸/전처리
# ----------------------------

def now_date() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d")


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    toks = normalize_ws(text).split(" ")
    out, i = [], 0
    while i < len(toks):
        out.append(" ".join(toks[i:i + size]))
        i += max(1, size - overlap)
    return [c for c in out if len(c) > 20]


def guess_category_from_url(url: str) -> str:
    host = (requests.utils.urlparse(url).netloc or "").lower()
    if "tessa" in host:
        return "미술품/명품"
    if "music" in host:
        return "저작권/음원"
    if "kasa" in host:
        return "부동산"
    return "조각투자/일반"


def fetch_url(url: str, timeout: int = 20, use_js: bool = False) -> Tuple[str, str]:
    """1차: requests+BS4, 본문이 작거나 실패 시(use_js=True) Playwright 폴백"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept-Language": "ko,ko-KR;q=0.9,en-US;q=0.8",
        "Cache-Control": "no-cache",
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.text.strip() if soup.title else url
        for t in soup(["script", "style", "noscript"]):
            t.extract()
        body = normalize_ws(soup.get_text(" "))
        if use_js and len(body) < 800:
            raise RuntimeError("Body too small; try JS render")
        return title, body
    except Exception as e:
        if not use_js:
            raise
        # JS 렌더 폴백
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                ctx = browser.new_context(locale="ko-KR", user_agent=headers["User-Agent"])
                page = ctx.new_page()
                page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
                title = page.title() or url
                body = page.locator("body").inner_text(timeout=timeout * 1000)
                browser.close()
                return title, normalize_ws(body)
        except Exception as e2:
            raise RuntimeError(f"fetch_url failed for {url}: {e2}")


def save_jsonl(items: List[Dict[str, Any]], path: str):
    with open(path, "a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

# ----------------------------
# 임베딩/인덱스 (글로벌 in-memory + 디스크 저장)
# ----------------------------
_embedder: Optional[SentenceTransformer] = None
_index: Optional[faiss.IndexFlatIP] = None
_texts: List[str] = []
_metas: List[Dict[str, Any]] = []
_bm25: Optional[BM25Okapi] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMB_MODEL_NAME)
    return _embedder


def encode_texts(texts: List[str]) -> np.ndarray:
    emb = get_embedder()
    vecs = emb.encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype="float32")


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def index_paths() -> Tuple[str, str]:
    return os.path.join(INDEX_DIR, "faiss.index"), os.path.join(INDEX_DIR, "meta.json")


def save_index(index: faiss.IndexFlatIP):
    faiss_path, meta_path = index_paths()
    try:
        faiss.write_index(index, faiss_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"count": len(_texts)}, f, ensure_ascii=False, indent=2)
    except Exception:
        # Windows 한글 경로 이슈 폴백
        fb_dir = os.path.join(r"C:\\tupick_data", "index") if os.name == "nt" else INDEX_DIR
        os.makedirs(fb_dir, exist_ok=True)
        faiss.write_index(index, os.path.join(fb_dir, "faiss.index"))
        with open(os.path.join(fb_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"count": len(_texts)}, f, ensure_ascii=False, indent=2)


def load_index() -> Optional[faiss.IndexFlatIP]:
    candidates = []
    if os.name == "nt":
        candidates.append(os.path.join(r"C:\\tupick_data", "index", "faiss.index"))
    faiss_path, _ = index_paths()
    candidates.append(faiss_path)
    for p in candidates:
        if os.path.exists(p):
            try:
                return faiss.read_index(p)
            except Exception:
                continue
    return None


def ensure_retriever():
    global _index, _bm25
    if _bm25 is None and _texts:
        _bm25 = BM25Okapi([t.split() for t in _texts])
    if _index is None and _texts:
        vecs = encode_texts(_texts)
        _index = build_index(vecs)
        save_index(_index)


def vec_search(query: str, k: int) -> List[int]:
    if _index is None or not _texts:
        return []
    qv = encode_texts([query])  # (1, dim)
    faiss.normalize_L2(qv)
    scores, idxs = _index.search(qv, max(k, 8))
    return idxs[0].tolist()


def bm25_rerank(query: str, candidates: List[int], k: int) -> List[int]:
    if not candidates:
        return []
    if _bm25 is None:
        return candidates[:k]
    scores = _bm25.get_scores(query.split())
    pairs = [(i, float(scores[i])) for i in candidates]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in pairs[:k]]

# ----------------------------
# LLM 백엔드
# ----------------------------
LLM_SYSTEM_PROMPT = (
    "너는 ‘조각투자 도메인 어시스턴트’야. 제공된 CONTEXT만을 근거로 한국어로 답변해. "
    "항상 [사용자 성향]의 리스크, 예산, 목표를 반드시 리포트에 반영해야 한다. "
    "숫자/기간/수익률 등은 컨텍스트에 있는 내용만 사용하고, 없으면 ‘자료 부족’이라고 말해. "
    "투자 권유처럼 단정하지 말고 정보 제공 관점으로 작성해. "
    "답변은 반드시 아래 리포트 템플릿 구조로 작성한다:\n"
    "A. 핵심요약 (3줄 이내, 사용자 성향 1줄 반영)\n"
    "B. 수익·비용 구조 요약 (근거 문장 끝에 [번호])\n"
    "C. 예산별 전략 (사용자 예산 기준 분할/티켓/유동성)\n"
    "D. 리스크 포인트 (사용자 리스크 선호 반영)\n"
    "E. 목표 적합성 평가 (사용자 목표 기준, [번호])\n"
    "F. 유사상품 비교 (있으면, [번호])\n"
    "G. 출처 (번호, 제목, URL)\n"
    "H. 안전 문구 (‘본 서비스는 정보 제공용이며, 수익 보장을 하지 않습니다.’)\n"
)


def build_user_prompt(question: str, passages: List[Dict[str, Any]], risk: str, budget: int, goal: str) -> str:
    ctx_lines = []
    for i, p in enumerate(passages, start=1):
        title = p.get("title", "Untitled"); url = p.get("url", ""); date = p.get("as_of_date", ""); text = p.get("text", "")
        ctx_lines.append(f"[{i}] {title} | {date} | {url}\n{text}")
    context = "\n\n".join(ctx_lines)
    return (
        f"[사용자 성향]\n리스크={risk}, 예산={budget}원, 목표={goal}\n\n"
        f"[질문]\n{question}\n\n"
        f"[CONTEXT]\n{context}\n"
    )


def call_ollama(prompt: str, system: str) -> str:
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


def call_openai(prompt: str, system: str) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# ----------------------------
# FastAPI 앱
# ----------------------------
app = FastAPI(title="Tupick RAG API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    try:
        global _texts, _metas, _index, _bm25
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(INDEX_DIR, exist_ok=True)
        
        # 기존 문서 로드 (에러 발생 가능성 있는 부분)
        rows = load_jsonl(DOCS_PATH)
        _texts = [r.get("text", "") for r in rows]
        _metas = [{k: r.get(k, "") for k in ["title","url","as_of_date","source","category","id","section"]} for r in rows]
        
        # 인덱스/BM25 준비
        if _texts:
            _bm25 = BM25Okapi([t.split() for t in _texts])
            _index = load_index()
            if _index is None:
                vecs = encode_texts(_texts)
                _index = build_index(vecs)
                save_index(_index)
        
        print("✅ 서버 시작 완료!")
        
    except Exception as e:
        print(f"⚠️ startup 에러: {e}")
        # 에러가 나도 서버는 실행되도록
        _texts = []
        _metas = []
        _index = None
        _bm25 = None


@app.get("/health")
def health():
    backend = f"OpenAI · {OPENAI_MODEL}" if OPENAI_API_KEY else f"Ollama · {OLLAMA_MODEL}"
    return {"status": "ok", "docs": len(_texts), "index": _index is not None, "backend": backend}

# === [ADD] User Profile API ===
@app.get("/users/profile")
def get_profile():
    return profile_cache

@app.post("/users/profile")
def set_profile(p: UserProfile):
    global profile_cache
    profile_cache = p.model_dump()
    save_profile(profile_cache)
    return {"ok": True, "profile": profile_cache}

@app.get("/stats")
def stats():
    return {"docs": len(_texts)}


@app.post("/ingest")
def ingest(req: IngestReq):
    global _texts, _metas, _index, _bm25
    if not req.urls:
        raise HTTPException(status_code=400, detail="urls empty")
    items: List[Dict[str, Any]] = []
    added = 0
    for u in req.urls:
        try:
            title, body = fetch_url(u, use_js=req.use_js)
            cat = guess_category_from_url(u)
            for ch in chunk_text(body):
                did = f"{req.source}_{uuid.uuid4().hex[:12]}"
                items.append(DocChunk(
                    id=did, source=req.source, category=cat, title=title, section="본문",
                    url=u, as_of_date=now_date(), text=ch
                ).model_dump())
        except Exception as e:
            # 개별 URL 실패는 계속 진행
            continue
    if not items:
        raise HTTPException(status_code=400, detail="no items crawled")

    # 디스크에 append
    save_jsonl(items, DOCS_PATH)
    added = len(items)

    # 메모리 갱신
    new_texts = [it["text"] for it in items]
    new_metas = [{k: it.get(k, "") for k in ["title","url","as_of_date","source","category","id","section"]} for it in items]
    _texts.extend(new_texts)
    _metas.extend(new_metas)

    # BM25 갱신 (전체 재생성)
    _bm25 = BM25Okapi([t.split() for t in _texts])

    # FAISS 인덱스 갱신 (증분: 없으면 신설, 있으면 add)
    new_vecs = encode_texts(new_texts)
    if _index is None:
        _index = build_index(new_vecs)
    else:
        faiss.normalize_L2(new_vecs)
        _index.add(new_vecs)
    save_index(_index)

    return {"added": added, "total": len(_texts)}


@app.post("/query", response_model=QueryRes)
def query(req: QueryReq):
    ensure_retriever()
    if not _texts or _index is None:
        raise HTTPException(status_code=400, detail="index empty; call /ingest first")

    # 1) 검색 → 재순위
    cand = vec_search(req.question, k=req.k)
    topk = bm25_rerank(req.question, cand, k=req.k)

    # 2) 컨텍스트 구성
    passages = []
    for i in topk:
        if 0 <= i < len(_texts):
            meta = _metas[i]
            passages.append({**meta, "text": _texts[i]})

    # 3) 프롬프트 & 백엔드 선택
    # 프로필 병합
    prof = profile_cache
    risk = req.risk or prof.get("risk", "중간")
    budget = req.budget or prof.get("budget", 1_000_000)
    goal = req.goal or prof.get("goal", "단기현금흐름")
    
    if req.use_profile:
        prof = _profile_cache or load_profile()
        risk = req.risk or prof.get("risk", "중간")
        budget = req.budget or prof.get("budget", 1_000_000)
        goal = req.goal or prof.get("goal", "단기현금흐름")
    else:
        risk, budget, goal = req.risk, req.budget, req.goal


    prompt = build_user_prompt(req.question, passages, risk, budget, goal)
    print("🧭 Prompt preview:", prompt[:400])
    backend = "openai" if OPENAI_API_KEY else "ollama"

    # 4) LLM 호출
    try:
        if backend == "openai":
            answer = call_openai(prompt, LLM_SYSTEM_PROMPT)
        else:
            answer = call_ollama(prompt, LLM_SYSTEM_PROMPT)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # 5) 출처 목록(번호 매핑) 반환
    sources = []
    for j, p in enumerate(passages, start=1):
        sources.append({
            "n": j,
            "title": p.get("title", "Untitled"),
            "url": p.get("url", ""),
            "as_of_date": p.get("as_of_date", ""),
            "source": p.get("source", ""),
        })

    return QueryRes(
        answer=answer,
        backend=(f"OpenAI · {OPENAI_MODEL}" if OPENAI_API_KEY else f"Ollama · {OLLAMA_MODEL}"),
        k=len(passages),
        sources=sources,
    )
    
# === [ADD] Product Comparison API ===
class CompareReq(BaseModel):
    platforms: list[str] = ["카사", "테사", "뮤직카우"]
    aspects: list[str] = ["수익률", "분배주기", "최소투자금", "리스크", "특징"]
    k: int = 5

@app.post("/compare", response_model=QueryRes)
def compare(req: CompareReq):
    ensure_retriever()
    if not _texts or _index is None:
        raise HTTPException(status_code=400, detail="index empty; call /ingest first")

    compare_q = f"{', '.join(req.platforms)}의 {', '.join(req.aspects)}를 표로 비교해줘."
    cand = vec_search(compare_q, k=req.k)
    topk = bm25_rerank(compare_q, cand, k=req.k)

    passages = [{**_metas[i], "text": _texts[i]} for i in topk if 0 <= i < len(_texts)]

    prompt = (
        "[비교 요청]\\n" + compare_q + "\\n\\n"
        "[CONTEXT]\\n" + "\\n\\n".join(
            f"[{i+1}] {p['title']} | {p['url']}\\n{p['text']}"
            for i, p in enumerate(passages)
        )
    )
    backend = "openai" if OPENAI_API_KEY else "ollama"
    answer = call_openai(prompt, LLM_SYSTEM_PROMPT) if backend == "openai" else call_ollama(prompt, LLM_SYSTEM_PROMPT)

    sources = [{"n": j+1, "title": p["title"], "url": p["url"]} for j, p in enumerate(passages)]
    return QueryRes(answer=answer, backend=backend, k=len(passages), sources=sources)


# 상단 import에 추가
from fastapi.responses import HTMLResponse, Response, RedirectResponse
import os

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return RedirectResponse(url="/docs")
    
# 이 라우트 추가!
@app.get("/index.html", response_class=HTMLResponse)
def read_index_html():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return {"error": "index.html not found"}

@app.get("/app.html", response_class=HTMLResponse)
def read_app():
    try:
        with open("app.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return {"error": "app.html not found"}

@app.get("/about.html", response_class=HTMLResponse)
def read_about():
    try:
        with open("about.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return {"error": "about.html not found"}
    
@app.get("/features.html", response_class=HTMLResponse)
def read_features():
    try:
        with open("features.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return {"error": "features.html not found"}
    
@app.get("/how-it-works.html", response_class=HTMLResponse)
def read_how_it_works():
    try:
        with open("how-it-works.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return {"error": "how-it-works.html not found"}

@app.get("/favicon.ico")
def favicon():
    # 파비콘 404 로그 없애기용
    return Response(status_code=204)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_fastapi:app", host="0.0.0.0", port=8000, reload=True)