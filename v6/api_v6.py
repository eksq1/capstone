"""
api_fastapi.py — Tupick RAG (FastAPI 백엔드) - SQLite + Email Auth 통합
===========================================
기능:
- SQLite 데이터베이스로 사용자 관리
- 이메일 회원가입/로그인
- Google OAuth 로그인
- OpenAI GPT 사용
- JWT 토큰 인증
"""

from __future__ import annotations
import os
import re
import json
import uuid
import sqlite3
import secrets
import smtplib
import redis
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime
from typing import List, Dict

import time 
import numpy as np
import faiss
import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, Response, FileResponse
from fastapi.security import HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv
from jose import JWTError, jwt
from passlib.context import CryptContext
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# 환경 변수 로드
load_dotenv()

# ===== 설정 =====
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 1440  # 24시간

# OpenAI 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# SMTP 설정 (이메일 인증용)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

# 인증 코드 저장 (메모리)
verification_codes = {}

# Google OAuth 설정
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

# 데이터베이스 설정
DB_PATH = os.getenv("DB_PATH", "./tupick.db")

# RAG 설정
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "BAAI/bge-m3")
TOP_K = int(os.getenv("TOP_K", 5))
CHUNK_SIZE = 550
CHUNK_OVERLAP = 60

# Redis 설정
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:8000")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# 비밀번호 암호화
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# OAuth 설정
oauth = OAuth()
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name='google',
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={
            'scope': 'openid email profile',
            'redirect_uri': GOOGLE_REDIRECT_URI
        }
    )

# ===== 데이터베이스 관리 =====

@contextmanager
def get_db():
    """SQLite 연결 컨텍스트 매니저"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """데이터베이스 초기화"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 사용자 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            hashed_password TEXT,
            auth_type TEXT DEFAULT 'email',
            picture TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 프로필 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            risk TEXT DEFAULT '중간',
            budget INTEGER DEFAULT 1000000,
            goal TEXT DEFAULT '단기현금흐름',
            interest_field TEXT DEFAULT '부동산, 예술, 음악',
            experience_level TEXT DEFAULT '초보자',
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        ''')
        
        # 문서 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            source TEXT,
            category TEXT,
            title TEXT,
            section TEXT,
            url TEXT,
            as_of_date TEXT,
            text TEXT,
            language TEXT DEFAULT 'ko-KR',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
        )
        ''')
        
        # 채팅 히스토리
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            question TEXT,
            answer TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        ''')
        
        # 인덱스 생성
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(user_id)')
        
        conn.commit()
        print("✅ Database initialized!")

# ===== Pydantic 모델 =====

def now_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

class User(BaseModel):
    id: str
    email: str
    name: str
    auth_type: str = "email"
    picture: Optional[str] = None
    created_at: str = Field(default_factory=now_date)

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: User

class UserProfile(BaseModel):
    risk: str = "중간"
    budget: int = 1_000_000
    goal: str = "단기현금흐름"
    interest_field: str = "부동산, 예술, 음악"
    experience_level: str = "초보자"

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
    source: str = "generic"
    use_js: bool = False

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

class ChatMessage(BaseModel):
    role: str  # "user" or "bot"
    content: str
    timestamp: str

class ChatSession(BaseModel):
    id: str
    user_email: str
    timestamp: str
    messages: List[ChatMessage]

class SaveChatRequest(BaseModel):
    messages: List[ChatMessage]

# ===== 사용자 관리 함수 =====

def get_user_by_email(email: str) -> Optional[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        return dict(row) if row else None

def create_user(user_data: UserCreate) -> User:
    user_id = str(uuid.uuid4())
    
    # SHA-256 pre-hashing으로 72바이트 제한 완전히 제거
    prepared_password = _prepare_password_for_bcrypt(user_data.password)
    hashed_pw = pwd_context.hash(prepared_password)
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO users (id, email, name, hashed_password, auth_type)
        VALUES (?, ?, ?, ?, 'email')
        ''', (user_id, user_data.email, user_data.name, hashed_pw))
        
        # 기본 프로필 생성
        cursor.execute('INSERT INTO user_profiles (user_id) VALUES (?)', (user_id,))
        conn.commit()
    
    return User(
        id=user_id,
        email=user_data.email,
        name=user_data.name,
        auth_type="email"
    )

def create_oauth_user(email: str, name: str, picture: str = None) -> User:
    """OAuth 사용자 생성"""
    user_id = str(uuid.uuid4())
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO users (id, email, name, auth_type, picture)
        VALUES (?, ?, ?, 'google', ?)
        ''', (user_id, email, name, picture))
        
        cursor.execute('INSERT INTO user_profiles (user_id) VALUES (?)', (user_id,))
        conn.commit()
    
    return User(
        id=user_id,
        email=email,
        name=name,
        auth_type="google",
        picture=picture
    )

# ===== 비밀번호 관리 =====

def _prepare_password_for_bcrypt(password: str) -> str:
    """
    비밀번호를 bcrypt로 해싱하기 전에 SHA-256으로 pre-hash
    이렇게 하면 72바이트 제한을 완전히 우회하면서도 보안성 유지
    """
    import hashlib
    # SHA-256 해시 (항상 64자의 hex 문자열 반환)
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """비밀번호 검증 (pre-hashing 적용)"""
    prepared_password = _prepare_password_for_bcrypt(plain_password)
    return pwd_context.verify(prepared_password, hashed_password)

def validate_password_strength(password: str) -> Tuple[bool, str]:
    if len(password) < 8:
        return False, "비밀번호는 최소 8자 이상이어야 합니다"
    if len(password) > 20:
        return False, "비밀번호는 최대 20자까지 가능합니다"
    
    # 대문자, 소문자, 숫자, 특수문자 체크
    has_lower = bool(re.search(r"[a-z]", password))
    has_upper = bool(re.search(r"[A-Z]", password))
    has_digit = bool(re.search(r"\d", password))
    has_special = bool(re.search(r"[!@#$%^&*(),.?\":{}|<>]", password))
    
    # 4가지 중 2가지 이상 포함 확인
    count = sum([has_lower, has_upper, has_digit, has_special])
    if count < 2:
        return False, "비밀번호는 대문자, 소문자, 숫자, 특수문자 중 2가지 이상을 포함해야 합니다"
    
    return True, "OK"

# ===== JWT 토큰 관리 =====

login_attempts = defaultdict(list)

def check_rate_limit(email: str, max_attempts: int = 5, window_minutes: int = 15):
    now = datetime.now()
    attempts = login_attempts[email]
    attempts = [t for t in attempts if now - t < timedelta(minutes=window_minutes)]
    login_attempts[email] = attempts
    
    if len(attempts) >= max_attempts:
        raise HTTPException(
            status_code=429,
            detail=f"너무 많은 로그인 시도. {window_minutes}분 후 다시 시도하세요"
        )
    attempts.append(now)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="인증에 실패했습니다",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user_dict = get_user_by_email(email)
    if user_dict is None:
        raise credentials_exception
    
    return User(**{k: v for k, v in user_dict.items() if k != "hashed_password"})

# ===== RAG 관련 함수들 =====

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

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    toks = normalize_ws(text).split(" ")
    out, i = [], 0
    while i < len(toks):
        out.append(" ".join(toks[i:i + size]))
        i += max(1, size - overlap)
    return [c for c in out if len(c) > 20]

def encode_texts(texts: List[str]) -> np.ndarray:
    emb = get_embedder()
    vecs = emb.encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype="float32")

def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

def vec_search(query: str, k: int) -> List[int]:
    if _index is None or not _texts:
        return []
    qv = encode_texts([query])
    faiss.normalize_L2(qv)
    scores, idxs = _index.search(qv, max(k, 8))
    return idxs[0].tolist()

def bm25_rerank(query: str, candidates: List[int], k: int) -> List[int]:
    if not candidates or _bm25 is None:
        return candidates[:k]
    scores = _bm25.get_scores(query.split())
    pairs = [(i, float(scores[i])) for i in candidates]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in pairs[:k]]

def fetch_url(url: str, timeout: int = 20, use_js: bool = False) -> Tuple[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "ko,ko-KR;q=0.9",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.title.text.strip() if soup.title else url
    for t in soup(["script", "style", "noscript"]):
        t.extract()
    body = normalize_ws(soup.get_text(" "))
    return title, body

'''
def call_openai(prompt: str, system: str) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(400, "OpenAI API key not configured")
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", 
                     headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
'''
def call_openai(prompt: str, system: str) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(400, "OpenAI API key not configured")
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    
    # Retry 로직 (429 에러 대응)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions", 
                headers=headers, 
                json=payload, 
                timeout=120
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 15  # 15초, 30초, 45초
                    print(f"⚠️ OpenAI Rate limit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 최종 실패 시 사용자에게 친절한 메시지
                    raise HTTPException(
                        status_code=429,
                        detail="OpenAI API 요청 한도를 초과했습니다. 1분 후 다시 시도해주세요."
                    )
            else:
                # 다른 HTTP 에러는 그대로 전달
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"OpenAI API 오류: {e.response.text}"
                )
                
        except requests.exceptions.Timeout:
            raise HTTPException(408, "OpenAI API 응답 시간 초과")
            
        except Exception as e:
            raise HTTPException(500, f"OpenAI API 호출 실패: {str(e)}")

LLM_SYSTEM_PROMPT = """당신은 '조각투자 전문 상담 AI'입니다.  
사용자는 부동산, 미술품, 음원저작권 등 다양한 대체자산에 대한 조각투자 관련 질문을 합니다.  
당신의 역할은 조각투자 시장과 플랫폼에 대한 객관적·정보 중심의 설명을 제공하는 것입니다.

---
### 🔹답변 규칙
1. **정확성** — 제공된 자료와 일반적으로 알려진 정보를 바탕으로 신뢰성 있게 답변합니다.  
2. **전문성** — 조각투자 관련 용어(수익증권, 지분거래, 환금성 등)를 자연스럽게 활용합니다.  
3. **맥락 해석** — 조각투자와 직접 관련 없는 질문도 조각투자 관점에서 재해석합니다.  
4. **중립성 유지** — 투자 권유나 추천을 하지 않습니다. ("~하세요" 대신 "~할 수 있습니다")  
5. **실용성** — 일반 사용자가 이해하기 쉬운 방식으로 설명합니다.  
6. **면책 문구** — 모든 답변 마지막에 아래 문구를 반드시 포함합니다.  
   > "본 서비스는 정보 제공용이며, 수익 보장을 하지 않습니다."

---
### 🔹한국의 주요 조각투자 플랫폼
아래 목록은 참고용이며, 특정 플랫폼을 추천하지 않습니다.

#### 💼 부동산 조각투자
- **KASA(카사)** — 상업용 빌딩 기반의 수익증권 투자 플랫폼  
- **Funble(펀블)** — 디지털 수익증권(STO) 기반 부동산 조각투자 플랫폼  
- **루센트블록(LucentBlock)** — Sou.place 브랜드로 부동산 지분화 투자 서비스 운영

#### 🎨 미술품 조각투자
- **TESSA(테사)** — 미술품 지분 투자 및 블록체인 거래 기술 활용  
- **Art & Guide(아트앤가이드, 열매컴퍼니)** — 고가 미술작품 공동투자형 서비스  
- **SOTWO(서울옥션블루)** — 미술시장과 경매 연계형 조각투자 모델

#### 🎵 저작권·엔터테인먼트 조각투자
- **뮤직카우(Musicow)** — 음원 저작권료 수익을 기반으로 한 조각투자  
- **WEA(위아)** — 공연 IP 및 콘텐츠 저작권 투자형 플랫폼

#### ⚙️ 기타 대체자산 기반
- **피스(Piece)** — 명품·한정판 스니커즈 등 실물자산 조각투자  
- **소투(SOTWO)** — 실물 예술품 및 한정판 컬렉터블 중심 투자  
- **Pica(피카)** — 예술 및 컬렉터블 자산의 토큰화 투자 플랫폼

---
### 🔹답변 톤 & 스타일
- 격식 있고 신뢰감 있는 어조 사용  
- 기술적 용어나 금융 용어는 가능한 한 풀어서 설명  
- 플랫폼 간 비교를 요청받을 경우, 장단점을 균형 있게 기술  
- 사용자의 투자 판단을 대신하지 않고, 정보 전달에 집중  

---
### 🔹면책 조항 (모든 답변 마지막에 포함)
> 본 서비스는 정보 제공용이며, 수익 보장을 하지 않습니다.
"""

def build_user_prompt(question: str, passages: List[Dict[str, Any]], 
                     risk: str, budget: int, goal: str) -> str:
    ctx_lines = []
    for i, p in enumerate(passages, start=1):
        ctx_lines.append(f"[{i}] {p.get('title', '')} | {p.get('url', '')}\n{p.get('text', '')}")
    context = "\n\n".join(ctx_lines)
    return (
        f"[사용자 성향]\n리스크={risk}, 예산={budget}원, 목표={goal}\n\n"
        # f"[질문]\n{question}\n\n[CONTEXT]\n{context}\n"
    )

# ===== FastAPI 앱 =====

app = FastAPI(title="Tupick RAG API with SQLite Auth", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=JWT_SECRET_KEY)
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/capstone.mp4")
async def get_video():
    return FileResponse("capstone.mp4", media_type="video/mp4")

@app.on_event("startup")
def startup():
    global _texts, _metas, _index, _bm25
    init_db()
    
    # 기존 문서 로드 (있으면)
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents")
        rows = cursor.fetchall()
        if rows:
            _texts = [dict(r)["text"] for r in rows]
            _metas = [{k: dict(r)[k] for k in ["id", "title", "url", "as_of_date", "source", "category"]} 
                     for r in rows]
            if _texts:
                _bm25 = BM25Okapi([t.split() for t in _texts])
                vecs = encode_texts(_texts)
                _index = build_index(vecs)

# ===== 인증 API =====

def generate_verification_code() -> str:
    """4자리 인증 코드 생성"""
    return ''.join([str(secrets.randbelow(10)) for _ in range(4)])

def send_verification_email(email: str, code: str) -> bool:
    """인증 코드 이메일 발송"""
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        print(f"📧 [개발 모드] 인증 코드: {email} -> {code}")
        return True
    
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = email
        msg['Subject'] = '[Tupick] 이메일 인증 코드'
        
        body = f"""안녕하세요, Tupick입니다.

회원가입을 위한 인증 코드는 다음과 같습니다:

인증 코드: {code}

이 코드는 10분간 유효합니다.

감사합니다.
Tupick 팀"""
        
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        print(f"❌ 이메일 발송 실패: {e}")
        return False

class VerificationRequest(BaseModel):
    email: EmailStr

@app.post("/auth/send-verification")
async def send_verification_code(req: VerificationRequest):
    """이메일 인증 코드 발송"""
    if get_user_by_email(req.email):
        raise HTTPException(400, "이미 등록된 이메일입니다")
    
    code = generate_verification_code()
    
    verification_codes[req.email] = {
        "code": code,
        "expires_at": datetime.now() + timedelta(minutes=10),
        "verified": False
    }
    
    success = send_verification_email(req.email, code)
    
    if success:
        return {
            "message": "인증 코드가 이메일로 발송되었습니다",
            "email": req.email,
            "dev_mode": not bool(SMTP_EMAIL),
            "code": code if not SMTP_EMAIL else None
        }
    else:
        raise HTTPException(500, "이메일 발송에 실패했습니다")

class VerifyCodeRequest(BaseModel):
    email: EmailStr
    code: str

@app.post("/auth/verify-code")
async def verify_code(req: VerifyCodeRequest):
    """인증 코드 확인"""
    if req.email not in verification_codes:
        raise HTTPException(400, "인증 코드가 발송되지 않았습니다")
    
    stored = verification_codes[req.email]
    
    if datetime.now() > stored["expires_at"]:
        del verification_codes[req.email]
        raise HTTPException(400, "인증 코드가 만료되었습니다. 다시 발송해주세요")
    
    if stored["code"] != req.code:
        raise HTTPException(400, "인증 코드가 일치하지 않습니다")
    
    verification_codes[req.email]["verified"] = True
    
    return {
        "message": "이메일 인증이 완료되었습니다",
        "verified": True,
        "email": req.email
    }

@app.post("/auth/register", response_model=Token)
async def register(user_data: UserCreate):
    """회원가입 - 이메일 인증 필요"""
    if get_user_by_email(user_data.email):
        raise HTTPException(400, "이미 등록된 이메일입니다")
    
    # 이메일 인증 확인
    if user_data.email not in verification_codes:
        raise HTTPException(400, "이메일 인증이 필요합니다. 먼저 인증 코드를 발송해주세요")
    
    if not verification_codes[user_data.email].get("verified"):
        raise HTTPException(400, "이메일 인증이 완료되지 않았습니다")
    
    # 인증 성공 후 코드 삭제
    del verification_codes[user_data.email]
    
    is_valid, message = validate_password_strength(user_data.password)
    if not is_valid:
        raise HTTPException(400, message)
    
    user = create_user(user_data)
    access_token = create_access_token({"sub": user.email})
    
    return Token(access_token=access_token, token_type="bearer", user=user)

@app.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    """로그인"""
    check_rate_limit(credentials.email)
    
    user_dict = get_user_by_email(credentials.email)
    if not user_dict or not user_dict.get("hashed_password"):
        raise HTTPException(401, "이메일 또는 비밀번호가 잘못되었습니다")
    
    if not verify_password(credentials.password, user_dict["hashed_password"]):
        raise HTTPException(401, "이메일 또는 비밀번호가 잘못되었습니다")
    
    access_token = create_access_token({"sub": user_dict["email"]})
    user = User(**{k: v for k, v in user_dict.items() if k != "hashed_password"})
    
    return Token(access_token=access_token, token_type="bearer", user=user)

@app.get("/auth/google")
async def google_auth(request: Request):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(400, "Google OAuth not configured")
    redirect_uri = request.url_for('google_auth_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/google/callback")
async def google_auth_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        
        if user_info:
            user_dict = get_user_by_email(user_info['email'])
            if user_dict:
                user = User(**{k: v for k, v in user_dict.items() if k != "hashed_password"})
            else:
                user = create_oauth_user(
                    user_info['email'],
                    user_info['name'],
                    user_info.get('picture')
                )
            
            access_token = create_access_token({"sub": user.email})
            return RedirectResponse(url=f"/?token={access_token}&user={user.email}")
    except Exception as e:
        print(f"Google auth error: {e}")
        return RedirectResponse(url="/login.html?error=auth_failed")

@app.get("/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# ===== 프로필 API =====

@app.get("/users/profile")
async def get_profile(current_user: User = Depends(get_current_user)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (current_user.id,))
        row = cursor.fetchone()
        return dict(row) if row else UserProfile().model_dump()

@app.post("/users/profile")
async def set_profile(profile: UserProfile, current_user: User = Depends(get_current_user)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        UPDATE user_profiles 
        SET risk=?, budget=?, goal=?, interest_field=?, experience_level=?
        WHERE user_id=?
        ''', (profile.risk, profile.budget, profile.goal, 
              profile.interest_field, profile.experience_level, current_user.id))
        conn.commit()
    return {"ok": True, "profile": profile}

@app.post("/chat/save")
async def save_chat_session(chat_request: SaveChatRequest, current_user: User = Depends(get_current_user)):
    """채팅 기록 저장"""
    try:
        chat_storage_key = f"chat_history_{current_user.email}"
        
        # 기존 채팅 기록 불러오기
        existing_chats = redis_client.get(chat_storage_key)
        if existing_chats:
            chat_history = json.loads(existing_chats)
        else:
            chat_history = []
        
        # 새 채팅 세션 추가
        new_chat = {
            "id": str(datetime.now().timestamp()),
            "user_email": current_user.email,
            "timestamp": datetime.now().isoformat(),
            "messages": [msg.dict() for msg in chat_request.messages]
        }
        
        chat_history.append(new_chat)
        
        # 최근 50개 채팅만 저장 (메모리 관리)
        if len(chat_history) > 50:
            chat_history = chat_history[-50:]
        
        # Redis에 저장
        redis_client.set(chat_storage_key, json.dumps(chat_history))
        
        return {"status": "success", "message": "Chat saved successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save chat: {str(e)}")

@app.get("/chat/history")
async def get_chat_history(current_user: User = Depends(get_current_user)):
    """채팅 기록 조회"""
    try:
        chat_storage_key = f"chat_history_{current_user.email}"
        
        existing_chats = redis_client.get(chat_storage_key)
        if existing_chats:
            chat_history = json.loads(existing_chats)
            # 최신 채팅이 먼저 오도록 정렬
            chat_history.sort(key=lambda x: x['timestamp'], reverse=True)
            return chat_history
        else:
            return []
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load chat history: {str(e)}")

@app.delete("/chat/{chat_id}")
async def delete_chat_session(chat_id: str, current_user: User = Depends(get_current_user)):
    """채팅 기록 삭제"""
    try:
        chat_storage_key = f"chat_history_{current_user.email}"
        
        existing_chats = redis_client.get(chat_storage_key)
        if existing_chats:
            chat_history = json.loads(existing_chats)
            # 해당 채팅 삭제
            chat_history = [chat for chat in chat_history if chat['id'] != chat_id]
            
            # Redis에 업데이트
            redis_client.set(chat_storage_key, json.dumps(chat_history))
            
        return {"status": "success", "message": "Chat deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {str(e)}")

# ===== RAG API =====

# ===== JSONL 파일 관리 (백업/호환성용) =====

DOCS_JSONL_PATH = os.path.join(os.path.dirname(DB_PATH), "docs.jsonl")

def load_jsonl_docs() -> List[Dict[str, Any]]:
    """JSONL 파일에서 문서 로드"""
    if not os.path.exists(DOCS_JSONL_PATH):
        return []
    
    docs = []
    try:
        with open(DOCS_JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
    except Exception as e:
        print(f"JSONL 로드 오류: {e}")
    
    return docs

def save_jsonl_docs(docs: List[Dict[str, Any]]):
    """JSONL 파일에 문서 저장 (덮어쓰기)"""
    try:
        with open(DOCS_JSONL_PATH, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"JSONL 저장 오류: {e}")

def clean_jsonl_duplicates(keep_latest: int = 1):
    """
    docs.jsonl에서 중복 URL 정리
    keep_latest: URL당 유지할 최신 문서 개수
    """
    docs = load_jsonl_docs()
    
    if not docs:
        return 0, 0
    
    # URL별로 그룹화
    url_groups = {}
    for doc in docs:
        url = doc.get("url", "")
        if url not in url_groups:
            url_groups[url] = []
        url_groups[url].append(doc)
    
    # 각 URL별로 최신 순으로 정렬 후 필요한 개수만 유지
    cleaned_docs = []
    total_before = len(docs)
    
    for url, doc_list in url_groups.items():
        # as_of_date 기준으로 정렬 (최신순)
        sorted_docs = sorted(
            doc_list, 
            key=lambda x: x.get("as_of_date", "1970-01-01"),
            reverse=True
        )
        
        # 최신 N개만 유지
        cleaned_docs.extend(sorted_docs[:keep_latest])
    
    # 파일에 덮어쓰기
    save_jsonl_docs(cleaned_docs)
    
    total_after = len(cleaned_docs)
    deleted = total_before - total_after
    
    return deleted, total_after

def sync_db_to_jsonl():
    """SQLite DB의 문서를 JSONL에 동기화 (백업)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
        rows = cursor.fetchall()
        
        docs = []
        for row in rows:
            doc_dict = dict(row)
            # created_at을 as_of_date로 변환
            doc_dict['as_of_date'] = doc_dict.get('created_at', now_date())
            docs.append(doc_dict)
        
        save_jsonl_docs(docs)
        return len(docs)

@app.post("/maintenance/cleanup-jsonl")
async def cleanup_jsonl(
    keep_per_url: int = 1,
    current_user: User = Depends(get_current_user)
):
    """
    docs.jsonl 파일 정리
    keep_per_url: URL당 유지할 문서 개수
    """
    deleted, remaining = clean_jsonl_duplicates(keep_latest=keep_per_url)
    
    return {
        "deleted": deleted,
        "remaining": remaining,
        "keep_per_url": keep_per_url,
        "message": f"{deleted}개 문서 삭제, {remaining}개 문서 유지됨"
    }

@app.post("/maintenance/sync-db-to-jsonl")
async def sync_to_jsonl(current_user: User = Depends(get_current_user)):
    """SQLite DB 내용을 JSONL 파일로 백업"""
    count = sync_db_to_jsonl()
    return {
        "synced": count,
        "message": f"{count}개 문서가 JSONL 파일로 백업되었습니다"
    }

@app.delete("/maintenance/delete-jsonl")
async def delete_jsonl_file(current_user: User = Depends(get_current_user)):
    """docs.jsonl 파일 완전 삭제"""
    if os.path.exists(DOCS_JSONL_PATH):
        os.remove(DOCS_JSONL_PATH)
        return {"message": "JSONL 파일이 삭제되었습니다"}
    return {"message": "JSONL 파일이 존재하지 않습니다"}

def clean_old_documents_by_url(url: str, keep_latest: int = 1):
    """
    같은 URL의 오래된 문서 삭제 (DB에서)
    keep_latest: 유지할 최신 문서 개수 (기본 1개)
    """
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 해당 URL의 모든 문서를 날짜순으로 조회
        cursor.execute('''
        SELECT id, created_at FROM documents 
        WHERE url = ? 
        ORDER BY created_at DESC
        ''', (url,))
        
        docs = cursor.fetchall()
        
        if len(docs) <= keep_latest:
            return 0  # 삭제할 문서 없음
        
        # 유지할 문서 제외하고 나머지 삭제
        docs_to_delete = [dict(d)["id"] for d in docs[keep_latest:]]
        
        if docs_to_delete:
            placeholders = ','.join('?' * len(docs_to_delete))
            cursor.execute(f'''
            DELETE FROM documents 
            WHERE id IN ({placeholders})
            ''', docs_to_delete)
            
            conn.commit()
            return len(docs_to_delete)
        
        return 0

def rebuild_index_from_db():
    """데이터베이스에서 전체 인덱스 재구축"""
    global _texts, _metas, _index, _bm25
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
        rows = cursor.fetchall()
        
        if not rows:
            _texts = []
            _metas = []
            _index = None
            _bm25 = None
            return
        
        _texts = [dict(r)["text"] for r in rows]
        _metas = [{k: dict(r)[k] for k in ["id", "title", "url", "as_of_date", "source", "category"]} 
                 for r in rows]
        
        # BM25 재구축
        _bm25 = BM25Okapi([t.split() for t in _texts])
        
        # FAISS 인덱스 재구축
        vecs = encode_texts(_texts)
        _index = build_index(vecs)

@app.post("/ingest")
async def ingest(req: IngestReq, current_user: User = Depends(get_current_user)):
    global _texts, _metas, _index, _bm25
    
    if not req.urls:
        raise HTTPException(400, "URLs가 비어있습니다")
    
    items = []
    urls_processed = []
    deleted_count = 0
    
    for url in req.urls:
        try:
            title, body = fetch_url(url, use_js=req.use_js)
            
            # 같은 URL의 오래된 문서 삭제
            deleted = clean_old_documents_by_url(url, keep_latest=0)
            deleted_count += deleted
            
            for chunk in chunk_text(body):
                doc_id = str(uuid.uuid4())
                items.append({
                    "id": doc_id,
                    "user_id": current_user.id,
                    "source": req.source,
                    "category": "조각투자",
                    "title": title,
                    "section": "본문",
                    "url": url,
                    "as_of_date": now_date(),
                    "text": chunk
                })
            
            urls_processed.append(url)
            
        except Exception as e:
            print(f"URL 처리 실패 {url}: {e}")
            continue
    
    if not items:
        raise HTTPException(400, "크롤링 실패")
    
    # DB에 새 문서 저장
    with get_db() as conn:
        cursor = conn.cursor()
        for item in items:
            cursor.execute('''
            INSERT INTO documents (id, user_id, source, category, title, section, url, as_of_date, text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (item["id"], item["user_id"], item["source"], item["category"],
                  item["title"], item["section"], item["url"], item["as_of_date"], item["text"]))
        conn.commit()
    
    # 전체 인덱스 재구축 (삭제된 문서 반영)
    rebuild_index_from_db()
    
    return {
        "added": len(items), 
        "deleted": deleted_count,
        "urls_processed": urls_processed,
        "total": len(_texts),
        "message": f"{len(items)}개 문서 추가, {deleted_count}개 오래된 문서 삭제됨"
    }


def build_user_prompt(question: str, passages: List[Dict], risk: str, budget: int, goal: str, interest_field: str = "부동산, 예술, 음악", experience_level: str = "초보자") -> str:
    """사용자 질문 + 검색된 문서 + 투자 성향 → LLM 프롬프트"""
    context_lines = []
    for i, p in enumerate(passages, start=1):
        txt = p.get("text", "")
        context_lines.append(f"[{i}] {txt}")
    
    context_str = "\n".join(context_lines) if context_lines else "(관련 문서 없음)"
    
    prompt = f"""[사용자 성향]
리스크 선호: {risk}
투자 예산: {budget:,}원
투자 목표: {goal}
관심 분야: {interest_field}
투자 경험: {experience_level}

[검색된 문서]
{context_str}

[질문]
{question}

위 정보를 바탕으로, 사용자의 투자 성향(리스크, 예산, 목표, 관심 분야, 경험)을 반드시 고려하여 조각투자 관점에서 맞춤형 답변해주세요.
특히 예산, 리스크 수준, 목표, 관심 분야에 맞는 실용적이고 구체적인 정보를 제공해주세요."""
    
    return prompt


@app.post("/query", response_model=QueryRes)
async def query(req: QueryReq, current_user: User = Depends(get_current_user)):
    # 문서가 없어도 답변 가능
    passages = []
    
    if _texts and _index is not None:
        try:
            cand = vec_search(req.question, k=req.k)
            topk = bm25_rerank(req.question, cand, k=req.k)
            
            for i in topk:
                if 0 <= i < len(_texts):
                    passages.append({**_metas[i], "text": _texts[i]})
        except Exception as e:
            print(f"⚠️ 검색 오류 (무시하고 진행): {e}")
    
    # 프로필 가져오기 - 항상 프로필을 적용
    profile = await get_profile(current_user)
    
    # 요청에 명시된 값이 있으면 우선 사용, 없으면 프로필 값 사용
    risk = req.risk if req.risk and req.risk != "중간" else profile.get("risk", "중간")
    budget = req.budget if req.budget and req.budget != 1_000_000 else profile.get("budget", 1_000_000)
    goal = req.goal if req.goal and req.goal != "단기현금흐름" else profile.get("goal", "단기현금흐름")
    interest_field = profile.get("interest_field", "부동산, 예술, 음악")
    experience_level = profile.get("experience_level", "초보자")
    
    # 문서가 있으면 RAG, 없으면 일반 답변
    if passages:
        prompt = build_user_prompt(req.question, passages, risk, budget, goal, interest_field, experience_level)
    else:
        prompt = f"""[사용자 성향]
리스크 선호: {risk}
투자 예산: {budget:,}원
투자 목표: {goal}
관심 분야: {interest_field}
투자 경험: {experience_level}

[질문]
{req.question}

위 사용자의 투자 성향을 반드시 고려하여 조각투자 전문가로서 맞춤형 답변을 제공해주세요. 
특히 사용자의 리스크 수준, 예산, 목표, 관심 분야, 경험 수준에 적합한 구체적인 조언을 포함해주세요."""
    
    answer = call_openai(prompt, LLM_SYSTEM_PROMPT)
    
    # 히스토리 저장
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO chat_history (id, user_id, question, answer)
        VALUES (?, ?, ?, ?)
        ''', (str(uuid.uuid4()), current_user.id, req.question, answer))
        conn.commit()
    
    sources = [{"n": j+1, "title": p["title"], "url": p["url"]} 
               for j, p in enumerate(passages)]
    
    return QueryRes(
        answer=answer,
        backend=f"OpenAI · {OPENAI_MODEL}",
        k=len(passages),
        sources=sources
    )

# ===== 문서 관리 API =====

@app.get("/documents")
async def list_documents(
    current_user: User = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0
):
    """사용자의 문서 목록 조회"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 전체 개수
        cursor.execute('''
        SELECT COUNT(DISTINCT url) FROM documents WHERE user_id = ?
        ''', (current_user.id,))
        total = cursor.fetchone()[0]
        
        # URL별로 그룹화하여 최신 문서만 조회
        cursor.execute('''
        SELECT 
            url,
            title,
            source,
            category,
            MAX(created_at) as latest_created,
            COUNT(*) as chunk_count
        FROM documents 
        WHERE user_id = ?
        GROUP BY url
        ORDER BY latest_created DESC
        LIMIT ? OFFSET ?
        ''', (current_user.id, limit, offset))
        
        docs = [dict(row) for row in cursor.fetchall()]
    
    return {
        "total": total,
        "documents": docs,
        "limit": limit,
        "offset": offset
    }

@app.delete("/documents/url")
async def delete_documents_by_url(
    url: str,
    current_user: User = Depends(get_current_user)
):
    """특정 URL의 모든 문서 삭제"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 삭제할 문서 개수 확인
        cursor.execute('''
        SELECT COUNT(*) FROM documents 
        WHERE url = ? AND user_id = ?
        ''', (url, current_user.id))
        
        count = cursor.fetchone()[0]
        
        if count == 0:
            raise HTTPException(404, "해당 URL의 문서를 찾을 수 없습니다")
        
        # 삭제 실행
        cursor.execute('''
        DELETE FROM documents 
        WHERE url = ? AND user_id = ?
        ''', (url, current_user.id))
        
        conn.commit()
    
    # 인덱스 재구축
    rebuild_index_from_db()
    
    return {
        "deleted": count,
        "url": url,
        "total_remaining": len(_texts)
    }

@app.delete("/documents/all")
async def delete_all_documents(current_user: User = Depends(get_current_user)):
    """사용자의 모든 문서 삭제"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT COUNT(*) FROM documents WHERE user_id = ?
        ''', (current_user.id,))
        count = cursor.fetchone()[0]
        
        cursor.execute('''
        DELETE FROM documents WHERE user_id = ?
        ''', (current_user.id,))
        
        conn.commit()
    
    # 인덱스 재구축
    rebuild_index_from_db()
    
    return {
        "deleted": count,
        "total_remaining": len(_texts)
    }

@app.post("/documents/cleanup")
async def cleanup_old_documents(
    keep_per_url: int = 1,
    current_user: User = Depends(get_current_user)
):
    """
    중복 URL 정리 - URL당 최신 N개만 유지
    keep_per_url: URL당 유지할 문서 개수 (기본 1)
    """
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 사용자의 모든 URL 조회
        cursor.execute('''
        SELECT DISTINCT url FROM documents WHERE user_id = ?
        ''', (current_user.id,))
        
        urls = [row[0] for row in cursor.fetchall()]
        
        total_deleted = 0
        cleaned_urls = []
        
        for url in urls:
            # 각 URL별로 오래된 문서 삭제
            cursor.execute('''
            SELECT id, created_at FROM documents 
            WHERE url = ? AND user_id = ?
            ORDER BY created_at DESC
            ''', (url, current_user.id))
            
            docs = cursor.fetchall()
            
            if len(docs) > keep_per_url:
                # 유지할 문서 제외하고 삭제
                docs_to_delete = [dict(d)["id"] for d in docs[keep_per_url:]]
                
                placeholders = ','.join('?' * len(docs_to_delete))
                cursor.execute(f'''
                DELETE FROM documents WHERE id IN ({placeholders})
                ''', docs_to_delete)
                
                total_deleted += len(docs_to_delete)
                cleaned_urls.append({
                    "url": url,
                    "deleted": len(docs_to_delete),
                    "kept": keep_per_url
                })
        
        conn.commit()
    
    # 인덱스 재구축
    rebuild_index_from_db()
    
    return {
        "total_deleted": total_deleted,
        "cleaned_urls": cleaned_urls,
        "total_remaining": len(_texts)
    }

@app.get("/documents/stats")
async def document_stats(current_user: User = Depends(get_current_user)):
    """문서 통계 정보"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 전체 문서 수
        cursor.execute('''
        SELECT COUNT(*) FROM documents WHERE user_id = ?
        ''', (current_user.id,))
        total_docs = cursor.fetchone()[0]
        
        # 고유 URL 수
        cursor.execute('''
        SELECT COUNT(DISTINCT url) FROM documents WHERE user_id = ?
        ''', (current_user.id,))
        unique_urls = cursor.fetchone()[0]
        
        # 소스별 통계
        cursor.execute('''
        SELECT source, COUNT(*) as count 
        FROM documents 
        WHERE user_id = ?
        GROUP BY source
        ''', (current_user.id,))
        by_source = [{"source": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        # 중복 URL 정보
        cursor.execute('''
        SELECT url, COUNT(*) as count
        FROM documents
        WHERE user_id = ?
        GROUP BY url
        HAVING count > 1
        ORDER BY count DESC
        ''', (current_user.id,))
        duplicates = [{"url": row[0], "count": row[1]} for row in cursor.fetchall()]
        
    return {
        "total_documents": total_docs,
        "unique_urls": unique_urls,
        "average_per_url": round(total_docs / unique_urls, 2) if unique_urls > 0 else 0,
        "by_source": by_source,
        "duplicate_urls": duplicates,
        "duplicate_count": len(duplicates)
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "db": "SQLite",
        "docs": len(_texts),
        "backend": f"OpenAI · {OPENAI_MODEL}"
    }

@app.get("/stats")
def stats():
    return {"docs": len(_texts)}

# ===== HTML 페이지 =====

@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
def read_root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return RedirectResponse(url="/docs")

@app.get("/login.html", response_class=HTMLResponse)
def read_login():
    try:
        with open("login.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return HTMLResponse("<h1>Login Page</h1><a href='/auth/google'>Google Login</a>")
    

@app.get("/register.html", response_class=HTMLResponse)
def read_register():
    try:
        with open("register.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return HTMLResponse("<h1>Register Page</h1><p>register.html 파일을 생성해주세요</p>")

@app.get("/app.html", response_class=HTMLResponse)
def read_app():
    try:
        with open("app.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return {"error": "app.html not found"}
    
@app.get("/features.html", response_class=HTMLResponse)
def read_features():
    try:
        with open("features.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return HTMLResponse("<h1>Features</h1>")

@app.get("/about.html", response_class=HTMLResponse)
def read_about():
    try:
        with open("about.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return HTMLResponse("<h1>About</h1>")

@app.get("/profile.html", response_class=HTMLResponse)
def read_profile():
    try:
        with open("profile.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return HTMLResponse("<h1>Profile</h1><p>profile.html 파일을 생성해주세요</p>")

@app.get("/how-it-works.html", response_class=HTMLResponse)
def read_how_it_works():
    try:
        with open("how-it-works.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return HTMLResponse("<h1>How it works</h1>")
    
@app.get("/profile-history.html", response_class=HTMLResponse)
def read_profile_history():
    try:
        with open("profile-history.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return HTMLResponse("<h1>프로필 기록 페이지</h1><p>profile-history.html 파일을 생성해주세요</p>")   

@app.get("/main.css")
async def read_main_css():
    try:
        with open("main.css", "r", encoding="utf-8") as f:
            css_content = f.read()
            return Response(
                content=css_content, 
                media_type="text/css; charset=utf-8",
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "Content-Type": "text/css; charset=utf-8"
                }
            )
    except Exception as e:
        print(f"❌ CSS 로드 실패: {e}")
        return Response(
            content="/* main.css not found */", 
            media_type="text/css; charset=utf-8"
        )

@app.get("/app.css")
async def read_app_css():
    try:
        with open("app.css", "r", encoding="utf-8") as f:
            css_content = f.read()
            return Response(
                content=css_content, 
                media_type="text/css; charset=utf-8",
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "Content-Type": "text/css; charset=utf-8"
                }
            )
    except Exception as e:
        print(f"❌ CSS 로드 실패: {e}")
        return Response(
            content="/* app.css not found */", 
            media_type="text/css; charset=utf-8"
        )

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_v6:app", host="0.0.0.0", port=8000, reload=True)