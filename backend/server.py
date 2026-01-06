from fastapi import FastAPI, APIRouter, HTTPException, Depends, status, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import FormData
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import bcrypt
import jwt
import requests
import shutil
import random
import string
import urllib.parse
import asyncio
import aiohttp

# Set Starlette max request size to 100MB BEFORE any other imports
os.environ["STARLETTE_MAX_REQUEST_SIZE"] = str(100 * 1024 * 1024)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Config
JWT_SECRET = os.environ.get('JWT_SECRET', 'your-secret-key')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

# TMDB Config
TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

# Uploads Config
UPLOADS_DIR = ROOT_DIR / "uploads" / "promotions"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB - must match server proxy limits

app = FastAPI()

# Increase max request size for file uploads
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class LimitUploadSize(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        if request.method == 'POST':
            if 'content-length' in request.headers:
                content_length = int(request.headers['content-length'])
                if content_length > self.max_upload_size:
                    return JSONResponse(
                        status_code=413,
                        content={'detail': f'File too large. Max size: {self.max_upload_size / (1024*1024):.0f}MB'}
                    )
        return await call_next(request)

# Add middleware with limit (this is just for validation, the real limit is set via env var above)
app.add_middleware(LimitUploadSize, max_upload_size=100 * 1024 * 1024)  # 100MB limit
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

# ============================================================================
# OPTIMIZACIÓN: Sistema de Caché para TMDB (reduce llamadas redundantes)
# ============================================================================
class TMDBCache:
    """Caché en memoria para respuestas de TMDB con TTL de 24 horas"""
    def __init__(self, ttl_hours=24):
        self.cache = {}
        self.ttl_hours = ttl_hours
    
    def get(self, key: str):
        """Obtener del caché si es válido"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now(timezone.utc) - timestamp < timedelta(hours=self.ttl_hours):
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, data):
        """Guardar en caché"""
        self.cache[key] = (data, datetime.now(timezone.utc))
    
    def clear_expired(self):
        """Limpiar entradas expiradas"""
        now = datetime.now(timezone.utc)
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= timedelta(hours=self.ttl_hours)
        ]
        for key in expired_keys:
            del self.cache[key]

tmdb_cache = TMDBCache()

# Mount static files for uploaded images
app.mount("/uploads", StaticFiles(directory=str(ROOT_DIR / "uploads")), name="uploads")

# CORS Config is handled by Nginx reverse proxy in production
# For local development, uncomment the section below:
ALLOWED_ORIGINS = [
    'http://localhost:3000',
    'http://localhost:5000',
    'http://localhost',
    'http://127.0.0.1:3000',
    'http://127.0.0.1:5000',
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ========== MODELS ==========

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    password_hash: str
    role: str = "user"  # user or admin
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    role: str
    created_at: datetime

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class Content(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tmdb_id: int
    content_type: str  # movie, tv, novela
    title: str
    original_title: Optional[str] = None
    overview: Optional[str] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    genres: List[str] = []
    countries: List[str] = []
    release_date: Optional[str] = None
    vote_average: Optional[float] = None
    seasons: Optional[List[dict]] = None  # For TV shows
    trailer_key: Optional[str] = None  # YouTube trailer key
    trailer_url: Optional[str] = None  # Full trailer URL
    added_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ContentCreate(BaseModel):
    tmdb_id: int
    content_type: str  # movie, tv, novela

class ContentSearchQuery(BaseModel):
    query: str
    content_type: str  # movie, tv, novela

class Review(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_id: str
    user_id: str
    user_email: str
    rating: int  # 1-5 stars
    review_text: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ReviewCreate(BaseModel):
    content_id: str
    rating: int
    review_text: str

class ReviewResponse(BaseModel):
    id: str
    content_id: str
    user_email: str
    rating: int
    review_text: str
    created_at: datetime
    content_title: Optional[str] = None  # Título del contenido

class Watchlist(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    content_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PasswordChange(BaseModel):
    old_password: str
    new_password: str

class AdminPasswordReset(BaseModel):
    new_password: str

# ========== PROMOTION MODELS ==========

class FeatureItem(BaseModel):
    icon: str
    text: str

class DetailItem(BaseModel):
    icon: str
    title: str
    description: str

class PromotionBase(BaseModel):
    title: str
    subtitle: str
    description: str
    buttonText: str
    bannerGradient1: str
    bannerGradient2: str
    bannerGradient3: str
    pageTitle: str
    pageContent: str
    prizeTitle: str
    prizeDescription: str
    prizeImage: Optional[str] = None
    features: List[FeatureItem]
    details: List[DetailItem]
    howToParticipate: List[str]
    termsText: str
    showFrequencyMinutes: int = 2
    isActive: bool = True

class PromotionCreate(PromotionBase):
    pass

class PromotionUpdate(BaseModel):
    title: Optional[str] = None
    subtitle: Optional[str] = None
    description: Optional[str] = None
    buttonText: Optional[str] = None
    bannerGradient1: Optional[str] = None
    bannerGradient2: Optional[str] = None
    bannerGradient3: Optional[str] = None
    pageTitle: Optional[str] = None
    pageContent: Optional[str] = None
    prizeTitle: Optional[str] = None
    prizeDescription: Optional[str] = None
    prizeImage: Optional[str] = None
    features: Optional[List[FeatureItem]] = None
    details: Optional[List[DetailItem]] = None
    howToParticipate: Optional[List[str]] = None
    termsText: Optional[str] = None
    showFrequencyMinutes: Optional[int] = None
    isActive: Optional[bool] = None

# ========== COUPON MODELS ==========

class Coupon(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    code: str  # Código único del cupón
    user_id: str  # Usuario al que va dirigido
    description: str  # Descripción del cupón
    discount_type: str  # "percentage", "currency", "content"
    discount_value: float  # Valor del descuento (% o CUP)
    content_id: Optional[str] = None  # ID del contenido para descuento tipo "content"
    expiry_date: datetime  # Fecha de vencimiento
    is_used: bool = False  # Si ha sido usado
    used_at: Optional[datetime] = None  # Cuándo se usó
    viewed_by_user: bool = False  # Si el usuario ha visto el cupón
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str  # ID del admin que lo creó
    scheduled_deletion_at: Optional[datetime] = None  # Para TTL index - usado/vencido, se elimina en 10 horas

class CouponCreate(BaseModel):
    user_id: Optional[str] = None  # None si es aleatorio
    is_random: bool = False  # True para asignar a usuario aleatorio
    description: str
    discount_type: str  # "percentage", "currency", "content"
    discount_value: float  # Valor del descuento
    content_id: Optional[str] = None  # Para tipo "content"
    expiry_days: int  # Días hasta vencimiento

class CouponResponse(BaseModel):
    id: str
    code: str
    description: str
    discount_type: str
    discount_value: float
    content_id: Optional[str] = None
    expiry_date: datetime
    is_used: bool
    used_at: Optional[datetime] = None
    created_at: datetime

class CouponWhatsAppRequest(BaseModel):
    worker_phone: str  # Teléfono del trabajador (ej: 525252425434)

# ========== PAGINATION MODELS ==========

class PaginationParams(BaseModel):
    page: int = 1
    limit: int = 20
    
    @property
    def skip(self) -> int:
        return (self.page - 1) * self.limit

class PaginatedResponse(BaseModel):
    data: List[dict]
    total: int
    page: int
    limit: int
    total_pages: int
    has_next: bool
    has_prev: bool

# ========== AUTH HELPERS ==========

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_token(user_id: str, email: str, role: str) -> str:
    payload = {
        'user_id': user_id,
        'email': email,
        'role': role,
        'exp': datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_admin_user(current_user: dict = Depends(get_current_user)):
    if current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# ========== TMDB INTEGRATION ==========

def search_tmdb_content(query: str, content_type: str):
    """Search TMDB by name"""
    if content_type == "movie":
        url = f"{TMDB_BASE_URL}/search/movie"
    else:  # tv or novela
        url = f"{TMDB_BASE_URL}/search/tv"
    
    params = {'api_key': TMDB_API_KEY, 'language': 'es-ES', 'query': query}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('results', [])
    return []

def fetch_movie_details(tmdb_id: int):
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
    params = {'api_key': TMDB_API_KEY, 'language': 'es-ES'}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def fetch_tv_details(tmdb_id: int):
    url = f"{TMDB_BASE_URL}/tv/{tmdb_id}"
    params = {'api_key': TMDB_API_KEY, 'language': 'es-ES'}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        # Fetch season details
        seasons_data = []
        if 'seasons' in data:
            for season in data['seasons']:
                season_number = season['season_number']
                season_url = f"{TMDB_BASE_URL}/tv/{tmdb_id}/season/{season_number}"
                season_response = requests.get(season_url, params=params)
                if season_response.status_code == 200:
                    season_details = season_response.json()
                    seasons_data.append({
                        'season_number': season_number,
                        'name': season_details.get('name', f'Temporada {season_number}'),
                        'episode_count': len(season_details.get('episodes', [])),
                        'episodes': [{
                            'episode_number': ep['episode_number'],
                            'name': ep['name'],
                            'overview': ep.get('overview', '')
                        } for ep in season_details.get('episodes', [])]
                    })
        data['seasons_detailed'] = seasons_data
        return data
    return None

def get_trailer_url(tmdb_id: int, content_type: str):
    """
    Extrae el trailer de YouTube de TMDB
    Prioriza: trailer en español, después cualquier trailer
    """
    try:
        if content_type == "movie":
            url = f"{TMDB_BASE_URL}/movie/{tmdb_id}/videos"
        else:  # tv or novela
            url = f"{TMDB_BASE_URL}/tv/{tmdb_id}/videos"
        
        params = {'api_key': TMDB_API_KEY, 'language': 'es-ES'}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            videos = data.get('results', [])
            
            # Buscar primero trailer en español
            spanish_trailer = None
            english_trailer = None
            any_trailer = None
            
            for video in videos:
                if video.get('site') == 'YouTube' and video.get('type') in ['Trailer', 'Teaser']:
                    # Preferencia: Trailer en español
                    if video.get('iso_639_1') == 'es':
                        spanish_trailer = video.get('key')
                        break
                    # Segunda opción: Trailer en inglés
                    elif video.get('iso_639_1') == 'en' and not english_trailer:
                        english_trailer = video.get('key')
                    # Última opción: cualquier trailer
                    elif not any_trailer:
                        any_trailer = video.get('key')
            
            # Usar el mejor disponible
            trailer_key = spanish_trailer or english_trailer or any_trailer
            if trailer_key:
                return trailer_key
        
        return None
    except Exception as e:
        logger.error(f"Error fetching trailer: {e}")
        return None

# ============================================================================
# OPTIMIZACIÓN: Funciones Asincrónicas para TMDB (llamadas paralelas)
# ============================================================================

async def fetch_movie_details_async(tmdb_id: int, session: Optional[aiohttp.ClientSession] = None):
    """Versión asincrónica de fetch_movie_details con caché"""
    cache_key = f"movie_{tmdb_id}"
    
    # Verificar caché primero
    cached = tmdb_cache.get(cache_key)
    if cached:
        return cached
    
    try:
        url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
        params = {'api_key': TMDB_API_KEY, 'language': 'es-ES'}
        
        if session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    tmdb_cache.set(cache_key, data)
                    return data
        else:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                tmdb_cache.set(cache_key, data)
                return data
    except asyncio.TimeoutError:
        logger.warning(f"Timeout fetching movie {tmdb_id}")
    except Exception as e:
        logger.error(f"Error fetching movie details: {e}")
    
    return None

async def fetch_tv_details_async(tmdb_id: int, session: Optional[aiohttp.ClientSession] = None):
    """Versión asincrónica de fetch_tv_details con caché"""
    cache_key = f"tv_{tmdb_id}"
    
    # Verificar caché primero
    cached = tmdb_cache.get(cache_key)
    if cached:
        return cached
    
    try:
        url = f"{TMDB_BASE_URL}/tv/{tmdb_id}"
        params = {'api_key': TMDB_API_KEY, 'language': 'es-ES'}
        
        if session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Fetch season details en paralelo (sin esperar series) 
                    if 'seasons' in data and len(data['seasons']) > 0:
                        # Para admin, mantener detalles simples (no traer todos los episodios)
                        data['seasons_detailed'] = [{
                            'season_number': s['season_number'],
                            'name': s.get('name', f"Temporada {s['season_number']}")
                        } for s in data['seasons']]
                    tmdb_cache.set(cache_key, data)
                    return data
        else:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'seasons' in data:
                    data['seasons_detailed'] = [{
                        'season_number': s['season_number'],
                        'name': s.get('name', f"Temporada {s['season_number']}")
                    } for s in data['seasons']]
                tmdb_cache.set(cache_key, data)
                return data
    except asyncio.TimeoutError:
        logger.warning(f"Timeout fetching TV {tmdb_id}")
    except Exception as e:
        logger.error(f"Error fetching TV details: {e}")
    
    return None

async def fetch_tmdb_batch_async(items_with_type: List[tuple]) -> List[dict]:
    """
    Fetch múltiples items de TMDB en paralelo
    items_with_type: Lista de (tmdb_id, content_type) tuples
    
    OPTIMIZACIÓN CRÍTICA: Reduce 10 llamadas secuenciales a 1-2 segundos en paralelo
    """
    tasks = []
    
    for tmdb_id, content_type in items_with_type:
        if content_type == 'movie':
            tasks.append(fetch_movie_details_async(tmdb_id))
        else:
            tasks.append(fetch_tv_details_async(tmdb_id))
    
    # Ejecutar todas las tareas en paralelo
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convertir excepciones a None
    return [None if isinstance(r, Exception) else r for r in results]

# ========== AUTH ROUTES ==========

@api_router.post("/auth/register", response_model=TokenResponse)
async def register(user_data: UserCreate):
    # Check if user exists
    existing_user = await db.users.find_one({"email": user_data.email}, {"_id": 0})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    try:
        # Create user
        user = User(
            email=user_data.email,
            password_hash=hash_password(user_data.password),
            role="user"
        )
        
        doc = user.model_dump()
        doc['created_at'] = doc['created_at'].isoformat()
        await db.users.insert_one(doc)
        
        # Create token
        token = create_token(user.id, user.email, user.role)
        
        return TokenResponse(
            access_token=token,
            user=UserResponse(
                id=user.id,
                email=user.email,
                role=user.role,
                created_at=user.created_at
            )
        )
    except Exception as e:
        # Handle MongoDB duplicate key error
        if "duplicate key" in str(e).lower() or "e11000" in str(e):
            raise HTTPException(status_code=400, detail="Email already registered")
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Error creating user")

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    user_doc = await db.users.find_one({"email": credentials.email}, {"_id": 0})
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not verify_password(credentials.password, user_doc['password_hash']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(user_doc['id'], user_doc['email'], user_doc['role'])
    
    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=user_doc['id'],
            email=user_doc['email'],
            role=user_doc['role'],
            created_at=datetime.fromisoformat(user_doc['created_at'])
        )
    )

@api_router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    user_doc = await db.users.find_one({"id": current_user['user_id']}, {"_id": 0})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        id=user_doc['id'],
        email=user_doc['email'],
        role=user_doc['role'],
        created_at=datetime.fromisoformat(user_doc['created_at'])
    )

# ========== CONTENT ROUTES ==========

@api_router.post("/content/search")
async def search_content(search_query: ContentSearchQuery):
    """Search TMDB content by name"""
    results = search_tmdb_content(search_query.query, search_query.content_type)
    return {"results": results}

@api_router.post("/content", response_model=Content)
async def add_content(content_data: ContentCreate, current_user: dict = Depends(get_admin_user)):
    # Check if content already exists
    existing = await db.content.find_one({
        "tmdb_id": content_data.tmdb_id,
        "content_type": content_data.content_type
    }, {"_id": 0})
    if existing:
        raise HTTPException(status_code=400, detail="Content already exists")
    
    # Fetch from TMDB
    if content_data.content_type == "movie":
        tmdb_data = fetch_movie_details(content_data.tmdb_id)
    else:  # tv or novela
        tmdb_data = fetch_tv_details(content_data.tmdb_id)
    
    if not tmdb_data:
        raise HTTPException(status_code=404, detail="Content not found in TMDB")
    
    # Obtener el trailer
    trailer_key = get_trailer_url(content_data.tmdb_id, content_data.content_type)
    trailer_url = f"https://www.youtube.com/embed/{trailer_key}" if trailer_key else None
    
    # Create content object
    content = Content(
        tmdb_id=content_data.tmdb_id,
        content_type=content_data.content_type,
        title=tmdb_data.get('title') or tmdb_data.get('name', ''),
        original_title=tmdb_data.get('original_title') or tmdb_data.get('original_name'),
        overview=tmdb_data.get('overview'),
        poster_path=tmdb_data.get('poster_path'),
        backdrop_path=tmdb_data.get('backdrop_path'),
        genres=[g['name'] for g in tmdb_data.get('genres', [])],
        countries=[c['name'] for c in tmdb_data.get('production_countries', [])],
        release_date=tmdb_data.get('release_date') or tmdb_data.get('first_air_date'),
        vote_average=tmdb_data.get('vote_average'),
        seasons=tmdb_data.get('seasons_detailed') if content_data.content_type in ['tv', 'novela'] else None,
        trailer_key=trailer_key,
        trailer_url=trailer_url,
        added_by=current_user['user_id']
    )
    
    doc = content.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.content.insert_one(doc)
    
    return content

@api_router.get("/content")
async def get_all_content(page: int = 1, limit: int = 50):
    """Get paginated content"""
    try:
        # Validar parámetros
        if page < 1:
            page = 1
        if limit < 1:
            limit = 50
        elif limit > 50000:
            limit = 50000
            
        skip = (page - 1) * limit
        
        # Contar total
        total = await db.content.count_documents({})
        
        # Obtener contenido paginado ordenado por más reciente primero
        # to_list(None) retorna todos los items del cursor (respetando el .limit())
        content_list = await db.content.find({}, {"_id": 0}).sort("created_at", -1).skip(skip).limit(limit).to_list(None)
        
        # Convertir strings a datetime
        for item in content_list:
            if isinstance(item['created_at'], str):
                item['created_at'] = datetime.fromisoformat(item['created_at'])
        
        total_pages = (total + limit - 1) // limit
        
        return {
            "data": content_list,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
        # Si no tiene trailer_url, intentar obtenerlo
        if not item.get('trailer_url'):
            trailer_key = get_trailer_url(item['tmdb_id'], item['content_type'])
            if trailer_key:
                trailer_url = f"https://www.youtube.com/embed/{trailer_key}"
                item['trailer_key'] = trailer_key
                item['trailer_url'] = trailer_url
                # Actualizar la BD para no tener que hacerlo cada vez
                await db.content.update_one(
                    {"id": item['id']},
                    {"$set": {"trailer_key": trailer_key, "trailer_url": trailer_url}}
                )
    
    return content_list

@api_router.get("/content/{content_id}", response_model=Content)
async def get_content_by_id(content_id: str):
    content = await db.content.find_one({"id": content_id}, {"_id": 0})
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    
    if isinstance(content['created_at'], str):
        content['created_at'] = datetime.fromisoformat(content['created_at'])
    
    # Si no tiene trailer_url, intentar obtenerlo
    if not content.get('trailer_url'):
        trailer_key = get_trailer_url(content['tmdb_id'], content['content_type'])
        if trailer_key:
            trailer_url = f"https://www.youtube.com/embed/{trailer_key}"
            content['trailer_key'] = trailer_key
            content['trailer_url'] = trailer_url
            # Actualizar la BD para no tener que hacerlo cada vez
            await db.content.update_one(
                {"id": content_id},
                {"$set": {"trailer_key": trailer_key, "trailer_url": trailer_url}}
            )
    
    return content


# ========== AUTOCOMPLETE ROUTES ==========
@api_router.get("/autocomplete/genres")
async def autocomplete_genres(q: str = "", limit: int = 10):
    """Return distinct genres matching prefix q (case-insensitive)"""
    if not q:
        # return top genres by frequency
        pipeline = [
            {"$unwind": "$genres"},
            {"$group": {"_id": "$genres", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ]
        res = await db.content.aggregate(pipeline).to_list(None)
        return [r['_id'] for r in res]

    regex = {"$regex": f"^{q}", "$options": "i"}
    pipeline = [
        {"$unwind": "$genres"},
        {"$match": {"genres": regex}},
        {"$group": {"_id": "$genres", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit}
    ]
    res = await db.content.aggregate(pipeline).to_list(None)
    return [r['_id'] for r in res]


@api_router.get("/autocomplete/actors")
async def autocomplete_actors(q: str = "", limit: int = 10):
    """Return distinct actors matching prefix q (case-insensitive). Assumes content documents may have 'actors' array or string."""
    if not q:
        pipeline = [
            {"$project": {"actors": 1}},
            {"$unwind": {"path": "$actors", "preserveNullAndEmptyArrays": False}},
            {"$group": {"_id": "$actors", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ]
        res = await db.content.aggregate(pipeline).to_list(None)
        return [r['_id'] for r in res]

    regex = {"$regex": f"^{q}", "$options": "i"}
    pipeline = [
        {"$project": {"actors": 1}},
        {"$unwind": {"path": "$actors", "preserveNullAndEmptyArrays": False}},
        {"$match": {"actors": regex}},
        {"$group": {"_id": "$actors", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit}
    ]
    res = await db.content.aggregate(pipeline).to_list(None)
    return [r['_id'] for r in res]


# ========== FILTER LISTS (genres, years) ==========
@api_router.get("/filters/genres")
async def get_filter_genres():
    """Return all distinct genres present in content, alphabetically sorted"""
    pipeline = [
        {"$unwind": "$genres"},
        {"$group": {"_id": "$genres"}},
        {"$sort": {"_id": 1}}
    ]
    res = await db.content.aggregate(pipeline).to_list(1000)
    return [r['_id'] for r in res if r['_id']]


@api_router.get("/filters/years")
async def get_filter_years():
    """Return distinct release years (YYYY) present in content, sorted desc"""
    pipeline = [
        {"$match": {"release_date": {"$exists": True, "$ne": ""}}},
        {"$project": {"year": {"$substrBytes": ["$release_date", 0, 4]}}},
        {"$group": {"_id": "$year"}},
        {"$match": {"_id": {"$ne": None, "$ne": ""}}},
        {"$sort": {"_id": -1}}
    ]
    res = await db.content.aggregate(pipeline).to_list(1000)
    # Filter only numeric years
    years = [r['_id'] for r in res if isinstance(r['_id'], str) and r['_id'].isdigit()]
    return years

@api_router.delete("/content/{content_id}")
async def delete_content(content_id: str, current_user: dict = Depends(get_admin_user)):
    result = await db.content.delete_one({"id": content_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Content not found")
    # Also delete associated reviews
    await db.reviews.delete_many({"content_id": content_id})
    return {"message": "Content deleted successfully"}

# ========== REVIEW ROUTES ==========

@api_router.post("/reviews", response_model=Review)
async def create_review(review_data: ReviewCreate, current_user: dict = Depends(get_current_user)):
    # Check if content exists
    content = await db.content.find_one({"id": review_data.content_id}, {"_id": 0})
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    
    # Check if user already reviewed
    existing = await db.reviews.find_one({
        "content_id": review_data.content_id,
        "user_id": current_user['user_id']
    }, {"_id": 0})
    
    if existing:
        raise HTTPException(status_code=400, detail="You already reviewed this content")
    
    review = Review(
        content_id=review_data.content_id,
        user_id=current_user['user_id'],
        user_email=current_user['email'],
        rating=review_data.rating,
        review_text=review_data.review_text
    )
    
    doc = review.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.reviews.insert_one(doc)
    
    return review

@api_router.get("/reviews/content/{content_id}")
async def get_content_reviews(content_id: str, page: int = 1, limit: int = 20):
    """Get paginated reviews for a content item"""
    try:
        # Validar parámetros
        if page < 1:
            page = 1
        if limit < 1:
            limit = 20
        elif limit > 50000:
            limit = 50000
            
        skip = (page - 1) * limit
        
        # Contar total
        total = await db.reviews.count_documents({"content_id": content_id})
        
        # Obtener reviews paginadas
        reviews = await db.reviews.find({"content_id": content_id}, {"_id": 0}).skip(skip).limit(limit).to_list(None)
        
        for review in reviews:
            if isinstance(review['created_at'], str):
                review['created_at'] = datetime.fromisoformat(review['created_at'])
        
        total_pages = (total + limit - 1) // limit
        
        return {
            "data": reviews,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/reviews/{review_id}")
async def delete_review(review_id: str, current_user: dict = Depends(get_current_user)):
    review = await db.reviews.find_one({"id": review_id}, {"_id": 0})
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    # Admin can delete any review, users can only delete their own
    if current_user['role'] != 'admin' and review['user_id'] != current_user['user_id']:
        raise HTTPException(status_code=403, detail="Not authorized to delete this review")
    
    await db.reviews.delete_one({"id": review_id})
    return {"message": "Review deleted successfully"}

# ========== ADMIN ROUTES ==========

@api_router.get("/admin/users")
async def get_all_users(current_user: dict = Depends(get_admin_user), page: int = 1, limit: int = 20):
    """Get paginated list of users"""
    try:
        # Validar parámetros
        if page < 1:
            page = 1
        if limit < 1:
            limit = 20
        elif limit > 1000:
            limit = 1000
            
        skip = (page - 1) * limit
        
        # Contar total
        total = await db.users.count_documents({})
        
        # Obtener usuarios paginados
        users = await db.users.find({}, {"_id": 0, "password_hash": 0}).skip(skip).limit(limit).to_list(None)
        
        for user in users:
            if isinstance(user['created_at'], str):
                user['created_at'] = datetime.fromisoformat(user['created_at'])
        
        total_pages = (total + limit - 1) // limit
        
        return {
            "data": users,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/admin/users/{user_id}", response_model=UserResponse)
async def get_user_by_id(user_id: str, current_user: dict = Depends(get_admin_user)):
    """Get a specific user by ID (admin only)"""
    user_doc = await db.users.find_one({"id": user_id}, {"_id": 0, "password_hash": 0})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    if isinstance(user_doc['created_at'], str):
        user_doc['created_at'] = datetime.fromisoformat(user_doc['created_at'])
    return user_doc

@api_router.get("/admin/users/{user_id}/watchlist")
async def get_user_watchlist(
    user_id: str, 
    current_user: dict = Depends(get_admin_user),
    page: int = 1,
    limit: int = 20
):
    """
    Get watchlist of a specific user (admin only)
    OPTIMIZADO: Llamadas TMDB paralelas + paginación
    
    Parámetros:
    - page: Número de página (default: 1)
    - limit: Items por página (default: 20, max: 100)
    """
    try:
        # Validar parámetros
        if page < 1:
            page = 1
        if limit < 1:
            limit = 20
        elif limit > 100:
            limit = 100
        
        # Verify user exists
        user = await db.users.find_one({"id": user_id}, {"_id": 0, "id": 1})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Contar total de items en watchlist
        total = await db.watchlist.count_documents({"user_id": user_id})
        
        # Get watchlist items con paginación
        skip = (page - 1) * limit
        watchlist_items = await db.watchlist.find(
            {"user_id": user_id}, 
            {"_id": 0, "content_id": 1, "added_at": 1}  # OPTIMIZACIÓN: Solo campos necesarios
        ).sort("added_at", -1).skip(skip).limit(limit).to_list(limit)
        
        if not watchlist_items:
            return {
                "data": [],
                "total": total,
                "page": page,
                "limit": limit,
                "total_pages": 0,
                "has_next": False,
                "has_prev": False
            }
        
        # OPTIMIZACIÓN: Obtener contenido en paralelo (no secuencial)
        content_ids = [item['content_id'] for item in watchlist_items]
        
        # Buscar todos los contenidos en paralelo
        content_docs = await asyncio.gather(*[
            db.content.find_one(
                {"id": cid},
                {"_id": 0, "content_type": 1, "tmdb_id": 1, "title": 1, "poster_path": 1, "overview": 1}
            )
            for cid in content_ids
        ])
        
        # OPTIMIZACIÓN CRÍTICA: Hacer calls a TMDB en paralelo (no secuencial)
        tmdb_requests = []
        for content in content_docs:
            if content:
                tmdb_id = content.get('tmdb_id')
                content_type = content.get('content_type', 'movie')
                if tmdb_id:
                    tmdb_requests.append((tmdb_id, content_type))
        
        # Ejecutar todas las calls TMDB en paralelo
        tmdb_results = {}
        if tmdb_requests:
            tmdb_data_list = await fetch_tmdb_batch_async(tmdb_requests)
            # Mapear resultados a tmdb_id
            for (tmdb_id, _), tmdb_data in zip(tmdb_requests, tmdb_data_list):
                if tmdb_data:
                    tmdb_results[tmdb_id] = tmdb_data
        
        # Construir respuesta
        content_with_details = []
        for item, content in zip(watchlist_items, content_docs):
            if content:
                tmdb_id = content.get('tmdb_id')
                tmdb_data = tmdb_results.get(tmdb_id)
                
                # Usar TMDB si disponible, si no usar DB
                if tmdb_data:
                    poster_path = tmdb_data.get('poster_path') or content.get('poster_path')
                    overview = tmdb_data.get('overview') or content.get('overview')
                    title = tmdb_data.get('title') or tmdb_data.get('name') or content.get('title')
                else:
                    poster_path = content.get('poster_path')
                    overview = content.get('overview')
                    title = content.get('title')
                
                content_with_details.append({
                    "content_id": item['content_id'],
                    "title": title,
                    "content_type": content.get('content_type', 'N/A'),
                    "poster_path": poster_path,
                    "overview": overview,
                    "added_at": item.get('added_at', 'N/A')
                })
        
        total_pages = (total + limit - 1) // limit
        
        return {
            "data": content_with_details,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    
    except Exception as e:
        logger.error(f"Error fetching watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.patch("/admin/users/{user_id}/reset-password")
async def admin_reset_password(user_id: str, body: AdminPasswordReset, current_user: dict = Depends(get_admin_user)):
    """Reset a user's password (admin only)"""
    # Verify user exists
    user = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Hash new password
    password_hash = hash_password(body.new_password)
    
    # Update password
    result = await db.users.update_one(
        {"id": user_id},
        {"$set": {"password_hash": password_hash}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=400, detail="Failed to reset password")
    
    return {"message": "Password reset successfully"}

@api_router.patch("/admin/users/{user_id}/role")
async def update_user_role(user_id: str, role: str, current_user: dict = Depends(get_admin_user)):
    if role not in ['user', 'admin']:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    result = await db.users.update_one(
        {"id": user_id},
        {"$set": {"role": role}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "Role updated successfully"}

@api_router.delete("/admin/users/{user_id}")
async def delete_user(user_id: str, current_user: dict = Depends(get_admin_user)):
    result = await db.users.delete_one({"id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    # Delete user's reviews
    await db.reviews.delete_many({"user_id": user_id})
    return {"message": "User deleted successfully"}

@api_router.get("/admin/reviews")
async def get_all_reviews(current_user: dict = Depends(get_admin_user), page: int = 1, limit: int = 20):
    """Get paginated reviews (admin only)"""
    try:
        # Validar parámetros
        if page < 1:
            page = 1
        if limit < 1:
            limit = 20
        elif limit > 1000:
            limit = 1000
            
        skip = (page - 1) * limit
        
        # Contar total
        total = await db.reviews.count_documents({})
        
        # Obtener reviews paginadas ordenadas por más reciente primero
        reviews = await db.reviews.find({}, {"_id": 0}).sort("created_at", -1).skip(skip).limit(limit).to_list(None)
        
        # Enriquecer las reseñas con el título del contenido
        for review in reviews:
            if isinstance(review['created_at'], str):
                review['created_at'] = datetime.fromisoformat(review['created_at'])
            
            # Buscar el contenido para obtener el título
            content = await db.content.find_one({"id": review['content_id']}, {"_id": 0, "title": 1})
            if content:
                review['content_title'] = content.get('title', 'Contenido no disponible')
            else:
                review['content_title'] = 'Contenido eliminado'
        
        total_pages = (total + limit - 1) // limit
        
        return {
            "data": reviews,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/admin/reviews/{review_id}")
async def delete_review(review_id: str, current_user: dict = Depends(get_admin_user)):
    review = await db.reviews.find_one({"id": review_id}, {"_id": 0})
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    await db.reviews.delete_one({"id": review_id})
    return {"message": "Review deleted successfully"}

@api_router.post("/admin/content/search-existing")
async def search_existing_content(
    search_data: dict,
    current_user: dict = Depends(get_admin_user)
):
    """Search for existing content in database by title"""
    try:
        query = search_data.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Buscar contenido por título (case-insensitive)
        # Usar regex para búsqueda flexible
        import re
        regex = re.compile(query, re.IGNORECASE)
        
        results = await db.content.find(
            {"title": regex},
            {"_id": 0}
        ).sort("created_at", -1).to_list(None)
        
        # Convertir strings a datetime
        for item in results:
            if isinstance(item['created_at'], str):
                item['created_at'] = datetime.fromisoformat(item['created_at'])
        
        return {
            "results": results,
            "total": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== COUPON ROUTES ==========

@api_router.post("/admin/coupons")
async def create_coupon(
    coupon_data: CouponCreate,
    current_user: dict = Depends(get_admin_user)
):
    """Create a new coupon (admin only)"""
    try:
        target_user_id = coupon_data.user_id
        
        # Si es aleatorio, seleccionar usuario aleatorio (excluyendo admins)
        if coupon_data.is_random:
            users = await db.users.find({"role": "user"}, {"_id": 0, "id": 1}).to_list(None)
            if not users:
                raise HTTPException(status_code=400, detail="No regular users available")
            target_user_id = users[len(users) // 2]["id"]  # Seleccionar usuario aleatorio
            import random
            target_user_id = random.choice(users)["id"]
        
        if not target_user_id:
            raise HTTPException(status_code=400, detail="User ID or is_random must be provided")
        
        # Verificar que el usuario existe y no es admin
        user = await db.users.find_one({"id": target_user_id}, {"_id": 0})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.get("role") == "admin":
            raise HTTPException(status_code=400, detail="Cannot create coupons for admins")
        
        # Generar código único
        import random
        import string
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        
        # Verificar que el código sea único
        while await db.coupons.find_one({"code": code}):
            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        
        coupon = Coupon(
            code=code,
            user_id=target_user_id,
            description=coupon_data.description,
            discount_type=coupon_data.discount_type,
            discount_value=coupon_data.discount_value,
            content_id=coupon_data.content_id,
            expiry_date=datetime.now(timezone.utc) + timedelta(days=coupon_data.expiry_days),
            created_by=current_user.get("user_id"),
            viewed_by_user=False
        )
        
        doc = coupon.model_dump()
        doc['created_at'] = doc['created_at'].isoformat()
        doc['expiry_date'] = doc['expiry_date'].isoformat()
        doc['used_at'] = None
        
        result = await db.coupons.insert_one(doc)
        
        created = await db.coupons.find_one({"_id": result.inserted_id}, {"_id": 0})
        return created
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/admin/coupons")
async def list_coupons(current_user: dict = Depends(get_admin_user), page: int = 1, limit: int = 20):
    """List paginated coupons (admin only)"""
    try:
        # Validar parámetros
        if page < 1:
            page = 1
        if limit < 1:
            limit = 20
        elif limit > 1000:
            limit = 1000
            
        skip = (page - 1) * limit
        
        # Contar total
        total = await db.coupons.count_documents({})
        
        # Obtener cupones paginados
        coupons = await db.coupons.find({}, {"_id": 0}).skip(skip).limit(limit).to_list(None)
        
        for coupon in coupons:
            if isinstance(coupon['created_at'], str):
                coupon['created_at'] = coupon['created_at']
            if isinstance(coupon['expiry_date'], str):
                coupon['expiry_date'] = coupon['expiry_date']
        
        total_pages = (total + limit - 1) // limit
        
        return {
            "data": coupons,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/admin/coupons/{coupon_id}")
async def delete_coupon(coupon_id: str, current_user: dict = Depends(get_admin_user)):
    """Delete a coupon (admin only)"""
    result = await db.coupons.delete_one({"id": coupon_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Coupon not found")
    return {"message": "Coupon deleted successfully"}

@api_router.patch("/admin/coupons/{coupon_id}/mark-used")
async def mark_coupon_used(coupon_id: str, current_user: dict = Depends(get_admin_user)):
    """Mark a coupon as used (admin only). Auto-deleted after 10 hours."""
    try:
        coupon = await db.coupons.find_one({"id": coupon_id})
        if not coupon:
            raise HTTPException(status_code=404, detail="Coupon not found")
        
        if coupon.get('is_used'):
            raise HTTPException(status_code=400, detail="Coupon is already used")
        
        # Establece scheduled_deletion_at a 10 horas después
        scheduled_deletion = datetime.now(timezone.utc) + timedelta(hours=10)
        
        result = await db.coupons.update_one(
            {"id": coupon_id},
            {
                "$set": {
                    "is_used": True,
                    "used_at": datetime.now(timezone.utc).isoformat(),
                    "scheduled_deletion_at": scheduled_deletion.isoformat()
                }
            }
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=400, detail="Could not mark coupon as used")
        
        return {"message": "Coupon marked as used successfully. Will be auto-deleted in 10 hours."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.patch("/admin/coupons/{coupon_id}/revert")
async def revert_coupon_status(coupon_id: str, current_user: dict = Depends(get_admin_user)):
    """Revert a coupon to not used (admin only)"""
    try:
        coupon = await db.coupons.find_one({"id": coupon_id})
        if not coupon:
            raise HTTPException(status_code=404, detail="Coupon not found")
        
        if not coupon.get('is_used'):
            raise HTTPException(status_code=400, detail="Coupon is not used")
        
        result = await db.coupons.update_one(
            {"id": coupon_id},
            {
                "$set": {
                    "is_used": False
                },
                "$unset": {
                    "used_at": ""
                }
            }
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=400, detail="Could not revert coupon status")
        
        return {"message": "Coupon status reverted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/user/coupons")
async def get_user_coupons(current_user: dict = Depends(get_current_user)):
    """Get coupons for current user"""
    try:
        coupons = await db.coupons.find(
            {"user_id": current_user['user_id']},
            {"_id": 0}
        ).to_list(None)
        
        # Convertir strings a datetime si es necesario
        for coupon in coupons:
            if isinstance(coupon['created_at'], str):
                coupon['created_at'] = datetime.fromisoformat(coupon['created_at'])
            if isinstance(coupon['expiry_date'], str):
                coupon['expiry_date'] = datetime.fromisoformat(coupon['expiry_date'])
            if coupon.get('used_at') and isinstance(coupon['used_at'], str):
                coupon['used_at'] = datetime.fromisoformat(coupon['used_at'])
        
        return coupons
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/user/coupons/{coupon_code}/use")
async def use_coupon(coupon_code: str, current_user: dict = Depends(get_current_user)):
    """Use a coupon (mark as used). Auto-deleted after 10 hours."""
    try:
        coupon = await db.coupons.find_one({"code": coupon_code}, {"_id": 0})
        
        if not coupon:
            raise HTTPException(status_code=404, detail="Coupon not found")
        
        # Verificar que pertenece al usuario
        if coupon['user_id'] != current_user['user_id']:
            raise HTTPException(status_code=403, detail="This coupon is not for you")
        
        # Verificar que no esté usado
        if coupon['is_used']:
            raise HTTPException(status_code=400, detail="Coupon already used")
        
        # Verificar fecha de vencimiento
        expiry = datetime.fromisoformat(coupon['expiry_date']) if isinstance(coupon['expiry_date'], str) else coupon['expiry_date']
        if datetime.now(timezone.utc) > expiry:
            # Marcar como expirado también con scheduled_deletion_at
            scheduled_deletion = datetime.now(timezone.utc) + timedelta(hours=10)
            await db.coupons.update_one(
                {"code": coupon_code},
                {
                    "$set": {
                        "scheduled_deletion_at": scheduled_deletion.isoformat()
                    }
                }
            )
            raise HTTPException(status_code=400, detail="Coupon expired")
        
        # Marcar como usado con scheduled_deletion_at para auto-delete en 10 horas
        scheduled_deletion = datetime.now(timezone.utc) + timedelta(hours=10)
        result = await db.coupons.update_one(
            {"code": coupon_code},
            {
                "$set": {
                    "is_used": True,
                    "used_at": datetime.now(timezone.utc).isoformat(),
                    "scheduled_deletion_at": scheduled_deletion.isoformat()
                }
            }
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=400, detail="Could not use coupon")
        
        return {
            "message": "Coupon used successfully. Will be auto-deleted in 10 hours.",
            "discount_type": coupon['discount_type'],
            "discount_value": coupon['discount_value'],
            "content_id": coupon.get('content_id')
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== COUPON WHATSAPP ROUTES ==========

@api_router.post("/coupons/{coupon_code}/generate-whatsapp-token")
async def generate_whatsapp_token(
    coupon_code: str,
    request: CouponWhatsAppRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate a WhatsApp token to mark coupon as used and send via WhatsApp"""
    try:
        worker_phone = request.worker_phone
        coupon = await db.coupons.find_one({"code": coupon_code}, {"_id": 0})
        
        if not coupon:
            raise HTTPException(status_code=404, detail="Coupon not found")
        
        # Verificar que pertenece al usuario
        if coupon['user_id'] != current_user['user_id']:
            raise HTTPException(status_code=403, detail="This coupon is not for you")
        
        # Verificar que no esté usado
        if coupon['is_used']:
            raise HTTPException(status_code=400, detail="Coupon already used")
        
        # Verificar fecha de vencimiento
        expiry = datetime.fromisoformat(coupon['expiry_date']) if isinstance(coupon['expiry_date'], str) else coupon['expiry_date']
        if datetime.now(timezone.utc) > expiry:
            raise HTTPException(status_code=400, detail="Coupon expired")
        
        # Generar token único
        use_token = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
        
        # Guardar token en la base de datos
        await db.coupon_use_tokens.insert_one({
            "token": use_token,
            "coupon_code": coupon_code,
            "user_id": current_user['user_id'],
            "worker_phone": worker_phone,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "used": False
        })
        
        # URL para marcar como usado
        use_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/coupon/use/{use_token}"
        
        # Mensaje WhatsApp
        whatsapp_message = f"Hola, te comparte este cupón para usar en tu compra:\n\nCupón: {coupon_code}\n{coupon['description']}\n\nHaz clic aquí para validar: {use_url}"
        
        # URL para abrir WhatsApp
        whatsapp_url = f"https://wa.me/{worker_phone}?text={urllib.parse.quote(whatsapp_message)}"
        
        return {
            "token": use_token,
            "whatsapp_url": whatsapp_url,
            "message": "WhatsApp token generated successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/coupons/use/{use_token}")
async def use_coupon_with_token(use_token: str):
    """Use a coupon with a WhatsApp token (no auth required)"""
    try:
        # Buscar el token
        token_doc = await db.coupon_use_tokens.find_one({"token": use_token}, {"_id": 0})
        
        if not token_doc:
            raise HTTPException(status_code=404, detail="Token not found or invalid")
        
        if token_doc['used']:
            raise HTTPException(status_code=400, detail="Coupon already marked as used")
        
        coupon_code = token_doc['coupon_code']
        
        # Buscar y actualizar el cupón
        coupon = await db.coupons.find_one({"code": coupon_code}, {"_id": 0})
        
        if not coupon:
            raise HTTPException(status_code=404, detail="Coupon not found")
        
        # Verificar que no esté usado
        if coupon['is_used']:
            raise HTTPException(status_code=400, detail="Coupon already used")
        
        # Verificar fecha de vencimiento
        expiry = datetime.fromisoformat(coupon['expiry_date']) if isinstance(coupon['expiry_date'], str) else coupon['expiry_date']
        if datetime.now(timezone.utc) > expiry:
            raise HTTPException(status_code=400, detail="Coupon expired")
        
        # Marcar como usado
        result = await db.coupons.update_one(
            {"code": coupon_code},
            {
                "$set": {
                    "is_used": True,
                    "used_at": datetime.now(timezone.utc).isoformat()
                }
            }
        )
        
        # Marcar token como usado
        await db.coupon_use_tokens.update_one(
            {"token": use_token},
            {"$set": {"used": True, "used_at": datetime.now(timezone.utc).isoformat()}}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=400, detail="Could not use coupon")
        
        return {
            "message": "Coupon used successfully",
            "coupon_code": coupon_code,
            "discount_type": coupon['discount_type'],
            "discount_value": coupon['discount_value'],
            "content_id": coupon.get('content_id')
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/user/coupons/new/count")
async def get_new_coupons_count(current_user: dict = Depends(get_current_user)):
    """Get count of new (unviewed) coupons for current user"""
    try:
        new_coupons = await db.coupons.find(
            {
                "user_id": current_user['user_id'],
                "viewed_by_user": False
            }
        ).to_list(None)
        
        return {"new_coupons_count": len(new_coupons)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/user/coupons/mark-as-viewed")
async def mark_coupons_as_viewed(current_user: dict = Depends(get_current_user)):
    """Mark all unviewed coupons as viewed"""
    try:
        result = await db.coupons.update_many(
            {
                "user_id": current_user['user_id'],
                "viewed_by_user": False
            },
            {"$set": {"viewed_by_user": True}}
        )
        
        return {
            "message": "Coupons marked as viewed",
            "updated_count": result.modified_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== WATCHLIST ROUTES ==========

@api_router.post("/watchlist/{content_id}")
async def add_to_watchlist(content_id: str, current_user: dict = Depends(get_current_user)):
    # Check if content exists
    content = await db.content.find_one({"id": content_id}, {"_id": 0})
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    
    # Check if already in watchlist
    existing = await db.watchlist.find_one({
        "user_id": current_user['user_id'],
        "content_id": content_id
    }, {"_id": 0})
    
    if existing:
        raise HTTPException(status_code=400, detail="Content already in watchlist")
    
    watchlist_item = Watchlist(
        user_id=current_user['user_id'],
        content_id=content_id
    )
    
    doc = watchlist_item.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.watchlist.insert_one(doc)
    
    return {"message": "Added to watchlist"}

@api_router.get("/watchlist")
async def get_watchlist(current_user: dict = Depends(get_current_user)):
    watchlist_items = await db.watchlist.find({"user_id": current_user['user_id']}, {"_id": 0}).to_list(1000)
    
    # Get content details for each item
    content_ids = [item['content_id'] for item in watchlist_items]
    contents = await db.content.find({"id": {"$in": content_ids}}, {"_id": 0}).to_list(1000)
    
    for content in contents:
        if isinstance(content['created_at'], str):
            content['created_at'] = datetime.fromisoformat(content['created_at'])
    
    return contents

@api_router.delete("/watchlist/{content_id}")
async def remove_from_watchlist(content_id: str, current_user: dict = Depends(get_current_user)):
    result = await db.watchlist.delete_one({
        "user_id": current_user['user_id'],
        "content_id": content_id
    })
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Content not in watchlist")
    
    return {"message": "Removed from watchlist"}

@api_router.get("/watchlist/check/{content_id}")
async def check_in_watchlist(content_id: str, current_user: dict = Depends(get_current_user)):
    existing = await db.watchlist.find_one({
        "user_id": current_user['user_id'],
        "content_id": content_id
    }, {"_id": 0})
    
    return {"in_watchlist": existing is not None}

@api_router.get("/admin/watchlists")
async def get_all_watchlists(current_user: dict = Depends(get_current_user)):
    """Get all watchlists by user (admin only)"""
    if current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Acceso denegado")
    
    # Get all users
    users = await db.users.find({}, {"_id": 0, "id": 1, "email": 1}).to_list(1000)
    
    result = []
    for user in users:
        watchlist_items = await db.watchlist.find(
            {"user_id": user['id']}, 
            {"_id": 0}
        ).to_list(1000)
        
        # Get content details for each watchlist item
        content_with_details = []
        for item in watchlist_items:
            content = await db.content.find_one(
                {"id": item['content_id']}, 
                {"_id": 0}
            )
            if content:
                content_with_details.append({
                    "content_id": item['content_id'],
                    "title": content.get('title', 'N/A'),
                    "content_type": content.get('content_type', 'N/A'),
                    "added_at": item.get('added_at', 'N/A')
                })
        
        result.append({
            "user_id": user['id'],
            "email": user['email'],
            "watchlist_count": len(content_with_details),
            "watchlist": content_with_details
        })
    
    return result

# ========== USER PROFILE ROUTES ==========

@api_router.get("/user/reviews", response_model=List[ReviewResponse])
async def get_user_reviews(current_user: dict = Depends(get_current_user)):
    reviews = await db.reviews.find({"user_id": current_user['user_id']}, {"_id": 0}).to_list(1000)
    for review in reviews:
        if isinstance(review['created_at'], str):
            review['created_at'] = datetime.fromisoformat(review['created_at'])
    return reviews

@api_router.post("/user/change-password")
async def change_password(password_data: PasswordChange, current_user: dict = Depends(get_current_user)):
    user_doc = await db.users.find_one({"id": current_user['user_id']}, {"_id": 0})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not verify_password(password_data.old_password, user_doc['password_hash']):
        raise HTTPException(status_code=401, detail="Invalid old password")
    
    new_hash = hash_password(password_data.new_password)
    await db.users.update_one(
        {"id": current_user['user_id']},
        {"$set": {"password_hash": new_hash}}
    )
    
    return {"message": "Password changed successfully"}

# ========== CONTENT UPDATE ROUTES ==========

@api_router.post("/admin/content/update-all")
async def update_all_content(current_user: dict = Depends(get_admin_user)):
    """Update all content items with fresh data from TMDB"""
    # Get all content
    all_content = await db.content.find({}, {"_id": 0}).to_list(10000)
    
    if not all_content:
        return {
            "message": "No content to update",
            "total": 0,
            "updated": 0,
            "failed": 0,
            "failed_items": []
        }
    
    updated_count = 0
    failed_count = 0
    failed_items = []
    
    for content in all_content:
        try:
            # Fetch fresh data from TMDB
            if content['content_type'] == 'movie':
                tmdb_data = fetch_movie_details(content['tmdb_id'])
            else:  # tv or novela
                tmdb_data = fetch_tv_details(content['tmdb_id'])
            
            if not tmdb_data:
                failed_count += 1
                failed_items.append({
                    "content_id": content['id'],
                    "title": content.get('title', 'Unknown'),
                    "reason": "Not found in TMDB"
                })
                continue
            
            # Get trailer
            trailer_key = get_trailer_url(content['tmdb_id'], content['content_type'])
            trailer_url = f"https://www.youtube.com/embed/{trailer_key}" if trailer_key else None
            
            # Update fields with fresh TMDB data
            update_data = {
                "title": tmdb_data.get('title') or tmdb_data.get('name', ''),
                "original_title": tmdb_data.get('original_title') or tmdb_data.get('original_name'),
                "overview": tmdb_data.get('overview'),
                "poster_path": tmdb_data.get('poster_path'),
                "backdrop_path": tmdb_data.get('backdrop_path'),
                "genres": [g['name'] for g in tmdb_data.get('genres', [])],
                "countries": [c['name'] for c in tmdb_data.get('production_countries', [])],
                "release_date": tmdb_data.get('release_date') or tmdb_data.get('first_air_date'),
                "vote_average": tmdb_data.get('vote_average'),
                "trailer_key": trailer_key,
                "trailer_url": trailer_url,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # For TV shows, update seasons
            if content['content_type'] in ['tv', 'novela']:
                update_data["seasons"] = tmdb_data.get('seasons_detailed')
            
            # Update in database
            result = await db.content.update_one(
                {"id": content['id']},
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                updated_count += 1
            else:
                failed_count += 1
                failed_items.append({
                    "content_id": content['id'],
                    "title": content.get('title', 'Unknown'),
                    "reason": "Database update failed"
                })
        
        except Exception as e:
            failed_items.append({
                "content_id": content['id'],
                "title": content.get('title', 'Unknown'),
                "reason": str(e)
            })
            failed_count += 1
    
    return {
        "message": "Content update completed",
        "total": len(all_content),
        "updated": updated_count,
        "failed": failed_count,
        "failed_items": failed_items
    }

@api_router.post("/admin/content/{content_id}/update")
async def update_single_content(content_id: str, current_user: dict = Depends(get_admin_user)):
    """Update a single content item with data from TMDB"""
    # Get content from DB
    content = await db.content.find_one({"id": content_id}, {"_id": 0})
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    
    # Fetch fresh data from TMDB
    try:
        if content['content_type'] == 'movie':
            tmdb_data = fetch_movie_details(content['tmdb_id'])
        else:  # tv or novela
            tmdb_data = fetch_tv_details(content['tmdb_id'])
        
        if not tmdb_data:
            raise HTTPException(status_code=404, detail="Content not found in TMDB")
        
        # Get trailer
        trailer_key = get_trailer_url(content['tmdb_id'], content['content_type'])
        trailer_url = f"https://www.youtube.com/embed/{trailer_key}" if trailer_key else None
        
        # Update fields with fresh TMDB data
        update_data = {
            "title": tmdb_data.get('title') or tmdb_data.get('name', ''),
            "original_title": tmdb_data.get('original_title') or tmdb_data.get('original_name'),
            "overview": tmdb_data.get('overview'),
            "poster_path": tmdb_data.get('poster_path'),
            "backdrop_path": tmdb_data.get('backdrop_path'),
            "genres": [g['name'] for g in tmdb_data.get('genres', [])],
            "countries": [c['name'] for c in tmdb_data.get('production_countries', [])],
            "release_date": tmdb_data.get('release_date') or tmdb_data.get('first_air_date'),
            "vote_average": tmdb_data.get('vote_average'),
            "trailer_key": trailer_key,
            "trailer_url": trailer_url,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # For TV shows, update seasons
        if content['content_type'] in ['tv', 'novela']:
            update_data["seasons"] = tmdb_data.get('seasons_detailed')
        
        # Update in database
        result = await db.content.update_one(
            {"id": content_id},
            {"$set": update_data}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=400, detail="Failed to update content")
        
        return {
            "message": "Content updated successfully",
            "content_id": content_id,
            "updated_fields": list(update_data.keys())
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating content: {str(e)}")

# ========== PROMOTION ROUTES ==========

@api_router.get("/promotions/active")
async def get_active_promotion():
    """Get the currently active promotion to display"""
    try:
        promo = await db.promotions.find_one({"isActive": True})
        if not promo:
            return None
        
        # Convert ObjectId to string
        promo["_id"] = str(promo["_id"])
        return promo
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/promotions/{promotion_id}")
async def get_promotion(promotion_id: str):
    """Get a specific promotion by ID"""
    try:
        from bson.objectid import ObjectId
        if not ObjectId.is_valid(promotion_id):
            raise HTTPException(status_code=400, detail="Invalid promotion ID")
        
        promo = await db.promotions.find_one({"_id": ObjectId(promotion_id)})
        if not promo:
            raise HTTPException(status_code=404, detail="Promotion not found")
        
        promo["_id"] = str(promo["_id"])
        return promo
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/admin/promotions", response_model=list)
async def list_promotions(current_user: dict = Depends(get_current_user)):
    """List all promotions (admin only)"""
    if current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        promos = await db.promotions.find().to_list(None)
        for promo in promos:
            promo["_id"] = str(promo["_id"])
        return promos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Upload image endpoint for promotions
@api_router.post("/admin/promotions/upload-image")
async def upload_promotion_image(
    file: UploadFile = File(...),
    admin = Depends(get_admin_user)
):
    """Upload image for promotion (admin only)"""
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Archivo no permitido. Extensiones válidas: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Check file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Archivo muy grande. Máximo: {MAX_FILE_SIZE / (1024*1024)}MB"
            )
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = UPLOADS_DIR / unique_filename
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(contents)
        
        # Return relative URL (client will handle the full URL based on backend domain)
        relative_url = f"/uploads/promotions/{unique_filename}"
        
        return {
            "success": True,
            "url": relative_url,
            "filename": unique_filename
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al subir imagen: {str(e)}"
        )

@api_router.post("/admin/promotions")

async def create_promotion(
    promotion: PromotionCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new promotion (admin only)"""
    if current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        from bson.objectid import ObjectId
        promo_dict = promotion.model_dump()
        promo_dict["created_at"] = datetime.now(timezone.utc)
        promo_dict["updated_at"] = datetime.now(timezone.utc)
        promo_dict["created_by"] = current_user.get("id")
        
        result = await db.promotions.insert_one(promo_dict)
        
        created_promo = await db.promotions.find_one({"_id": result.inserted_id})
        created_promo["_id"] = str(created_promo["_id"])
        
        return created_promo
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.put("/admin/promotions/{promotion_id}")
async def update_promotion(
    promotion_id: str,
    promotion: PromotionUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update a promotion (admin only)"""
    if current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        from bson.objectid import ObjectId
        if not ObjectId.is_valid(promotion_id):
            raise HTTPException(status_code=400, detail="Invalid promotion ID")
        
        # Get current promotion to check if image is being replaced
        current_promo = await db.promotions.find_one({"_id": ObjectId(promotion_id)})
        if not current_promo:
            raise HTTPException(status_code=404, detail="Promotion not found")
        
        update_dict = {k: v for k, v in promotion.model_dump().items() if v is not None}
        update_dict["updated_at"] = datetime.now(timezone.utc)
        
        # If prizeImage is being updated with a new value, delete old image
        if "prizeImage" in update_dict and update_dict["prizeImage"]:
            old_image_url = current_promo.get("prizeImage")
            new_image_url = update_dict["prizeImage"]
            
            # Only delete if it's a different image
            if old_image_url and old_image_url != new_image_url:
                try:
                    old_filename = old_image_url.split("/")[-1]
                    old_image_path = UPLOADS_DIR / old_filename
                    
                    if old_image_path.exists():
                        old_image_path.unlink()
                        logger.info(f"Deleted old image: {old_filename}")
                except Exception as e:
                    logger.warning(f"Error deleting old image: {str(e)}")
        
        result = await db.promotions.update_one(
            {"_id": ObjectId(promotion_id)},
            {"$set": update_dict}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Promotion not found")
        
        updated_promo = await db.promotions.find_one({"_id": ObjectId(promotion_id)})
        updated_promo["_id"] = str(updated_promo["_id"])
        
        return updated_promo
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/admin/promotions/{promotion_id}")
async def delete_promotion(
    promotion_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a promotion (admin only)"""
    if current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        from bson.objectid import ObjectId
        if not ObjectId.is_valid(promotion_id):
            raise HTTPException(status_code=400, detail="Invalid promotion ID")
        
        # Get promotion before deleting to access image path
        promotion = await db.promotions.find_one({"_id": ObjectId(promotion_id)})
        
        if not promotion:
            raise HTTPException(status_code=404, detail="Promotion not found")
        
        # Delete image file if exists
        if promotion.get("prizeImage"):
            try:
                # Extract filename from URL: /uploads/promotions/uuid.ext
                image_url = promotion["prizeImage"]
                filename = image_url.split("/")[-1]  # Get last part
                image_path = UPLOADS_DIR / filename
                
                # Delete file if it exists
                if image_path.exists():
                    image_path.unlink()  # Remove file
                    logger.info(f"Deleted image: {filename}")
            except Exception as e:
                # Log error but continue with promotion deletion
                logger.warning(f"Error deleting image file: {str(e)}")
        
        # Delete promotion document
        result = await db.promotions.delete_one({"_id": ObjectId(promotion_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Promotion not found")
        
        return {"message": "Promotion deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== WORKER ROUTES ==========

@api_router.get("/admin/workers")
async def get_all_workers(current_user: dict = Depends(get_admin_user)):
    """Get all workers (admin only)"""
    try:
        workers = await db.workers.find({}, {"_id": 0}).sort("created_at", -1).to_list(None)
        
        # Convertir strings a datetime
        for worker in workers:
            if isinstance(worker.get('created_at'), str):
                worker['created_at'] = datetime.fromisoformat(worker['created_at'])
            if isinstance(worker.get('updated_at'), str):
                worker['updated_at'] = datetime.fromisoformat(worker['updated_at'])
        
        return {"data": workers, "total": len(workers)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/admin/workers")
async def create_worker(
    worker_data: dict,
    current_user: dict = Depends(get_admin_user)
):
    """Create a new worker (admin only)"""
    try:
        worker = {
            "id": str(uuid.uuid4()),
            "name": worker_data.get("name"),
            "phone": worker_data.get("phone"),
            "full_phone": worker_data.get("full_phone"),
            "work_days": worker_data.get("work_days", ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        result = await db.workers.insert_one(worker)
        worker["_id"] = str(result.inserted_id)
        
        return worker
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/admin/workers/{worker_id}")
async def get_worker(
    worker_id: str,
    current_user: dict = Depends(get_admin_user)
):
    """Get a specific worker (admin only)"""
    try:
        worker = await db.workers.find_one({"id": worker_id}, {"_id": 0})
        if not worker:
            raise HTTPException(status_code=404, detail="Worker not found")
        
        if isinstance(worker.get('created_at'), str):
            worker['created_at'] = datetime.fromisoformat(worker['created_at'])
        if isinstance(worker.get('updated_at'), str):
            worker['updated_at'] = datetime.fromisoformat(worker['updated_at'])
        
        return worker
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.put("/admin/workers/{worker_id}")
async def update_worker(
    worker_id: str,
    worker_data: dict,
    current_user: dict = Depends(get_admin_user)
):
    """Update a worker (admin only)"""
    try:
        worker = await db.workers.find_one({"id": worker_id}, {"_id": 0})
        if not worker:
            raise HTTPException(status_code=404, detail="Worker not found")
        
        update_data = {}
        if "name" in worker_data:
            update_data["name"] = worker_data["name"]
        if "phone" in worker_data:
            update_data["phone"] = worker_data["phone"]
        if "full_phone" in worker_data:
            update_data["full_phone"] = worker_data["full_phone"]
        if "work_days" in worker_data:
            update_data["work_days"] = worker_data["work_days"]
        
        update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        result = await db.workers.update_one(
            {"id": worker_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Worker not found")
        
        updated_worker = await db.workers.find_one({"id": worker_id}, {"_id": 0})
        
        if isinstance(updated_worker.get('created_at'), str):
            updated_worker['created_at'] = datetime.fromisoformat(updated_worker['created_at'])
        if isinstance(updated_worker.get('updated_at'), str):
            updated_worker['updated_at'] = datetime.fromisoformat(updated_worker['updated_at'])
        
        return updated_worker
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/admin/workers/{worker_id}")
async def delete_worker(
    worker_id: str,
    current_user: dict = Depends(get_admin_user)
):
    """Delete a worker (admin only)"""
    try:
        result = await db.workers.delete_one({"id": worker_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Worker not found")
        
        return {"message": "Worker deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/workers/available/{day_of_week}")
async def get_available_workers(day_of_week: str):
    """Get workers available for a specific day of week (public)"""
    try:
        # Validar día
        valid_days = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        if day_of_week not in valid_days:
            raise HTTPException(status_code=400, detail=f"Invalid day. Must be one of {valid_days}")
        
        workers = await db.workers.find(
            {"work_days": day_of_week},
            {"_id": 0}
        ).to_list(None)
        
        if not workers:
            # Si no hay workers para ese día, devolver todos
            workers = await db.workers.find({}, {"_id": 0}).to_list(None)
        
        return {"data": workers, "total": len(workers)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(api_router)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    # Create unique index on email to prevent duplicate registrations
    try:
        await db.users.create_index("email", unique=True)
        logger.info("Email unique index created successfully")
    except Exception as e:
        logger.warning(f"Email index creation warning (may already exist): {e}")
    
    # Create indexes for performance optimization
    try:
        # Content indexes
        await db.content.create_index("content_type")
        await db.content.create_index("genres")
        await db.content.create_index("created_at")
        logger.info("Content indexes created successfully")
    except Exception as e:
        logger.warning(f"Content indexes warning: {e}")
    
    try:
        # Reviews indexes
        await db.reviews.create_index("content_id")
        await db.reviews.create_index("user_id")
        await db.reviews.create_index("created_at")
        logger.info("Review indexes created successfully")
    except Exception as e:
        logger.warning(f"Review indexes warning: {e}")
    
    try:
        # Watchlist indexes
        await db.watchlist.create_index("user_id")
        await db.watchlist.create_index("content_id")
        logger.info("Watchlist indexes created successfully")
    except Exception as e:
        logger.warning(f"Watchlist indexes warning: {e}")
    
    try:
        # Coupons indexes
        await db.coupons.create_index("user_id")
        await db.coupons.create_index("code", unique=True)
        await db.coupons.create_index("is_used")
        await db.coupons.create_index("expiry_date")
        logger.info("Coupon indexes created successfully")
    except Exception as e:
        logger.warning(f"Coupon indexes warning: {e}")
    
    try:
        # TTL index for auto-delete of used/expired coupons (10 hours)
        await db.coupons.create_index("scheduled_deletion_at", expireAfterSeconds=0, sparse=True)
        logger.info("Coupon TTL index created successfully")
    except Exception as e:
        logger.warning(f"Coupon TTL index warning: {e}")
    
    try:
        # Workers indexes
        await db.workers.create_index("id", unique=True)
        await db.workers.create_index("work_days")
        logger.info("Workers indexes created successfully")
    except Exception as e:
        logger.warning(f"Workers indexes warning: {e}")

app.include_router(api_router)
async def shutdown_event():
    client.close()