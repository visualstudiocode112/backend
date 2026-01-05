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

# Mount static files for uploaded images
app.mount("/uploads", StaticFiles(directory=str(ROOT_DIR / "uploads")), name="uploads")

# CORS Config is handled by Nginx reverse proxy in production
# For local development, uncomment the section below:
# ALLOWED_ORIGINS = [
#     'http://localhost:3000',
#     'http://localhost:5000',
#     'http://localhost',
#     'http://127.0.0.1:3000',
#     'http://127.0.0.1:5000',
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=ALLOWED_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
#     expose_headers=["*"],
# )

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
    status: str = "active"  # "active", "used", "expired", "cancelled"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str  # ID del admin que lo creó

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

@api_router.get("/content", response_model=List[Content])
async def get_all_content():
    content_list = await db.content.find({}, {"_id": 0}).to_list(None)
    for item in content_list:
        if isinstance(item['created_at'], str):
            item['created_at'] = datetime.fromisoformat(item['created_at'])
        
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
        res = await db.content.aggregate(pipeline).to_list(limit)
        return [r['_id'] for r in res]

    regex = {"$regex": f"^{q}", "$options": "i"}
    pipeline = [
        {"$unwind": "$genres"},
        {"$match": {"genres": regex}},
        {"$group": {"_id": "$genres", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit}
    ]
    res = await db.content.aggregate(pipeline).to_list(limit)
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
        res = await db.content.aggregate(pipeline).to_list(limit)
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
    res = await db.content.aggregate(pipeline).to_list(limit)
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

@api_router.get("/reviews/content/{content_id}", response_model=List[ReviewResponse])
async def get_content_reviews(content_id: str):
    reviews = await db.reviews.find({"content_id": content_id}, {"_id": 0}).to_list(1000)
    for review in reviews:
        if isinstance(review['created_at'], str):
            review['created_at'] = datetime.fromisoformat(review['created_at'])
    return reviews

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

@api_router.get("/admin/users", response_model=List[UserResponse])
async def get_all_users(current_user: dict = Depends(get_admin_user)):
    users = await db.users.find({}, {"_id": 0, "password_hash": 0}).to_list(1000)
    for user in users:
        if isinstance(user['created_at'], str):
            user['created_at'] = datetime.fromisoformat(user['created_at'])
    return users

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
async def get_user_watchlist(user_id: str, current_user: dict = Depends(get_admin_user)):
    """Get watchlist of a specific user (admin only)"""
    # Verify user exists
    user = await db.users.find_one({"id": user_id}, {"_id": 0, "id": 1})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get watchlist items
    watchlist_items = await db.watchlist.find(
        {"user_id": user_id}, 
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
            # Get fresh data from TMDB
            tmdb_data = None
            try:
                if content.get('content_type') == 'movie':
                    tmdb_data = fetch_movie_details(content.get('tmdb_id'))
                else:
                    tmdb_data = fetch_tv_details(content.get('tmdb_id'))
            except Exception as e:
                print(f"Error fetching TMDB data: {e}")
            
            # Use TMDB data if available, otherwise use cached data
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
    
    return content_with_details

@api_router.get("/admin/users/{user_id}/coupons")
async def get_user_coupons_admin(user_id: str, current_user: dict = Depends(get_admin_user)):
    """Get coupons of a specific user (admin only)"""
    try:
        # Verify user exists
        user = await db.users.find_one({"id": user_id}, {"_id": 0})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get user coupons
        coupons = await db.coupons.find(
            {"user_id": user_id},
            {"_id": 0}
        ).to_list(1000)
        
        # Convertir strings a datetime si es necesario
        for coupon in coupons:
            if isinstance(coupon['created_at'], str):
                coupon['created_at'] = datetime.fromisoformat(coupon['created_at'])
            if isinstance(coupon['expiry_date'], str):
                coupon['expiry_date'] = datetime.fromisoformat(coupon['expiry_date'])
            if coupon.get('used_at') and isinstance(coupon['used_at'], str):
                coupon['used_at'] = datetime.fromisoformat(coupon['used_at'])
        
        return coupons
    except HTTPException:
        raise
    except Exception as e:
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

@api_router.get("/admin/reviews", response_model=List[ReviewResponse])
async def get_all_reviews(current_user: dict = Depends(get_admin_user)):
    reviews = await db.reviews.find({}, {"_id": 0}).to_list(1000)
    
    # Enriquecer las reseñas con el título del contenido
    for review in reviews:
        if isinstance(review['created_at'], str):
            review['created_at'] = datetime.fromisoformat(review['created_at'])
        
        # Buscar el contenido para obtener el título
        content = await db.content.find_one({"id": review['content_id']}, {"_id": 0})
        if content:
            review['content_title'] = content.get('title', 'Contenido no disponible')
        else:
            review['content_title'] = 'Contenido eliminado'
    
    return reviews

@api_router.delete("/admin/reviews/{review_id}")
async def delete_review(review_id: str, current_user: dict = Depends(get_admin_user)):
    review = await db.reviews.find_one({"id": review_id}, {"_id": 0})
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    await db.reviews.delete_one({"id": review_id})
    return {"message": "Review deleted successfully"}

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
async def list_coupons(current_user: dict = Depends(get_admin_user)):
    """List all coupons (admin only)"""
    try:
        coupons = await db.coupons.find({}, {"_id": 0}).to_list(None)
        for coupon in coupons:
            if isinstance(coupon['created_at'], str):
                coupon['created_at'] = coupon['created_at']
            if isinstance(coupon['expiry_date'], str):
                coupon['expiry_date'] = coupon['expiry_date']
        return coupons
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/admin/coupons/{coupon_id}")
async def delete_coupon(coupon_id: str, current_user: dict = Depends(get_admin_user)):
    """Delete a coupon (admin only)"""
    result = await db.coupons.delete_one({"id": coupon_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Coupon not found")
    return {"message": "Coupon deleted successfully"}

@api_router.put("/admin/coupons/{coupon_id}/status")
async def update_coupon_status(
    coupon_id: str,
    new_status: str,
    current_user: dict = Depends(get_admin_user)
):
    """Update coupon status (admin only)"""
    try:
        # Validar estado válido
        valid_statuses = ["active", "used", "expired", "cancelled"]
        if new_status not in valid_statuses:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid status. Must be one of {valid_statuses}"
            )
        
        # Verificar que el cupón existe
        coupon = await db.coupons.find_one({"id": coupon_id}, {"_id": 0})
        if not coupon:
            raise HTTPException(status_code=404, detail="Coupon not found")
        
        # Actualizar estado
        update_data = {"status": new_status}
        
        # Si se marca como usado, registrar la fecha
        if new_status == "used":
            update_data["is_used"] = True
            update_data["used_at"] = datetime.now(timezone.utc).isoformat()
        
        result = await db.coupons.update_one(
            {"id": coupon_id},
            {"$set": update_data}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=400, detail="Could not update coupon")
        
        # Retornar cupón actualizado
        updated_coupon = await db.coupons.find_one({"id": coupon_id}, {"_id": 0})
        return updated_coupon
    
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
    """Use a coupon (mark as used)"""
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
        
        if result.modified_count == 0:
            raise HTTPException(status_code=400, detail="Could not use coupon")
        
        return {
            "message": "Coupon used successfully",
            "discount_type": coupon['discount_type'],
            "discount_value": coupon['discount_value'],
            "content_id": coupon.get('content_id')
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/coupons/{coupon_code}/generate-token")
async def generate_coupon_token(coupon_code: str, current_user: dict = Depends(get_current_user)):
    """Generate a shareable token to use a coupon without authentication"""
    try:
        coupon = await db.coupons.find_one({"code": coupon_code}, {"_id": 0})
        
        if not coupon:
            raise HTTPException(status_code=404, detail="Coupon not found")
        
        # Verificar que pertenece al usuario
        if coupon['user_id'] != current_user['user_id']:
            raise HTTPException(status_code=403, detail="This coupon is not for you")
        
        # Generar token especial de una sola vez
        token_payload = {
            'coupon_code': coupon_code,
            'user_id': current_user['user_id'],
            'exp': datetime.now(timezone.utc) + timedelta(hours=24)
        }
        token = jwt.encode(token_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        
        # Generar el URL
        frontend_url = os.environ.get('FRONTEND_URL', 'http://localhost:3000')
        shareable_url = f"{frontend_url}/coupon/use/{token}"
        
        return {
            "token": token,
            "url": shareable_url,
            "coupon_code": coupon_code
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/coupons/use/{token}")
async def use_coupon_with_token(token: str):
    """Use a coupon with a shareable token (no authentication required)"""
    try:
        # Validar token
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        coupon_code = payload.get('coupon_code')
        user_id = payload.get('user_id')
        
        if not coupon_code or not user_id:
            raise HTTPException(status_code=400, detail="Invalid token")
        
        coupon = await db.coupons.find_one({"code": coupon_code}, {"_id": 0})
        
        if not coupon:
            raise HTTPException(status_code=404, detail="Coupon not found")
        
        # Verificar que pertenece al usuario del token
        if coupon['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="This coupon is not for this token")
        
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
                    "used_at": datetime.now(timezone.utc).isoformat(),
                    "status": "used"
                }
            }
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

@app.on_event("shutdown")
async def shutdown_event():
    client.close()