from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
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

app = FastAPI()
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

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
        added_by=current_user['user_id']
    )
    
    doc = content.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.content.insert_one(doc)
    
    return content

@api_router.get("/content", response_model=List[Content])
async def get_all_content():
    content_list = await db.content.find({}, {"_id": 0}).to_list(1000)
    for item in content_list:
        if isinstance(item['created_at'], str):
            item['created_at'] = datetime.fromisoformat(item['created_at'])
    return content_list

@api_router.get("/content/{content_id}", response_model=Content)
async def get_content_by_id(content_id: str):
    content = await db.content.find_one({"id": content_id}, {"_id": 0})
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    
    if isinstance(content['created_at'], str):
        content['created_at'] = datetime.fromisoformat(content['created_at'])
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
                        # Update the database with the new data
                        update_data = {}
                        if not content.get('poster_path') and poster_path:
                            update_data['poster_path'] = poster_path
                        if not content.get('overview') and overview:
                            update_data['overview'] = overview
                        
                        if update_data:
                            await db.content.update_one(
                                {"id": item['content_id']},
                                {"$set": update_data}
                            )
                except:
                    pass
            
            content_with_details.append({
                "content_id": item['content_id'],
                "title": content.get('title', 'N/A'),
                "content_type": content.get('content_type', 'N/A'),
                "poster_path": poster_path,
                "overview": overview or '',
                "added_at": item.get('added_at', 'N/A')
            })
    
    return content_with_details

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
    for review in reviews:
        if isinstance(review['created_at'], str):
            review['created_at'] = datetime.fromisoformat(review['created_at'])
    return reviews

@api_router.delete("/admin/reviews/{review_id}")
async def delete_review(review_id: str, current_user: dict = Depends(get_admin_user)):
    review = await db.reviews.find_one({"id": review_id}, {"_id": 0})
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    await db.reviews.delete_one({"id": review_id})
    return {"message": "Review deleted successfully"}

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

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://edrastore.net",
        "http://edrastore.net",
        "https://www.edrastore.net",
        "http://www.edrastore.net",
        "http://localhost",
        "http://localhost:3000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "https://localhost",
        "https://localhost:3000",
    ],
    # Allow any subdomain and both http/https for edrastore.net (includes ports)
    # Also allow localhost and 127.0.0.1 with optional ports
    allow_origin_regex=r"^https?://([a-z0-9-]+\.)*edrastore\.net(:\d+)?$|^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    max_age=86400,
)

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