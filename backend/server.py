from fastapi import FastAPI, APIRouter, HTTPException, Request, Response, Depends, Query, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from json import JSONEncoder
import json
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import httpx
from google.auth.transport import requests
from google.oauth2 import id_token
import shutil

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Create uploads directory if it doesn't exist
UPLOADS_DIR = ROOT_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Custom JSON Encoder for ObjectId
class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

# Create the main app with custom JSON encoder
app = FastAPI(title="Laushop API")
app.json_encoder = CustomJSONEncoder

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== UTILITY FUNCTIONS ====================

def convert_objectid(obj):
    """Recursively convert ObjectId instances to strings in dictionaries and lists."""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_objectid(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid(item) for item in obj]
    return obj

# ==================== MODELS ====================

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    role: str = "customer"
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserUpdate(BaseModel):
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None

class ProductVariation(BaseModel):
    variation_id: str = Field(default_factory=lambda: f"var_{uuid.uuid4().hex[:8]}")
    name: str
    value: str
    price_modifier: float = 0
    stock: int = 0
    images: List[str] = []  # Imágenes específicas para esta variación

class Product(BaseModel):
    model_config = ConfigDict(extra="ignore")
    product_id: str = Field(default_factory=lambda: f"prod_{uuid.uuid4().hex[:12]}")
    name: str
    description: str
    original_price: float
    discount_price: Optional[float] = None
    cost_price: float = 0  # Precio de costo para calcular margen de ganancia
    category_id: str
    images: List[str] = []
    variations: List[ProductVariation] = []
    stock: int = 0
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ProductCreate(BaseModel):
    name: str
    description: str
    original_price: float
    discount_price: Optional[float] = None
    cost_price: float = 0  # Precio de costo
    category_id: str
    images: List[str] = []
    variations: List[Dict[str, Any]] = []
    stock: int = 0

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    original_price: Optional[float] = None
    discount_price: Optional[float] = None
    cost_price: Optional[float] = None  # Precio de costo
    category_id: Optional[str] = None
    images: Optional[List[str]] = None
    variations: Optional[List[Dict[str, Any]]] = None
    stock: Optional[int] = None
    is_active: Optional[bool] = None

class Category(BaseModel):
    model_config = ConfigDict(extra="ignore")
    category_id: str = Field(default_factory=lambda: f"cat_{uuid.uuid4().hex[:8]}")
    name: str
    description: Optional[str] = None
    image: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CategoryCreate(BaseModel):
    name: str
    description: Optional[str] = None
    image: Optional[str] = None

class CartItem(BaseModel):
    product_id: str
    variation_id: Optional[str] = None
    quantity: int = 1

class Cart(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cart_id: str = Field(default_factory=lambda: f"cart_{uuid.uuid4().hex[:12]}")
    user_id: str
    items: List[CartItem] = []
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Favorite(BaseModel):
    model_config = ConfigDict(extra="ignore")
    favorite_id: str = Field(default_factory=lambda: f"fav_{uuid.uuid4().hex[:12]}")
    user_id: str
    product_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Review(BaseModel):
    model_config = ConfigDict(extra="ignore")
    review_id: str = Field(default_factory=lambda: f"rev_{uuid.uuid4().hex[:12]}")
    product_id: str
    user_id: str
    user_name: str
    rating: int = Field(ge=1, le=5)
    comment: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ReviewCreate(BaseModel):
    product_id: str
    rating: int = Field(ge=1, le=5)
    comment: str

class OrderItem(BaseModel):
    product_id: str
    product_name: str
    product_image: Optional[str] = None
    variation_id: Optional[str] = None
    variation_name: Optional[str] = None
    quantity: int
    unit_price: float
    total_price: float

class Order(BaseModel):
    model_config = ConfigDict(extra="ignore")
    order_id: str = Field(default_factory=lambda: f"ord_{uuid.uuid4().hex[:12]}")
    user_id: str
    user_name: str
    user_email: str
    phone: str
    address: str
    city: str
    items: List[OrderItem] = []
    subtotal: float
    discount: float = 0
    total: float
    status: str = "pending"
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class OrderCreate(BaseModel):
    phone: str
    address: str
    city: str
    notes: Optional[str] = None
    promotion_code: Optional[str] = None

class Promotion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    promotion_id: str = Field(default_factory=lambda: f"promo_{uuid.uuid4().hex[:8]}")
    code: str
    description: str
    discount_type: str = "percentage"  # percentage or fixed
    discount_value: float
    min_purchase: float = 0
    max_uses: Optional[int] = None
    uses_count: int = 0
    is_active: bool = True
    start_date: datetime
    end_date: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PromotionCreate(BaseModel):
    code: str
    description: str
    discount_type: str = "percentage"
    discount_value: float
    min_purchase: float = 0
    max_uses: Optional[int] = None
    start_date: datetime
    end_date: datetime

# ==================== AUTH HELPERS ====================

async def get_current_user(request: Request) -> Optional[User]:
    """Get current user - optional, no validation required"""
    user_id = request.headers.get("X-User-ID")
    
    if not user_id:
        return None
    
    user = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    if not user:
        return None
    
    return User(**user)

async def get_admin_user(request: Request) -> Optional[User]:
    """Get current user and verify admin role"""
    user = await get_current_user(request)
    if user and user.role == "admin":
        return user
    return None

async def get_optional_user(request: Request) -> Optional[User]:
    """Get current user if authenticated, otherwise None"""
    try:
        return await get_current_user(request)
    except HTTPException:
        return None

# ==================== AUTH ENDPOINTS ====================

@api_router.post("/auth/login")
async def login(request: Request):
    """Login with email and password"""
    body = await request.json()
    email = body.get("email")
    password = body.get("password")
    
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    
    # Find user by email
    user = await db.users.find_one({"email": email}, {"_id": 0})
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Simple password check (in production, use bcrypt)
    stored_password = user.get("password")
    if not stored_password or stored_password != password:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Create cart if doesn't exist
    existing_cart = await db.carts.find_one({"user_id": user["user_id"]}, {"_id": 0})
    if not existing_cart:
        cart = {"cart_id": f"cart_{uuid.uuid4().hex[:12]}", "user_id": user["user_id"], "items": [], "updated_at": datetime.now(timezone.utc).isoformat()}
        await db.carts.insert_one(cart)
    
    return user

@api_router.post("/auth/register")
async def register(request: Request):
    """Register with email and password"""
    body = await request.json()
    name = body.get("name")
    email = body.get("email")
    password = body.get("password")
    
    if not name or not email or not password:
        raise HTTPException(status_code=400, detail="Name, email and password required")
    
    # Check if user already exists
    existing_user = await db.users.find_one({"email": email}, {"_id": 0})
    if existing_user:
        raise HTTPException(status_code=409, detail="Email already registered")
    
    # Create new user
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    new_user = {
        "user_id": user_id,
        "email": email,
        "name": name,
        "password": password,  # Store plaintext password (use bcrypt in production)
        "role": "customer",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.users.insert_one(new_user)
    
    # Create cart for new user
    cart = {"cart_id": f"cart_{uuid.uuid4().hex[:12]}", "user_id": user_id, "items": [], "updated_at": datetime.now(timezone.utc).isoformat()}
    await db.carts.insert_one(cart)
    
    # Return user data (without password)
    result = {
        "user_id": new_user["user_id"],
        "email": new_user["email"],
        "name": new_user["name"],
        "role": new_user["role"]
    }
    return result

@api_router.post("/auth/logout")
async def logout():
    """Logout"""
    return {"message": "Logged out successfully"}

# ==================== CATEGORY ENDPOINTS ====================

@api_router.get("/categories")
async def get_categories():
    """Get all active categories"""
    categories = await db.categories.find({"is_active": True}, {"_id": 0}).to_list(100)
    return convert_objectid(categories)

@api_router.get("/categories/all")
async def get_all_categories():
    """Get all categories"""
    categories = await db.categories.find({}, {"_id": 0}).to_list(100)
    return convert_objectid(categories)

@api_router.post("/categories")
async def create_category(data: CategoryCreate):
    """Create new category"""
    category = Category(**data.model_dump())
    doc = category.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.categories.insert_one(doc)
    # Fetch back without _id to avoid ObjectId serialization issues
    created_category = await db.categories.find_one({"category_id": category.category_id}, {"_id": 0})
    return convert_objectid(created_category)

@api_router.put("/categories/{category_id}")
async def update_category(category_id: str, data: Dict[str, Any]):
    """Update category"""
    result = await db.categories.update_one(
        {"category_id": category_id},
        {"$set": data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Category not found")
    category = await db.categories.find_one({"category_id": category_id}, {"_id": 0})
    return convert_objectid(category)

@api_router.delete("/categories/{category_id}")
async def delete_category(category_id: str):
    """Delete category"""
    result = await db.categories.delete_one({"category_id": category_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Category not found")
    return {"message": "Category deleted"}

# ==================== PRODUCT ENDPOINTS ====================

@api_router.get("/upload/test")
async def test_upload():
    """Test endpoint to verify uploads are working"""
    return {
        "status": "ok",
        "uploads_dir": str(UPLOADS_DIR),
        "uploads_dir_exists": UPLOADS_DIR.exists(),
        "uploads_dir_writable": os.access(UPLOADS_DIR, os.W_OK)
    }

@api_router.get("/products")
async def get_products(
    category_id: Optional[str] = None,
    search: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    page: int = 1,
    limit: int = 12
):
    """Get products with filters"""
    query = {"is_active": True}
    
    if category_id:
        query["category_id"] = category_id
    if search:
        query["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"description": {"$regex": search, "$options": "i"}}
        ]
    if min_price is not None:
        query["$and"] = query.get("$and", [])
        query["$and"].append({"$or": [{"discount_price": {"$gte": min_price}}, {"original_price": {"$gte": min_price}}]})
    if max_price is not None:
        query["$and"] = query.get("$and", [])
        query["$and"].append({"$or": [{"discount_price": {"$lte": max_price}}, {"original_price": {"$lte": max_price}}]})
    
    skip = (page - 1) * limit
    products = await db.products.find(query, {"_id": 0}).skip(skip).limit(limit).to_list(limit)
    total = await db.products.count_documents(query)
    
    return {"products": convert_objectid(products), "total": total, "page": page, "pages": (total + limit - 1) // limit}

@api_router.get("/products/featured")
async def get_featured_products():
    """Get featured products (discounted)"""
    products = await db.products.find(
        {"is_active": True, "discount_price": {"$ne": None}},
        {"_id": 0}
    ).limit(8).to_list(8)
    return convert_objectid(products)

@api_router.get("/products/all")
async def get_all_products():
    """Get all products"""
    products = await db.products.find({}, {"_id": 0}).to_list(1000)
    return convert_objectid(products)

@api_router.get("/products/{product_id}")
async def get_product(product_id: str):
    """Get single product"""
    product = await db.products.find_one({"product_id": product_id}, {"_id": 0})
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return convert_objectid(product)

@api_router.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload product image"""
    logger.info(f"Upload image endpoint called - File: {file.filename}, Type: {file.content_type}")
    
    try:
        # Validate file type
        allowed_types = ['image/jpeg', 'image/png', 'image/webp', 'image/gif']
        if file.content_type not in allowed_types:
            logger.warning(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="Invalid file type. Allowed: JPEG, PNG, WebP, GIF")
        
        # Validate file size (max 5MB)
        file_content = await file.read()
        file_size = len(file_content)
        logger.info(f"File size: {file_size} bytes")
        
        if file_size > 5 * 1024 * 1024:
            logger.warning(f"File too large: {file_size}")
            raise HTTPException(status_code=400, detail="File too large. Max size: 5MB")
        
        # Generate unique filename
        file_extension = Path(file.filename).suffix
        unique_filename = f"product_{uuid.uuid4().hex}{file_extension}"
        file_path = UPLOADS_DIR / unique_filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Return the URL to access the image
        image_url = f"/uploads/{unique_filename}"
        logger.info(f"Image uploaded successfully: {unique_filename}")
        
        return {
            "url": image_url,
            "filename": unique_filename,
            "size": file_size
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")

@api_router.post("/products")
async def create_product(data: ProductCreate):
    """Create new product"""
    product_data = data.model_dump()
    variations = []
    for v in product_data.get("variations", []):
        var = ProductVariation(**v)
        variations.append(var.model_dump())
    product_data["variations"] = variations
    
    product = Product(**product_data)
    doc = product.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    doc["updated_at"] = doc["updated_at"].isoformat()
    await db.products.insert_one(doc)
    # Fetch back without _id to avoid ObjectId serialization issues
    created_product = await db.products.find_one({"product_id": product.product_id}, {"_id": 0})
    return created_product

@api_router.put("/products/{product_id}")
async def update_product(product_id: str, data: ProductUpdate):
    """Update product"""
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}
    if "variations" in update_data:
        variations = []
        for v in update_data["variations"]:
            if "variation_id" not in v:
                v["variation_id"] = f"var_{uuid.uuid4().hex[:8]}"
            variations.append(v)
        update_data["variations"] = variations
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    result = await db.products.update_one(
        {"product_id": product_id},
        {"$set": update_data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Product not found")
    product = await db.products.find_one({"product_id": product_id}, {"_id": 0})
    return product

@api_router.delete("/products/{product_id}")
async def delete_product(product_id: str):
    """Delete product"""
    result = await db.products.delete_one({"product_id": product_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"message": "Product deleted"}

# ==================== CART ENDPOINTS ====================

@api_router.get("/cart")
async def get_cart(request: Request):
    """Get user cart with product details"""
    user = await get_current_user(request)
    
    # Use user_id from auth or create a temporary one
    user_id = user.user_id if user else f"anon_{uuid.uuid4().hex[:16]}"
    
    cart = await db.carts.find_one({"user_id": user_id}, {"_id": 0})
    if not cart:
        cart = {"cart_id": f"cart_{uuid.uuid4().hex[:12]}", "user_id": user_id, "items": [], "updated_at": datetime.now(timezone.utc).isoformat()}
        await db.carts.insert_one(cart)
    
    # Enrich with product details
    enriched_items = []
    for item in cart.get("items", []):
        product = await db.products.find_one({"product_id": item["product_id"]}, {"_id": 0})
        if product:
            variation = None
            if item.get("variation_id"):
                for v in product.get("variations", []):
                    if v.get("variation_id") == item["variation_id"]:
                        variation = v
                        break
            enriched_items.append({
                **item,
                "product": product,
                "variation": variation
            })
    
    cart["items"] = enriched_items
    return convert_objectid(cart)

@api_router.post("/cart/add")
async def add_to_cart(item: CartItem, request: Request):
    """Add item to cart"""
    user = await get_current_user(request)
    user_id = user.user_id if user else f"anon_{uuid.uuid4().hex[:16]}"
    
    cart = await db.carts.find_one({"user_id": user_id}, {"_id": 0})
    if not cart:
        cart = {"cart_id": f"cart_{uuid.uuid4().hex[:12]}", "user_id": user_id, "items": [], "updated_at": datetime.now(timezone.utc).isoformat()}
        await db.carts.insert_one(cart)
    
    items = cart.get("items", [])
    found = False
    for i, existing in enumerate(items):
        if existing["product_id"] == item.product_id and existing.get("variation_id") == item.variation_id:
            items[i]["quantity"] += item.quantity
            found = True
            break
    
    if not found:
        items.append(item.model_dump())
    
    await db.carts.update_one(
        {"user_id": user_id},
        {"$set": {"items": items, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    return {"message": "Item added to cart"}

@api_router.put("/cart/update")
async def update_cart_item(item: CartItem, request: Request):
    """Update cart item quantity"""
    user = await get_current_user(request)
    user_id = user.user_id if user else f"anon_{uuid.uuid4().hex[:16]}"
    cart = await db.carts.find_one({"user_id": user_id}, {"_id": 0})
    if not cart:
        raise HTTPException(status_code=404, detail="Cart not found")
    
    items = cart.get("items", [])
    for i, existing in enumerate(items):
        if existing["product_id"] == item.product_id and existing.get("variation_id") == item.variation_id:
            if item.quantity <= 0:
                items.pop(i)
            else:
                items[i]["quantity"] = item.quantity
            break
    
    await db.carts.update_one(
        {"user_id": user_id},
        {"$set": {"items": items, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    return {"message": "Cart updated"}

@api_router.delete("/cart/remove/{product_id}")
async def remove_from_cart(product_id: str, variation_id: Optional[str] = None, request: Request = None):
    """Remove item from cart"""
    user = None
    user_id = f"anon_{uuid.uuid4().hex[:16]}"
    if request:
        user = await get_current_user(request)
        user_id = user.user_id if user else user_id
    cart = await db.carts.find_one({"user_id": user_id}, {"_id": 0})
    if not cart:
        raise HTTPException(status_code=404, detail="Cart not found")
    
    items = [i for i in cart.get("items", []) if not (i["product_id"] == product_id and i.get("variation_id") == variation_id)]
    
    await db.carts.update_one(
        {"user_id": user_id},
        {"$set": {"items": items, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    return {"message": "Item removed from cart"}

@api_router.delete("/cart/clear")
async def clear_cart(request: Request = None):
    """Clear cart"""
    user = None
    user_id = f"anon_{uuid.uuid4().hex[:16]}"
    if request:
        user = await get_current_user(request)
        user_id = user.user_id if user else user_id
    await db.carts.update_one(
        {"user_id": user_id},
        {"$set": {"items": [], "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    return {"message": "Cart cleared"}

# ==================== FAVORITES ENDPOINTS ====================

@api_router.get("/favorites")
async def get_favorites(request: Request = None):
    """Get user favorites with product details"""
    user = None
    user_id = f"anon_{uuid.uuid4().hex[:16]}"
    if request:
        user = await get_current_user(request)
        user_id = user.user_id if user else user_id
    
    favorites = await db.favorites.find({"user_id": user_id}, {"_id": 0}).to_list(100)
    enriched = []
    for fav in favorites:
        product = await db.products.find_one({"product_id": fav["product_id"]}, {"_id": 0})
        if product:
            enriched.append({**fav, "product": product})
    return enriched

@api_router.post("/favorites/{product_id}")
async def add_favorite(product_id: str, request: Request = None):
    """Add product to favorites"""
    user = None
    user_id = f"anon_{uuid.uuid4().hex[:16]}"
    if request:
        user = await get_current_user(request)
        user_id = user.user_id if user else user_id
    
    existing = await db.favorites.find_one({"user_id": user_id, "product_id": product_id})
    if existing:
        return {"message": "Already in favorites"}
    
    fav = Favorite(user_id=user_id, product_id=product_id)
    doc = fav.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.favorites.insert_one(doc)
    return {"message": "Added to favorites"}

@api_router.delete("/favorites/{product_id}")
async def remove_favorite(product_id: str, request: Request = None):
    """Remove product from favorites"""
    user = None
    user_id = f"anon_{uuid.uuid4().hex[:16]}"
    if request:
        user = await get_current_user(request)
        user_id = user.user_id if user else user_id
    
    result = await db.favorites.delete_one({"user_id": user_id, "product_id": product_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not in favorites")
    return {"message": "Removed from favorites"}

# ==================== REVIEWS ENDPOINTS ====================

@api_router.get("/reviews/{product_id}")
async def get_product_reviews(product_id: str):
    """Get reviews for a product"""
    reviews = await db.reviews.find({"product_id": product_id}, {"_id": 0}).to_list(100)
    return reviews

@api_router.post("/reviews")
async def create_review(data: ReviewCreate, request: Request = None):
    """Create a review"""
    user = None
    user_id = f"anon_{uuid.uuid4().hex[:16]}"
    user_name = "Anonymous"
    if request:
        user = await get_current_user(request)
        user_id = user.user_id if user else user_id
        user_name = user.name if user else "Anonymous"
    
    existing = await db.reviews.find_one({"product_id": data.product_id, "user_id": user_id})
    if existing:
        raise HTTPException(status_code=400, detail="You already reviewed this product")
    
    review = Review(
        product_id=data.product_id,
        user_id=user_id,
        user_name=user_name,
        rating=data.rating,
        comment=data.comment
    )
    doc = review.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.reviews.insert_one(doc)
    # Fetch back without _id to avoid ObjectId serialization issues
    created_review = await db.reviews.find_one({"review_id": review.review_id}, {"_id": 0})
    return created_review

@api_router.delete("/reviews/{review_id}")
async def delete_review(review_id: str, request: Request = None):
    """Delete own review"""
    user = None
    user_id = f"anon_{uuid.uuid4().hex[:16]}"
    if request:
        user = await get_current_user(request)
        user_id = user.user_id if user else user_id
    
    result = await db.reviews.delete_one({"review_id": review_id, "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Review not found or not yours")
    return {"message": "Review deleted"}

# ==================== ORDERS ENDPOINTS ====================

@api_router.post("/orders")
async def create_order(data: OrderCreate, request: Request = None):
    """Create order from cart"""
    user = None
    user_id = f"anon_{uuid.uuid4().hex[:16]}"
    if request:
        user = await get_current_user(request)
        user_id = user.user_id if user else user_id
    
    cart = await db.carts.find_one({"user_id": user_id}, {"_id": 0})
    if not cart or not cart.get("items"):
        raise HTTPException(status_code=400, detail="Cart is empty")
    
    # Calculate totals and prepare stock updates
    order_items = []
    subtotal = 0
    stock_updates = []  # Track all stock updates to rollback if needed
    
    for item in cart["items"]:
        product = await db.products.find_one({"product_id": item["product_id"]}, {"_id": 0})
        if not product:
            continue
        
        unit_price = product.get("discount_price") or product["original_price"]
        variation_name = None
        variation_id = item.get("variation_id")
        quantity_to_reduce = item["quantity"]
        
        # Check and update stock for variations or base product
        if variation_id:
            for v in product.get("variations", []):
                if v.get("variation_id") == variation_id:
                    if v.get("stock", 0) < quantity_to_reduce:
                        raise HTTPException(status_code=400, detail=f"Stock insuficiente para {product['name']} - {v.get('name')}: {v.get('value')}")
                    
                    unit_price += v.get("price_modifier", 0)
                    variation_name = f"{v.get('name')}: {v.get('value')}"
                    stock_updates.append({
                        "type": "variation",
                        "product_id": product["product_id"],
                        "variation_id": variation_id,
                        "quantity": quantity_to_reduce
                    })
                    break
        else:
            if product.get("stock", 0) < quantity_to_reduce:
                raise HTTPException(status_code=400, detail=f"Stock insuficiente para {product['name']}")
            stock_updates.append({
                "type": "product",
                "product_id": product["product_id"],
                "quantity": quantity_to_reduce
            })
        
        total_price = unit_price * item["quantity"]
        subtotal += total_price
        
        # Get product image - safely extract first image if exists
        product_image = None
        images = product.get("images", [])
        if images and len(images) > 0 and images[0]:
            product_image = images[0]
        
        order_items.append(OrderItem(
            product_id=item["product_id"],
            product_name=product["name"],
            product_image=product_image,
            variation_id=item.get("variation_id"),
            variation_name=variation_name,
            quantity=item["quantity"],
            unit_price=unit_price,
            total_price=total_price
        ).model_dump())
    
    # Apply promotion
    discount = 0
    if data.promotion_code:
        promo = await db.promotions.find_one({
            "code": data.promotion_code.upper(),
            "is_active": True
        }, {"_id": 0})
        if promo:
            now = datetime.now(timezone.utc)
            start = datetime.fromisoformat(promo["start_date"]) if isinstance(promo["start_date"], str) else promo["start_date"]
            end = datetime.fromisoformat(promo["end_date"]) if isinstance(promo["end_date"], str) else promo["end_date"]
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            
            if start <= now <= end and subtotal >= promo.get("min_purchase", 0):
                if promo.get("max_uses") is None or promo.get("uses_count", 0) < promo["max_uses"]:
                    if promo["discount_type"] == "percentage":
                        discount = subtotal * (promo["discount_value"] / 100)
                    else:
                        discount = promo["discount_value"]
                    await db.promotions.update_one(
                        {"promotion_id": promo["promotion_id"]},
                        {"$inc": {"uses_count": 1}}
                    )
    
    total = subtotal - discount
    
    order = Order(
        user_id=user_id,
        user_name=user.name if user else "Customer",
        user_email=user.email if user else "guest@example.com",
        phone=data.phone,
        address=data.address,
        city=data.city,
        items=order_items,
        subtotal=subtotal,
        discount=discount,
        total=total,
        notes=data.notes
    )
    
    # Update stock for all items
    for update in stock_updates:
        if update["type"] == "variation":
            await db.products.update_one(
                {"product_id": update["product_id"], "variations.variation_id": update["variation_id"]},
                {"$inc": {"variations.$.stock": -update["quantity"]}}
            )
        else:
            await db.products.update_one(
                {"product_id": update["product_id"]},
                {"$inc": {"stock": -update["quantity"]}}
            )
    
    doc = order.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.orders.insert_one(doc)
    
    # Clear cart
    await db.carts.update_one(
        {"user_id": user_id},
        {"$set": {"items": [], "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    
    # Fetch back without _id to avoid ObjectId serialization issues
    created_order = await db.orders.find_one({"order_id": order.order_id}, {"_id": 0})
    return created_order

@api_router.get("/orders")
async def get_user_orders(request: Request = None):
    """Get user orders"""
    user = None
    user_id = f"anon_{uuid.uuid4().hex[:16]}"
    if request:
        user = await get_current_user(request)
        user_id = user.user_id if user else user_id
    
    orders = await db.orders.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1).to_list(100)
    return convert_objectid(orders)

@api_router.get("/orders/all")
async def get_all_orders(
    status: Optional[str] = None,
    page: int = 1,
    limit: int = 20
):
    """Get all orders"""
    query = {}
    if status:
        query["status"] = status
    
    skip = (page - 1) * limit
    orders = await db.orders.find(query, {"_id": 0}).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
    total = await db.orders.count_documents(query)
    
    return {"orders": convert_objectid(orders), "total": total, "page": page, "pages": (total + limit - 1) // limit}

@api_router.put("/orders/{order_id}/status")
async def update_order_status(order_id: str, status: str, request: Request = None):
    """Update order status"""
    # Get current order to check previous status
    order = await db.orders.find_one({"order_id": order_id}, {"_id": 0})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # If changing to cancelled, return stock to products
    if status == "cancelled" and order.get("status") != "cancelled":
        for item in order.get("items", []):
            if item.get("variation_id"):
                # Return stock for variation
                await db.products.update_one(
                    {"product_id": item["product_id"], "variations.variation_id": item["variation_id"]},
                    {"$inc": {"variations.$.stock": item["quantity"]}}
                )
            else:
                # Return stock for base product
                await db.products.update_one(
                    {"product_id": item["product_id"]},
                    {"$inc": {"stock": item["quantity"]}}
                )
    
    # If changing from cancelled to another status, reduce stock again
    elif status != "cancelled" and order.get("status") == "cancelled":
        for item in order.get("items", []):
            if item.get("variation_id"):
                # Reduce stock for variation
                await db.products.update_one(
                    {"product_id": item["product_id"], "variations.variation_id": item["variation_id"]},
                    {"$inc": {"variations.$.stock": -item["quantity"]}}
                )
            else:
                # Reduce stock for base product
                await db.products.update_one(
                    {"product_id": item["product_id"]},
                    {"$inc": {"stock": -item["quantity"]}}
                )
    
    result = await db.orders.update_one(
        {"order_id": order_id},
        {"$set": {"status": status}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Order not found")
    return {"message": "Status updated"}

# ==================== PROMOTIONS ENDPOINTS ====================

@api_router.get("/promotions")
async def get_active_promotions():
    """Get active promotions"""
    now = datetime.now(timezone.utc)
    promotions = await db.promotions.find({
        "is_active": True,
        "start_date": {"$lte": now.isoformat()},
        "end_date": {"$gte": now.isoformat()}
    }, {"_id": 0}).to_list(100)
    return convert_objectid(promotions)

@api_router.get("/promotions/all")
async def get_all_promotions():
    """Get all promotions"""
    promotions = await db.promotions.find({}, {"_id": 0}).to_list(100)
    return convert_objectid(promotions)

@api_router.post("/promotions")
async def create_promotion(data: PromotionCreate):
    """Create promotion"""
    promo = Promotion(**data.model_dump())
    doc = promo.model_dump()
    doc["code"] = doc["code"].upper()
    doc["created_at"] = doc["created_at"].isoformat()
    doc["start_date"] = doc["start_date"].isoformat()
    doc["end_date"] = doc["end_date"].isoformat()
    await db.promotions.insert_one(doc)
    # Fetch back without _id to avoid ObjectId serialization issues
    created_promo = await db.promotions.find_one({"promotion_id": promo.promotion_id}, {"_id": 0})
    return created_promo

@api_router.put("/promotions/{promotion_id}")
async def update_promotion(promotion_id: str, data: Dict[str, Any]):
    """Update promotion"""
    if "code" in data:
        data["code"] = data["code"].upper()
    if "start_date" in data and isinstance(data["start_date"], datetime):
        data["start_date"] = data["start_date"].isoformat()
    if "end_date" in data and isinstance(data["end_date"], datetime):
        data["end_date"] = data["end_date"].isoformat()
    
    result = await db.promotions.update_one(
        {"promotion_id": promotion_id},
        {"$set": data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Promotion not found")
    promo = await db.promotions.find_one({"promotion_id": promotion_id}, {"_id": 0})
    return promo

@api_router.delete("/promotions/{promotion_id}")
async def delete_promotion(promotion_id: str):
    """Delete promotion"""
    result = await db.promotions.delete_one({"promotion_id": promotion_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Promotion not found")
    return {"message": "Promotion deleted"}

@api_router.post("/promotions/validate")
async def validate_promotion(code: str, subtotal: float):
    """Validate promotion code"""
    promo = await db.promotions.find_one({"code": code.upper(), "is_active": True}, {"_id": 0})
    if not promo:
        raise HTTPException(status_code=404, detail="Invalid promotion code")
    
    now = datetime.now(timezone.utc)
    start = datetime.fromisoformat(promo["start_date"]) if isinstance(promo["start_date"], str) else promo["start_date"]
    end = datetime.fromisoformat(promo["end_date"]) if isinstance(promo["end_date"], str) else promo["end_date"]
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    
    if now < start or now > end:
        raise HTTPException(status_code=400, detail="Promotion not active")
    
    if subtotal < promo.get("min_purchase", 0):
        raise HTTPException(status_code=400, detail=f"Minimum purchase ${promo['min_purchase']} required")
    
    if promo.get("max_uses") and promo.get("uses_count", 0) >= promo["max_uses"]:
        raise HTTPException(status_code=400, detail="Promotion limit reached")
    
    discount = 0
    if promo["discount_type"] == "percentage":
        discount = subtotal * (promo["discount_value"] / 100)
    else:
        discount = promo["discount_value"]
    
    return {"valid": True, "discount": discount, "promotion": promo}

# ==================== USERS ENDPOINTS (ADMIN) ====================

@api_router.get("/users")
async def get_users(
    page: int = 1,
    limit: int = 20,
    search: Optional[str] = None
):
    """Get all users"""
    query = {}
    if search:
        query["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"email": {"$regex": search, "$options": "i"}}
        ]
    
    skip = (page - 1) * limit
    users = await db.users.find(query, {"_id": 0}).skip(skip).limit(limit).to_list(limit)
    total = await db.users.count_documents(query)
    
    return {"users": users, "total": total, "page": page, "pages": (total + limit - 1) // limit}

@api_router.put("/users/{user_id}/role")
async def update_user_role(user_id: str, role: str):
    """Update user role"""
    if role not in ["customer", "admin"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    result = await db.users.update_one(
        {"user_id": user_id},
        {"$set": {"role": role}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "Role updated"}

@api_router.put("/profile")
async def update_profile(data: UserUpdate, request: Request = None):
    """Update user profile"""
    user = None
    user_id = f"anon_{uuid.uuid4().hex[:16]}"
    if request:
        user = await get_current_user(request)
        user_id = user.user_id if user else user_id
    
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}
    await db.users.update_one(
        {"user_id": user_id},
        {"$set": update_data}
    )
    updated_user = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    return updated_user

# ==================== DASHBOARD STATS (ADMIN) ====================

@api_router.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    # Total products
    total_products = await db.products.count_documents({"is_active": True})
    
    # Total users
    total_users = await db.users.count_documents({})
    
    # Total orders
    total_orders = await db.orders.count_documents({})
    
    # Orders by status
    pending_orders = await db.orders.count_documents({"status": "pending"})
    completed_orders = await db.orders.count_documents({"status": "completed"})
    
    # Revenue calculation
    pipeline = [
        {"$match": {"status": {"$in": ["pending", "completed", "shipped"]}}},
        {"$group": {"_id": None, "total": {"$sum": "$total"}}}
    ]
    revenue_result = await db.orders.aggregate(pipeline).to_list(1)
    total_revenue = revenue_result[0]["total"] if revenue_result else 0
    
    # Average profit margin calculation
    # Get all products with cost_price to calculate margin
    profit_pipeline = [
        {"$match": {"cost_price": {"$gt": 0}}},
        {"$group": {
            "_id": None,
            "avg_margin": {
                "$avg": {
                    "$multiply": [
                        {"$divide": [
                            {"$subtract": ["$original_price", "$cost_price"]},
                            "$original_price"
                        ]},
                        100
                    ]
                }
            }
        }}
    ]
    profit_result = await db.products.aggregate(profit_pipeline).to_list(1)
    avg_profit_margin = profit_result[0]["avg_margin"] if profit_result else 0
    
    # Calculate total earnings (revenue - cost of goods sold)
    # For orders with cost tracking, calculate actual profit
    all_orders = await db.orders.find({"status": {"$in": ["pending", "completed", "shipped"]}}, {"_id": 0}).to_list(1000)
    total_earnings = 0
    
    for order in all_orders:
        order_revenue = order.get("total", 0)
        order_cost = 0
        
        # Calculate cost of goods for this order
        for item in order.get("items", []):
            product = await db.products.find_one({"product_id": item.get("product_id")}, {"_id": 0})
            if product:
                cost_price = product.get("cost_price", 0)
                quantity = item.get("quantity", 0)
                order_cost += (cost_price * quantity)
        
        # Earnings = revenue - cost
        total_earnings += (order_revenue - order_cost)
    
    # Recent orders
    recent_orders = await db.orders.find({}, {"_id": 0}).sort("created_at", -1).limit(5).to_list(5)
    
    # Sales and earnings by day (last 30 days for more data flexibility)
    thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    sales_pipeline = [
        {"$match": {"created_at": {"$gte": thirty_days_ago}}},
        {"$group": {
            "_id": {"$substr": ["$created_at", 0, 10]},
            "count": {"$sum": 1},
            "revenue": {"$sum": "$total"}
        }},
        {"$sort": {"_id": 1}}
    ]
    sales_by_day = await db.orders.aggregate(sales_pipeline).to_list(30)
    
    # Calculate earnings by day (revenue - cost of goods sold)
    earnings_by_day = {}
    orders_by_date = await db.orders.find({"created_at": {"$gte": thirty_days_ago}}, {"_id": 0}).to_list(1000)
    
    for order in orders_by_date:
        date_key = order.get("created_at", "")[:10]
        order_revenue = order.get("total", 0)
        order_cost = 0
        
        # Calculate cost for this order
        for item in order.get("items", []):
            product = await db.products.find_one({"product_id": item.get("product_id")}, {"_id": 0})
            if product:
                cost_price = product.get("cost_price", 0)
                quantity = item.get("quantity", 0)
                order_cost += (cost_price * quantity)
        
        order_earnings = order_revenue - order_cost
        
        if date_key not in earnings_by_day:
            earnings_by_day[date_key] = 0
        earnings_by_day[date_key] += order_earnings
    
    # Convert to list format and merge with sales_by_day
    for sale in sales_by_day:
        sale["earnings"] = earnings_by_day.get(sale["_id"], 0)
    
    # Top products
    top_products_pipeline = [
        {"$unwind": "$items"},
        {"$group": {
            "_id": "$items.product_id",
            "name": {"$first": "$items.product_name"},
            "total_sold": {"$sum": "$items.quantity"},
            "revenue": {"$sum": "$items.total_price"}
        }},
        {"$sort": {"total_sold": -1}},
        {"$limit": 5}
    ]
    top_products = await db.orders.aggregate(top_products_pipeline).to_list(5)
    
    return {
        "total_products": total_products,
        "total_users": total_users,
        "total_orders": total_orders,
        "pending_orders": pending_orders,
        "completed_orders": completed_orders,
        "total_revenue": total_revenue,
        "total_earnings": total_earnings,
        "avg_profit_margin": avg_profit_margin,
        "recent_orders": recent_orders,
        "sales_by_day": sales_by_day,
        "top_products": top_products
    }

# ==================== ROOT ====================

@api_router.get("/")
async def root():
    return {"message": "Laushop API"}

# Include the router
app.include_router(api_router)

# Mount static files directory for uploaded images
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
