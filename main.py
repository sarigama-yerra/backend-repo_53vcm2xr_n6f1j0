import os
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from bson import ObjectId

from database import db, create_document
from schemas import Clip

# Auth settings
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret-incommon-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Use a hash algorithm that does not require external C extensions
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
# Make token optional for endpoints that accept anonymous users
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

app = FastAPI(title="Incommon API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static uploads directory
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# ------------------- Models -------------------
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserPublic(BaseModel):
    id: str
    email: EmailStr
    name: Optional[str] = None
    handle: Optional[str] = None
    avatar_url: Optional[str] = None


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None
    handle: Optional[str] = None


class ClipCreate(BaseModel):
    caption: str
    video_url: str
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None
    place_name: Optional[str] = None


class ClipPublic(BaseModel):
    id: str
    caption: str
    video_url: str
    like_count: int
    comment_count: int
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None
    place_name: Optional[str] = None
    author_handle: Optional[str] = None


class CommentCreate(BaseModel):
    author: Optional[str] = None
    text: str


class CommentPublic(BaseModel):
    id: str
    clip_id: str
    author: Optional[str] = None
    text: str


# ------------------- Auth Utils -------------------

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[dict]:
    if not token:
        return None
    if db is None:
        # If DB is not configured, treat as anonymous
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        doc = db["user"].find_one({"_id": ObjectId(user_id)})
        return doc
    except JWTError:
        return None


# ------------------- Basic -------------------
@app.get("/")
def read_root():
    return {"message": "Incommon backend running"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from Incommon API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Check environment variables
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# ------------------- Auth Endpoints -------------------
@app.post("/auth/register", response_model=UserPublic)
def register(user: UserCreate):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    if db["user"].find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    doc = {
        "email": str(user.email),
        "name": user.name,
        "handle": user.handle or user.email.split("@")[0],
        "password_hash": get_password_hash(user.password),
        "avatar_url": None,
        "is_active": True,
    }
    user_id = create_document("user", doc)
    return UserPublic(id=user_id, email=user.email, name=user.name, handle=doc["handle"], avatar_url=None)


@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    user = db["user"].find_one({"email": form_data.username})
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    if not verify_password(form_data.password, user.get("password_hash", "")):
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    token = create_access_token({"sub": str(user["_id"])})
    return Token(access_token=token)


@app.get("/auth/me", response_model=UserPublic)
def me(current_user: dict = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return UserPublic(
        id=str(current_user["_id"]),
        email=current_user["email"],
        name=current_user.get("name"),
        handle=current_user.get("handle"),
        avatar_url=current_user.get("avatar_url"),
    )


# ------------------- Uploads -------------------
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Accept a video file upload and return a public URL under /uploads"""
    allowed = {"video/mp4", "video/webm", "video/ogg", "application/octet-stream"}
    if file.content_type not in allowed:
        if not file.filename.lower().endswith((".mp4", ".webm", ".mov", ".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported file type")

    ext = os.path.splitext(file.filename)[1] or ".mp4"
    fname = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(UPLOAD_DIR, fname)

    try:
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save file")

    public_url = f"/uploads/{fname}"
    return {"url": public_url}


# ------------------- Clips -------------------
@app.post("/api/clips", response_model=ClipPublic)
def create_clip(payload: ClipCreate, current_user: dict = Depends(get_current_user)):
    author_handle = current_user.get("handle") if current_user else None
    clip = Clip(
        caption=payload.caption,
        video_url=payload.video_url,
        location_lat=payload.location_lat,
        location_lng=payload.location_lng,
        place_name=payload.place_name,
    )
    # persist + lightweight author handle
    data = clip.model_dump()
    if author_handle:
        data["author_handle"] = author_handle
    inserted_id = create_document("clip", data)
    return ClipPublic(
        id=inserted_id,
        caption=clip.caption,
        video_url=clip.video_url,
        like_count=clip.like_count,
        comment_count=clip.comment_count,
        location_lat=clip.location_lat,
        location_lng=clip.location_lng,
        place_name=clip.place_name,
        author_handle=author_handle,
    )


def _serialize_clip(doc: dict) -> ClipPublic:
    return ClipPublic(
        id=str(doc.get("_id")),
        caption=doc.get("caption"),
        video_url=doc.get("video_url"),
        like_count=doc.get("like_count", 0),
        comment_count=doc.get("comment_count", 0),
        location_lat=doc.get("location_lat"),
        location_lng=doc.get("location_lng"),
        place_name=doc.get("place_name"),
        author_handle=doc.get("author_handle"),
    )


@app.get("/api/clips", response_model=List[ClipPublic])
def list_clips(
    lat: Optional[float] = Query(None),
    lng: Optional[float] = Query(None),
    radius_km: float = Query(10.0, alias="radiusKm"),
    limit: int = Query(20, ge=1, le=100),
    north: Optional[float] = Query(None),
    south: Optional[float] = Query(None),
    east: Optional[float] = Query(None),
    west: Optional[float] = Query(None),
):
    # Bounds query (from map) takes precedence
    if all(v is not None for v in [north, south, east, west]):
        bbox_filter = {
            "location_lat": {"$gte": south, "$lte": north},
            "location_lng": {"$gte": west, "$lte": east},
        }
        docs = list(db["clip"].find(bbox_filter).sort("created_at", -1).limit(limit))
        return [_serialize_clip(d) for d in docs]

    # If no location provided, return latest clips
    base_cursor = db["clip"].find({}).sort("created_at", -1).limit(limit)
    docs = list(base_cursor)

    if lat is None or lng is None:
        return [_serialize_clip(d) for d in docs]

    import math
    lat_delta = radius_km / 111.0
    lng_delta = radius_km / max(0.0001, (111.0 * abs(math.cos(math.radians(lat)))))

    bbox_filter = {
        "location_lat": {"$gte": lat - lat_delta, "$lte": lat + lat_delta},
        "location_lng": {"$gte": lng - lng_delta, "$lte": lng + lng_delta},
    }

    docs = list(db["clip"].find(bbox_filter).sort("created_at", -1).limit(limit))

    return [_serialize_clip(d) for d in docs]


@app.post("/api/clips/{clip_id}/like", response_model=ClipPublic)
def like_clip(clip_id: str):
    try:
        _id = ObjectId(clip_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid clip id")

    doc = db["clip"].find_one({"_id": _id})
    if not doc:
        raise HTTPException(status_code=404, detail="Clip not found")

    db["clip"].update_one({"_id": _id}, {"$inc": {"like_count": 1}, "$set": {"updated_at": datetime.utcnow()}})
    updated = db["clip"].find_one({"_id": _id})
    return _serialize_clip(updated)


# ------------------- Comments -------------------
@app.get("/api/clips/{clip_id}/comments", response_model=List[CommentPublic])
def list_comments(clip_id: str, limit: int = Query(50, ge=1, le=200)):
    try:
        _id = ObjectId(clip_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid clip id")

    if not db["clip"].find_one({"_id": _id}):
        raise HTTPException(status_code=404, detail="Clip not found")

    docs = list(db["comment"].find({"clip_id": clip_id}).sort("created_at", 1).limit(limit))
    results: List[CommentPublic] = []
    for d in docs:
        results.append(CommentPublic(
            id=str(d.get("_id")),
            clip_id=d.get("clip_id"),
            author=d.get("author"),
            text=d.get("text"),
        ))
    return results


@app.post("/api/clips/{clip_id}/comments", response_model=CommentPublic)
def create_comment(clip_id: str, payload: CommentCreate, current_user: dict = Depends(get_current_user)):
    try:
        _id = ObjectId(clip_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid clip id")

    clip = db["clip"].find_one({"_id": _id})
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # prefer authenticated user's handle for author
    author = None
    if current_user:
        author = current_user.get("handle") or current_user.get("email")
    else:
        author = payload.author

    comment_doc = {
        "clip_id": clip_id,
        "author": author,
        "text": payload.text,
    }
    inserted_id = create_document("comment", comment_doc)

    db["clip"].update_one({"_id": _id}, {"$inc": {"comment_count": 1}, "$set": {"updated_at": datetime.utcnow()}})

    return CommentPublic(id=inserted_id, clip_id=clip_id, author=author, text=payload.text)


# Simple schema endpoint for viewer tools
@app.get("/schema")
def get_schema():
    try:
        from schemas import User, Clip as ClipSchema, Comment
        return {
            "user": User.model_json_schema(),
            "clip": ClipSchema.model_json_schema(),
            "comment": Comment.model_json_schema(),
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
