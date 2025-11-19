import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bson import ObjectId

from database import db, create_document, get_documents
from schemas import Clip

app = FastAPI(title="Incommon API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# ------------------- Clips -------------------

@app.post("/api/clips", response_model=ClipPublic)
def create_clip(payload: ClipCreate):
    clip = Clip(
        caption=payload.caption,
        video_url=payload.video_url,
        location_lat=payload.location_lat,
        location_lng=payload.location_lng,
        place_name=payload.place_name,
    )
    inserted_id = create_document("clip", clip)
    return ClipPublic(
        id=inserted_id,
        caption=clip.caption,
        video_url=clip.video_url,
        like_count=clip.like_count,
        comment_count=clip.comment_count,
        location_lat=clip.location_lat,
        location_lng=clip.location_lng,
        place_name=clip.place_name,
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
    )


@app.get("/api/clips", response_model=List[ClipPublic])
def list_clips(
    lat: Optional[float] = Query(None),
    lng: Optional[float] = Query(None),
    radius_km: float = Query(10.0, alias="radiusKm"),
    limit: int = Query(20, ge=1, le=100),
):
    # If no location provided, return latest clips
    base_cursor = db["clip"].find({}).sort("created_at", -1).limit(limit)
    docs = list(base_cursor)

    if lat is None or lng is None:
        return [_serialize_clip(d) for d in docs]

    # Rough filter by bounding box first to reduce computation
    # 1 degree of lat ~ 111 km; lng scaling by cos(lat)
    lat_delta = radius_km / 111.0
    lng_delta = radius_km / max(0.0001, (111.0 * abs(__import__('math').cos(__import__('math').radians(lat)))))

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

    db["clip"].update_one({"_id": _id}, {"$inc": {"like_count": 1}, "$set": {"updated_at": __import__('datetime').datetime.utcnow()}})
    updated = db["clip"].find_one({"_id": _id})
    return _serialize_clip(updated)


# Simple schema endpoint for viewer tools
@app.get("/schema")
def get_schema():
    try:
        from schemas import User, Clip as ClipSchema, Comment
        # Return field descriptions
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
