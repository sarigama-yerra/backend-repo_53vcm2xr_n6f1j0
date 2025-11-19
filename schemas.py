"""
Database Schemas for Incommon

Each Pydantic model represents a MongoDB collection. The collection name is
lowercase of the class name.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class User(BaseModel):
    """
    Users of Incommon
    Collection name: "user"
    """
    name: str = Field(..., description="Display name")
    email: str = Field(..., description="Email address")
    avatar_url: Optional[str] = Field(None, description="Profile picture URL")
    is_active: bool = Field(True, description="Whether user is active")


class Clip(BaseModel):
    """
    Short video posts
    Collection name: "clip"
    """
    caption: str = Field(..., description="Clip caption")
    video_url: str = Field(..., description="Public URL to MP4/MOV/WEBM video")
    like_count: int = Field(0, ge=0, description="Number of likes")
    comment_count: int = Field(0, ge=0, description="Number of comments")
    location_lat: Optional[float] = Field(None, description="Latitude")
    location_lng: Optional[float] = Field(None, description="Longitude")
    place_name: Optional[str] = Field(None, description="Human-readable place name")


class Comment(BaseModel):
    """
    Comments on clips
    Collection name: "comment"
    """
    clip_id: str = Field(..., description="Associated clip ID (stringified ObjectId)")
    author: Optional[str] = Field(None, description="Display name of commenter")
    text: str = Field(..., description="Comment text")
