from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from config.objectIdConterver import PyObjectId


# Custom type that automatically converts ObjectId to string

class Keyword(BaseModel):
    keyword: str
    urls: Optional[list[str]] = []  # Optional with default empty list

class KeywordOut(Keyword):
    id: PyObjectId = Field(alias="_id")  # Auto-converts ObjectId to string
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True
    )