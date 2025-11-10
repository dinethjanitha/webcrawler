from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from config.objectIdConterver import PyObjectId
from schema.sitesDataSchema import SiteDataOut
from schema.summarySchema import SummaryOut

# Custom type that automatically converts ObjectId to string

class FullSchema(BaseModel):
    keyword: str
    urls: Optional[list[str]] = []  # Optional with default empty list
    content: Optional[list[str]] = []  # Optional with default empty list
    summary: str
  
class FullSchemaOut(FullSchema):
    id: PyObjectId = Field(alias="_id")  # Auto-converts ObjectId to string
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True
    )