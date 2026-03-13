from pydantic import BaseModel

class ProductOut(BaseModel):
    id: int
    name: str
    price: float
    image_url: str

    model_config = {"from_attributes": True}

class RecommendationOut(BaseModel):
    recommendations: list[ProductOut]
