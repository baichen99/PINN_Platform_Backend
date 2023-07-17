from pydantic import BaseModel, Field
from config.common import CommonConfig
from pydantic import BaseModel
import torch
from typing import Optional


class JobCreate(BaseModel):
    name: str = Field(..., example="test")

class JobStart(BaseModel):
    ...