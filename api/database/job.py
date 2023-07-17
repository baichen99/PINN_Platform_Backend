from sqlmodel import Field, SQLModel
from typing import Optional
from enum import Enum
from datetime import datetime

# define status enum
class JobStatus(Enum):
    pending = "pending"
    running = "running"
    failed = "failed"
    completed = "completed"

class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    status: JobStatus
    files: str = '[]'
    
    created_at: Optional[str] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[str] = Field(default=None)
    completed_at: Optional[str] = Field(default=None)
    failed_at: Optional[str] = Field(default=None)
    
