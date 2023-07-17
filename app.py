from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
from api.database.db import create_db_and_tables

from api.routes.job import router as job_router

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    create_db_and_tables()

app.include_router(job_router, prefix="/job", tags=["job"])

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
