from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, File, UploadFile
from starlette.responses import FileResponse
from api.models.job import JobCreate
import pathlib
import uuid
from config.common import CommonConfig
from train.pinn import PINN
from models.MLP import MLP
from api.database.job import Job
from api.database.db import engine
import json
from sqlmodel import Session
import pandas as pd
import numpy as np


router = APIRouter()

@router.post("/")
def create_job(data: JobCreate):
    # 数据库创建job记录，创建对应文件夹
    with Session(engine) as session:
        # 检查是否有同名job
        job = session.query(Job).filter(Job.name == data.name).first()
        if job:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job already exists")
        job = Job(name=data.name, status="pending")
        session.add(job)
        session.commit()
        session.refresh(job)
    # make directory for job
    pathlib.Path(f"jobs_dir/{job.id}").mkdir(parents=True, exist_ok=False)
    pathlib.Path(f"jobs_dir/{job.id}/logs").mkdir(parents=True, exist_ok=False)
    return {"job_id": job.id}

@router.get("/config")
def get_default_config():
    config = CommonConfig()
    return config

@router.delete("/{job_id}")
def delete_job(job_id: int):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        session.delete(job)
        session.commit()
        # delete dir
        pathlib.Path(f"jobs_dir/{job_id}").rmdir()
    return {"message": "Job deleted"}

@router.get("/")
def get_jobs():
    with Session(engine) as session:
        jobs = session.query(Job).all()
        return jobs

@router.post("/upload/{job_id}")
def upload_files(job_id: int, file: UploadFile = File(...)):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        if file.filename in job.files:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File already exists")
        # save to jobs_dir/{job_id}
        contents = file.file.read()
        with open(f"jobs_dir/{job_id}/{file.filename}", "wb") as f:
            f.write(contents)
        # add to job.files
        
        job_files = json.loads(job.files)
        job_files.append(file.filename)
        job.files = json.dumps(job_files)
        session.add(job)
        session.commit()
        return {"filenames": job_files}

@router.get("/{job_id}")
def get_info(job_id: int):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        job.files = json.loads(job.files)
        return job
    
@router.post("/{job_id}/run")
def run_job(job_id: int, config: CommonConfig, background_tasks: BackgroundTasks):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        job.status = "running"
        session.add(job)
        session.commit()
        # 修改路径
        # 检查ic_data_path, bc_data_path, test_data_path, pde_data_path是否只是一个filename
        # 如果是，修改为jobs_dir/{job_id}/{filename}
        # 如果不是，不通过
        for attr in ["ic_data_path", "bc_data_path", "test_data_path", "pde_data_path"]:
            if getattr(config, attr) and pathlib.Path(getattr(config, attr)).name == getattr(config, attr):
                setattr(config, attr, f"jobs_dir/{job_id}/{getattr(config, attr)}")
        
        config.log_dir = f"jobs_dir/{job_id}/logs"
        
        model = MLP(config.net, config.activation_fn).to(config.device)
        def train_and_change_status():
            try:
                PINN(config, model).train()
                with Session(engine) as session:
                    job = session.get(Job, job_id)
                    if not job:
                        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
                    job.status = "completed"
                    session.add(job)
                    session.commit()
                    print("Job completed")
            except Exception as e:
                raise e
            
        background_tasks.add_task(
            train_and_change_status
        )
        # background_tasks.add_task(..., job_id)
        return {"message": "Job started"}
    
# delete file
@router.delete("/{job_id}/file/{filename}")
def delete_file(job_id: int, filename: str):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        if filename not in job.files:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
        # delete file
        pathlib.Path(f"jobs_dir/{job_id}/{filename}").unlink()
        # delete from job.files
        job_files = json.loads(job.files)
        job_files.remove(filename)
        job.files = json.dumps(job_files)
        session.add(job)
        session.commit()
        return

# download file
@router.get("/{job_id}/file/{filename}")
def download_file(job_id: int, filename: str):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        if filename not in job.files:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
        return FileResponse(f"jobs_dir/{job_id}/{filename}", filename=filename)

# get log
@router.get("/{job_id}/log/labels")
def get_log(job_id: int):
    log_path = f"jobs_dir/{job_id}/logs/log.csv"
    if not pathlib.Path(log_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log not found")
    df = pd.read_csv(log_path)
    labels = df.columns
    return {"labels": labels.to_list()}

@router.get("/{job_id}/log/{label}")
def get_log(job_id: int, label: str):
    log_path = f"jobs_dir/{job_id}/logs/log.csv"
    if not pathlib.Path(log_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log not found")
    df = pd.read_csv(log_path)
    # drop na
    df = df.dropna(how="any", axis=0, subset=[label])
    x = df["epoch"].to_list()
    y = df[label].to_list()
    return {"x": x, "y": y}