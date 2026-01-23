from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from app.storage import models, database
from app.worker import analyze_plate_task
from app.analysis import detector
import os

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(
    title="Car Plate Analysis API",
    description="API do analizy tablic rejestracyjnych",
    version="1.0.0"
)

# os.makedirs("debug_images", exist_ok=True)
# app.mount("/debug", StaticFiles(directory="debug_images"), name="debug")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_image_sync(file: UploadFile = File(...)):
    try:
        content = await file.read()
        
        plate_number = detector.detect_plate(content)
        
        return {
            "filename": file.filename,
            "plate_number": plate_number,
            "status": "COMPLETED"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Błąd podczas przetwarzania obrazu: {str(e)}"
        )

@app.post("/analyze-async")
async def analyze_image_async(
        file: UploadFile = File(...),
        db: Session = Depends(database.get_db)
):
    content = await file.read()

    new_job = models.DetectionResult(
        filename=file.filename,
        status="PENDING"
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    analyze_plate_task.delay(new_job.id, content.hex())

    return {
        "message": "Zadanie przyjęte do kolejki",
        "job_id": new_job.id,
        "status": "PENDING"
    }

@app.get("/jobs/{job_id}")
def get_job_status(job_id: int, db: Session = Depends(database.get_db)):
    job = db.query(models.DetectionResult).filter(models.DetectionResult.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Zadanie nie istnieje")

    return {
        "job_id": job.id,
        "status": job.status,
        "result": job.plate_number if job.status == "COMPLETED" else None,
        "created_at": job.created_at
    }

@app.get("/results")
def get_all_results(db: Session = Depends(database.get_db)):
    results = db.query(models.DetectionResult).all()
    return results