from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
from app.analysis.detector import detect_plate
from app.storage import models, database
# from typing import List

models.Base.metadata.create_all(bind=database.engine)
app = FastAPI(
    title="Car Plate Analysis API",
    description="API do analizy tablic rejestracyjnych (Zaawansowane Programowanie / Dobre Praktyki)",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {
        "project": "Car Plate Analysis",
        "status": "Running",
        "message": "Witaj w systemie analizy obrazu!"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/analyze-sync")
async def analyze_image_sync(
        file: UploadFile = File(...),
        db: Session = Depends(database.get_db)
):
    image_data = await file.read()
    plate_number = detect_plate(image_data)

    # ZAPIS DO BAZY:
    new_result = models.DetectionResult(
        filename=file.filename,
        plate_number=plate_number
    )
    db.add(new_result)
    db.commit()
    db.refresh(new_result)

    return {
        "id": new_result.id,
        "filename": new_result.filename,
        "detected_plate": new_result.plate_number,
        "saved_at": new_result.created_at
    }

@app.get("/results")
def get_all_results(db: Session = Depends(database.get_db)):
    """
    Pobiera wszystkie wyniki zapisane w bazie danych.
    """
    # Odpowiednik: SELECT * FROM detection_results;
    results = db.query(models.DetectionResult).all()
    return results