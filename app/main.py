from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.staticfiles import StaticFiles  # <--- NOWY IMPORT
from sqlalchemy.orm import Session
from app.storage import models, database
from app.worker import analyze_plate_task
import os  # <--- NOWY IMPORT

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(
    title="Car Plate Analysis API",
    description="API do analizy tablic rejestracyjnych",
    version="1.0.0"
)

# --- NOWE: Montowanie folderu debugowania ---
# Upewniamy się, że folder istnieje (na wypadek restartu aplikacji)
os.makedirs("debug_images", exist_ok=True)
# Udostępniamy pliki z folderu "debug_images" pod ścieżką URL "/debug"
app.mount("/debug", StaticFiles(directory="debug_images"), name="debug")
# --------------------------------------------

@app.get("/")
def read_root():
    return {
        "project": "Car Plate Analysis",
        "status": "Running",
        "message": "Witaj! Podgląd zdjęć dostępny pod /debug/"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/analyze-async")
async def analyze_image_async(
        file: UploadFile = File(...),
        db: Session = Depends(database.get_db)
):
    # 1. Odczytaj plik
    content = await file.read()

    # 2. Stwórz wstępny rekord w bazie
    new_job = models.DetectionResult(
        filename=file.filename,
        status="PENDING"
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    # 3. Wyślij zadanie do kolejki (przekazujemy ID i obraz jako tekst HEX)
    # .delay() sprawia, że funkcja nie wykonuje się teraz, tylko leci do Redisa
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