from celery import Celery
# import time
from app.analysis.detector import detect_plate
from app.storage.database import SessionLocal
from app.storage import models

celery_app = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

@celery_app.task(name="analyze_plate_task")
def analyze_plate_task(detection_id: int, image_hex: str):
    db = SessionLocal()
    record = None

    try:
        # 1. Znajdź rekord w bazie
        record = db.query(models.DetectionResult).filter(models.DetectionResult.id == detection_id).first()
        if not record:
            return "Record not found"

        # 2. Zmień status na PROCESSING
        record.status = "PROCESSING"
        db.commit()

        # 3. Wykonaj analizę
        image_bytes = bytes.fromhex(image_hex)
        
        # --- ZMIANA: Przekazujemy job_id=detection_id ---
        result_plate = detect_plate(image_bytes, job_id=detection_id)
        # ------------------------------------------------

        # 4. Zapisz wynik i zmień status na COMPLETED
        record.plate_number = result_plate
        record.status = "COMPLETED"
        db.commit()

        return f"Success: {result_plate}"
    except Exception as e:
        if record:
            record.status = "FAILED"
            db.commit()
        return f"Error: {str(e)}"
    finally:
        db.close()