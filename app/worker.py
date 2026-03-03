from celery import Celery
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
        record = db.query(models.DetectionResult).filter(models.DetectionResult.id == detection_id).first()
        if not record:
            return "Record not found"

        record.status = "PROCESSING"
        db.commit()

        image_bytes = bytes.fromhex(image_hex)
        
        result_plate = detect_plate(image_bytes, job_id=detection_id)

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