from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from .database import Base

class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    plate_number = Column(String, nullable=True)
    status = Column(String, default="PENDING")
    created_at = Column(DateTime, default=func.now())