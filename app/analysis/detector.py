import time
import random


def detect_plate(image_bytes: bytes):
    """
        Symuluje wykrywanie tablicy rejestracyjnej.
        Później zastąpimy to prawdziwym modelem AI (np. YOLO lub EasyOCR).
        """
    # Symulacja czasu trwania analizy (np. 1.5 sekundy)
    time.sleep(1.5)

    # Symulowane wyniki
    mock_plates = ["PO-12345", "W0-REBA", "KR-99887", "DW-55221"]
    return random.choice(mock_plates)