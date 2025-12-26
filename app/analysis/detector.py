import easyocr
import numpy as np
import cv2
import re

reader = easyocr.Reader(['pl'], gpu=False)

def detect_plate(image_bytes: bytes) -> str:
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return "ERROR: Niepoprawny format obrazu"

        # 1. Konwersja do skali szarości
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Zwiększenie kontrastu (CLAHE - wyrównanie histogramu)
        # To pomaga "wyciągnąć" napisy z cienia bez robienia czarnych plam
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_img = clahe.apply(gray)

        # 3. Odczyt (tym razem pozwalamy EasyOCR na więcej swobody)
        # Zmieniamy parametry: contrast_ths i adjust_contrast
        results = reader.readtext(contrast_img, detail=0)

        if not results:
            return "Nie wykryto tekstu"

        # 4. Łączymy i czyścimy wynik za pomocą RegEx
        full_text = "".join(results).upper()

        # Usuwamy wszystko co nie jest literą lub cyfrą
        cleaned_text = re.sub(r'[^A-Z0-9]', '', full_text)

        return cleaned_text if cleaned_text else "Brak czytelnych znaków"

    except Exception as e:
        return f"ERROR: {str(e)}"