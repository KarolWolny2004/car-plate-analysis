import easyocr
import numpy as np
import cv2
import re
from ultralytics import YOLO

print("Inicjalizacja modeli AI...")
# 1. EasyOCR do czytania tekstu
reader = easyocr.Reader(['pl'], gpu=False)
# 2. YOLOv8 do detekcji samochodu (pobierze się automatycznie przy pierwszym użyciu)
yolo_model = YOLO('yolov8n.pt')
print("Modele gotowe.")

def crop_car_with_yolo(img_cv2):
    """
    Znajduje samochód na zdjęciu i wycina go, aby usunąć otoczenie (trawę, inne auta).
    """
    # conf=0.5 - bierzemy obiekty z pewnością > 50%
    results = yolo_model(img_cv2, conf=0.5, verbose=False)

    for r in results:
        for box in r.boxes:
            # Sprawdzamy, czy wykryty obiekt to pojazd
            cls_id = int(box.cls[0])
            class_name = yolo_model.names[cls_id]

            if class_name in ['car', 'truck', 'bus']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Wycinamy fragment zdjęcia
                return img_cv2[y1:y2, x1:x2]

    # Jeśli nie znaleziono auta, zwracamy oryginał (fallback)
    return img_cv2

def validate_plate(text_list):
    """
    Wybiera z listy tekstów ten, który najbardziej przypomina tablicę rejestracyjną.
    """
    # Wzorzec polskiej tablicy:
    # ^ - start
    # [A-Z]{2,3} - 2 lub 3 litery (np. PO, SCZ)
    # [0-9A-Z]{4,5} - 4 lub 5 znaków dalszych
    # $ - koniec
    pattern = re.compile(r'^[A-Z]{2,3}[0-9A-Z]{4,5}$')

    potential_plates = []

    for text in text_list:
        # Czyścimy ze spacji i znaków specjalnych
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Sprawdzamy długość (polskie tablice mają 7 lub 8 znaków)
        if 7 <= len(clean) <= 8:
            # Jeśli pasuje do ścisłego wzorca regex - to nasz faworyt
            if pattern.match(clean):
                return clean
            # Jeśli ma dobrą długość, ale nie pasuje do wzorca, dodajemy do potencjalnych
            potential_plates.append(clean)

    # Jeśli nie ma idealnego dopasowania, zwracamy pierwszy o dobrej długości
    if potential_plates:
        return potential_plates[0]

    return None

def detect_plate(image_bytes: bytes) -> str:
    try:
        # Dekodowanie obrazu
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return "ERROR: Niepoprawny format obrazu"

        # KROK 1: Wycięcie samochodu (YOLO)
        cropped_img = crop_car_with_yolo(img)

        # KROK 2: Preprocessing (opcjonalny, ale pomaga EasyOCR)
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        # Zwiększenie kontrastu
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_img = clahe.apply(gray)

        # KROK 3: Odczyt tekstu (detail=0 zwraca listę stringów)
        detected_texts = reader.readtext(contrast_img, detail=0)

        # Jeśli lista pusta, spróbujmy na oryginale wycinka (bez CLAHE)
        if not detected_texts:
            detected_texts = reader.readtext(cropped_img, detail=0)

        # KROK 4: Filtracja wyników (Logic Change)
        # Zamiast łączyć wszystko ("".join), szukamy konkretnej tablicy
        best_match = validate_plate(detected_texts)

        if best_match:
            return best_match

        # Fallback: Jeśli filtracja nic nie dała, zwracamy wszystko,
        # ale usuwamy nazwy salonów (twoje poprzednie rozwiązanie)
        full_text = "".join(detected_texts).upper()
        clean_text = re.sub(r'[^A-Z0-9]', '', full_text)

        # # Proste usuwanie znanych śmieci, jeśli RegEx zawiódł
        # junk = ["GLIWICE", "KELLER", "HYUNDAI", "PL"]
        # for j in junk:
        #     clean_text = clean_text.replace(j, "")

        return clean_text[:8] if clean_text else "Nie wykryto"

    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        return f"ERROR: {str(e)}"