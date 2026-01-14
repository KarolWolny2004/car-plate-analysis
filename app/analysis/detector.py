import easyocr
import numpy as np
import cv2
import re
import os
from ultralytics import YOLO

print("Inicjalizacja modeli AI...")
reader = easyocr.Reader(['pl'], gpu=False)

# 1. Ładujemy Twój nowy model do tablic
# Upewnij się, że plik nazywa się 'plate_model.pt' i jest w tym samym folderze co detector.py
# Jeśli jest w innym miejscu, podaj pełną ścieżkę absolutną.
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')

if os.path.exists(MODEL_PATH):
    yolo_model = YOLO(MODEL_PATH)
    print(f"Załadowano model tablic: {MODEL_PATH}")
else:
    print("OSTRZEŻENIE: Nie znaleziono plate_model.pt, pobieram standardowy yolov8n.pt (będzie wykrywał auta, nie tablice!)")
    yolo_model = YOLO('yolov8n.pt')

print("Modele gotowe.")

DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

def crop_plate_directly(img_cv2):
    """
    Wycina tablicę rejestracyjną bezpośrednio z całego zdjęcia.
    Dodaje margines (padding), aby ułatwić pracę OCR.
    """
    # conf=0.25 - standardowy próg, można zwiększyć jeśli łapie śmieci
    results = yolo_model(img_cv2, conf=0.25, verbose=False)

    best_crop = None
    max_conf = 0

    h_img, w_img, _ = img_cv2.shape

    for r in results:
        for box in r.boxes:
            # W dedykowanym modelu tablica to zazwyczaj klasa 0.
            # Jeśli model wykrywa też auta, trzeba sprawdzić nazwę klasy.
            # Tutaj zakładamy, że model wykrywa tablice jako klasę interesującą nas.
            
            conf = float(box.conf[0])
            
            # Jeśli na zdjęciu jest kilka tablic, bierzemy tę z największą pewnością
            if conf > max_conf:
                max_conf = conf
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # --- DODAWANIE MARGINESU (PADDING) ---
                # OCR działa gorzej, gdy tekst dotyka krawędzi.
                pad_x = int((x2 - x1) * 0.05) # 5% szerokości
                pad_y = int((y2 - y1) * 0.10) # 10% wysokości

                # Upewniamy się, że nie wyjdziemy poza obraz
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w_img, x2 + pad_x)
                y2 = min(h_img, y2 + pad_y)

                best_crop = img_cv2[y1:y2, x1:x2]

    return best_crop

def validate_plate(text_list):
    """
    Wybiera z listy tekstów ten, który najbardziej przypomina tablicę rejestracyjną.
    """
    # Prosty regex dla polskich tablic (2-3 litery + 4-5 cyfr/liter)
    pattern = re.compile(r'^[A-Z]{2,3}[0-9A-Z]{4,5}$')
    
    potential_plates = []
    for text in text_list:
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        # Polskie tablice mają 7 lub 8 znaków (klasyczne)
        if 7 <= len(clean) <= 8:
            if pattern.match(clean):
                return clean
            potential_plates.append(clean)
            
    if potential_plates:
        return potential_plates[0]
        
    return None

def detect_plate(image_bytes: bytes, job_id: int = None) -> str:
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return "ERROR: Niepoprawny format obrazu"

        # KROK 1: Bezpośrednie wycięcie tablicy (YOLOv11)
        plate_img = crop_plate_directly(img)

        # Jeśli YOLO nie znalazło tablicy, przerywamy
        if plate_img is None:
            if job_id is not None:
                # Zapisujemy oryginał, żeby zobaczyć dlaczego nie znalazło
                cv2.imwrite(os.path.join(DEBUG_DIR, f"job_{job_id}_FAILED.jpg"), img)
            return "Nie wykryto tablicy (YOLO)"

        # Zapis podglądu wyciętej tablicy
        if job_id is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, f"job_{job_id}_crop.jpg"), plate_img)

        # KROK 2: Preprocessing
        # Dla EasyOCR warto spróbować skali szarości + lekkiego podbicia kontrastu
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Opcjonalnie: progowanie (binaryzacja), jeśli zdjęcia są słabe
        # _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # KROK 3: OCR
        # allowlist - ogranicza znaki tylko do liter i cyfr (przyspiesza i zmniejsza błędy)
        detected_texts = reader.readtext(gray, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

        if not detected_texts:
            # Fallback na kolorowy obraz
            detected_texts = reader.readtext(plate_img, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

        # KROK 4: Walidacja
        best_match = validate_plate(detected_texts)
        if best_match:
            return best_match

        # Jeśli walidacja nic nie dała, zwróćmy surowy ciąg znaków
        full_text = "".join(detected_texts).upper()
        return full_text if full_text else "Nie odczytano znaków"

    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        return f"ERROR: {str(e)}"