import easyocr
import numpy as np
import cv2
import re
import os
from ultralytics import YOLO

print("Inicjalizacja modeli AI...")
reader = easyocr.Reader(['pl'], gpu=False, verbose=False)

MODEL_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../best.pt'))

if os.path.exists(MODEL_PATH):
    yolo_model = YOLO(MODEL_PATH)
    print(f"Załadowano model tablic: {MODEL_PATH}")
else:
    print("OSTRZEŻENIE: Nie znaleziono plate_model.pt, pobieram standardowy yolov8n.pt (będzie wykrywał auta, nie tablice!)")
    yolo_model = YOLO('yolov8n.pt')


DEBUG_DIR = "debug_failures"
os.makedirs(DEBUG_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def smart_postprocess(detected: str) -> str:
    detected = clean_text(detected)
    if len(detected) < 3:
        return detected

    chars = list(detected)
    MAX_LEN = 8

    for i, c in enumerate(chars):
        if i < 2:
            if c == '0': chars[i] = 'O'
            elif c == '1': chars[i] = 'I'
            elif c == '2': chars[i] = 'Z'
            elif c == '5': chars[i] = 'S'
            elif c == '6': chars[i] = 'G'
            elif c == '8': chars[i] = 'B'
            elif c == '4': chars[i] = 'A'
        else:
            if c in {'O', 'Q', 'D'}:
                chars[i] = '0'
            elif c == 'B': chars[i] = '8'
            elif c == 'Z': chars[i] = '2'

    result = "".join(chars)
    if len(result) > MAX_LEN:
        result = result[:MAX_LEN]

    return result

class PlateOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['pl'], gpu=False, verbose=False)

    def _cut_blue_strip(self, img: np.ndarray) -> np.ndarray:
        if img.size == 0:
            return img

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]

        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        scan_limit = int(w * 0.30)
        max_safe_crop = int(w * 0.18)

        cut_location = 0
        in_blue = False

        for x in range(scan_limit):
            density = np.count_nonzero(mask[:, x]) / h
            if density > 0.35:
                in_blue = True
                cut_location = x
            elif in_blue:
                break

        cut_location = min(cut_location, max_safe_crop)

        if cut_location > 0:
            return img[:, cut_location + 2:]

        return img

    def process_and_read(self, plate_bgr: np.ndarray) -> tuple[str, np.ndarray]:
        if plate_bgr is None or plate_bgr.size == 0:
            return "", plate_bgr

        h, w = plate_bgr.shape[:2]
        margin = 0.02
        plate = plate_bgr[
            int(h * margin):int(h * (1 - margin)),
            int(w * margin):int(w * (1 - margin))
        ]

        plate = self._cut_blue_strip(plate)

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        if gray.shape[0] < 60:
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        inv = cv2.bitwise_not(binary)
        contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ih, iw = binary.shape[:2]

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if (ch > ih * 0.85 and cw < iw * 0.08) or (cw > iw * 0.4) or (ch < ih * 0.3):
                cv2.drawContours(binary, [cnt], -1, 255, -1)

        kernel = np.ones((2, 1), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

        padded = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

        results = self.reader.readtext(
            padded,
            detail=0,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            decoder='greedy'
        )

        return "".join(results), padded


def detect_and_crop(img: np.ndarray, conf_threshold: float = 0.25) -> tuple[np.ndarray | None, list[int] | None]:
    """
    Wykryj tablicę rejestracyjną i przytnij ją z obrazu.
    Zwraca (obraz_tablicy, bbox).
    """
    results = yolo_model(img, conf=conf_threshold, verbose=False, imgsz=640)
    best_crop = None
    best_box = None
    max_conf = 0
    h_img, w_img, _ = img.shape

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > max_conf:
                max_conf = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best_box = [x1, y1, x2, y2]
                
                pad_x = int((x2 - x1) * 0.05)
                pad_y = int((y2 - y1) * 0.10)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w_img, x2 + pad_x)
                y2 = min(h_img, y2 + pad_y)
                
                best_crop = img[y1:y2, x1:x2]
                
    return best_crop, best_box


def detect_plate(image_bytes: bytes, job_id: int = None) -> str:
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return "ERROR: Niepoprawny format obrazu"

        plate_img, _ = detect_and_crop(img)

        if plate_img is None:
            if job_id is not None:
                cv2.imwrite(os.path.join(DEBUG_DIR, f"job_{job_id}_FAILED.jpg"), img)
            return "Nie wykryto tablicy (YOLO)"

        if job_id is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, f"job_{job_id}_crop.jpg"), plate_img)

        ocr_engine = PlateOCR()
        raw_text, processed_debug = ocr_engine.process_and_read(plate_img)
        final_text = smart_postprocess(raw_text)

        return final_text if final_text else "Nie odczytano znaków"

    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        return f"ERROR: {str(e)}"