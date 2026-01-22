import os
import time
import cv2
import glob
import xml.etree.ElementTree as ET
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from shapely.geometry import box as shapely_box
import shutil
import sys


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
            if c == '0':
                chars[i] = 'O'
            elif c == '1':
                chars[i] = 'I'
            elif c == '2':
                chars[i] = 'Z'
            elif c == '5':
                chars[i] = 'S'
            elif c == '6':
                chars[i] = 'G'
            elif c == '8':
                chars[i] = 'B'
            elif c == '4':
                chars[i] = 'A'
        else:
            if c in {'O', 'Q', 'D'}:
                chars[i] = '0'
            elif c == 'B':
                chars[i] = '8'
            elif c == 'Z':
                chars[i] = '2'

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


def load_yolo(model_path: str) -> YOLO:
    if os.path.exists(model_path):
        print(f"Loaded plate detection model: {model_path}")
        return YOLO(model_path)
    else:
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)


def detect_and_crop(model: YOLO, img: np.ndarray, conf_threshold: float = 0.25) -> tuple[np.ndarray | None, list[int] | None]:
    results = model(img, conf=conf_threshold, verbose=False, imgsz=640)
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


def get_iou(box_a: list[float], box_b: list[float]) -> float:
    if not box_a or not box_b:
        return 0.0
    poly_a = shapely_box(*box_a)
    poly_b = shapely_box(*box_b)
    intersection = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    if union == 0:
        return 0.0
    return intersection / union


def calculate_final_grade(accuracy: float, time_sec: float) -> float:
    if accuracy < 60:
        return 2.0
    if time_sec > 60:
        return 2.0
    acc_norm = (accuracy - 60) / 40
    time_norm = max(0, (60 - time_sec) / 50)
    score = 0.7 * acc_norm + 0.3 * time_norm
    grade = 2.0 + 3.0 * score
    return round(grade * 2) / 2


def parse_ground_truth(xml_path: str) -> dict:
    if not os.path.exists(xml_path):
        return {}
    tree = ET.parse(xml_path)
    gt_data = {}
    for image in tree.getroot().findall('image'):
        name = image.get('name')
        for box in image.findall('box'):
            if box.get('label') == 'plate':
                xtl, ytl = float(box.get('xtl')), float(box.get('ytl'))
                xbr, ybr = float(box.get('xbr')), float(box.get('ybr'))
                txt = box.find("attribute[@name='plate number']").text or ""
                gt_data[name] = {"bbox": [xtl, ytl, xbr, ybr], "text": txt}
                break
    return gt_data


def main(val_images_dir: str = None, annotations_file: str = None, model_path: str = None, debug_dir: str = "debug_failures"):
    if val_images_dir is None:
        val_images_dir = os.path.join("datasets", "polish_plates", "images", "val")
    if annotations_file is None:
        annotations_file = os.path.join("datasets", "polish_plates", "annotations.xml")
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'best.pt')

    print("Initializing evaluation...")
    
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir, exist_ok=True)
    
    yolo_model = load_yolo(model_path)
    ocr_engine = PlateOCR()
    
    ground_truth = parse_ground_truth(annotations_file)
    images = glob.glob(os.path.join(val_images_dir, "*.jpg"))
    if not images:
        print("No images found!")
        return
    
    test_samples = images
    while len(test_samples) < 100:
        test_samples += images
    test_samples = test_samples[:100]
    
    print(f"Starting evaluation on {len(test_samples)} samples...")
    
    correct_exact = 0
    iou_scores = []
    
    start_time = time.perf_counter()
    
    for i, path in enumerate(test_samples):
        sys.stdout.write(f"\rProcessing: {i+1}/{len(test_samples)}")
        sys.stdout.flush()
        
        name = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            continue
        
        plate_crop, detected_box = detect_and_crop(yolo_model, img)
        
        if name in ground_truth:
            gt_box = ground_truth[name]['bbox']
            iou_scores.append(get_iou(detected_box, gt_box) if detected_box else 0)
        
        final_text = "NOT_DETECTED"
        processed_debug = None
        
        if plate_crop is not None:
            raw_text, processed_debug = ocr_engine.process_and_read(plate_crop)
            final_text = smart_postprocess(raw_text)
        
        if name in ground_truth:
            gt_text = ground_truth[name]['text']
            
            if final_text == gt_text:
                correct_exact += 1
            else:
                # print(f"\nFAILURE [{i}]: Expected: {gt_text} | Got: {final_text}")
                
                safe_name = f"fail_{i}_{re.sub(r'[^A-Z0-9]', '', gt_text)}_vs_{re.sub(r'[^A-Z0-9]', '', final_text)}.jpg"
                try:
                    img_to_save = processed_debug if processed_debug is not None else (plate_crop if plate_crop is not None else img)
                    cv2.imwrite(os.path.join(debug_dir, safe_name), img_to_save)
                except Exception:
                    pass

    total_time = time.perf_counter() - start_time
    print("\n")
    
    accuracy = (correct_exact / len(test_samples)) * 100
    avg_iou = np.mean(iou_scores) if iou_scores else 0
    time_per_100 = total_time * (100 / len(test_samples))
    grade = calculate_final_grade(accuracy, time_per_100)
    
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(f"Processing time (100 samples): {time_per_100:.2f} s")
    print(f"Average IoU:                    {avg_iou:.4f}")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.1f}% ({correct_exact}/{len(test_samples)})")
    print("-" * 50)
    print(f"Final Grade: {grade}")
    print("="*50)
    print(f"Failure samples saved to: {debug_dir}")


if __name__ == "__main__":
    main()