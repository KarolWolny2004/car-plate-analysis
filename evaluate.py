import os
import time
import cv2
import glob
import xml.etree.ElementTree as ET
import numpy as np
from ultralytics import YOLO
import re
from shapely.geometry import box as shapely_box
import shutil
import sys
from app.analysis.detector import (
    smart_postprocess,
    PlateOCR,
    detect_and_crop,
)


def load_yolo(model_path: str) -> YOLO:
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        print(f"BŁĄD: Model nie znaleziony w {model_path}")
        sys.exit(1)


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

    print("Inicjalizacja ewaluacji...")
    
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir, exist_ok=True)
    
    yolo_model = load_yolo(model_path)
    ocr_engine = PlateOCR()
    
    ground_truth = parse_ground_truth(annotations_file)
    images = glob.glob(os.path.join(val_images_dir, "*.jpg"))
    if not images:
        print("Nie znaleziono obrazów!")
        return
    
    test_samples = images
    while len(test_samples) < 100:
        test_samples += images
    test_samples = test_samples[:100]
    
    print(f"Rozpoczęcie ewaluacji.")
    
    correct_exact = 0
    iou_scores = []
    
    start_time = time.perf_counter()
    
    for i, path in enumerate(test_samples):
        sys.stdout.write(f"\rPrzetwarzanie: {i+1}/{len(test_samples)}")
        sys.stdout.flush()
        
        name = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            continue
        
        plate_crop, detected_box = detect_and_crop(img)
        
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

    total_time = time.perf_counter() - start_time
    print("\n")
    
    accuracy = (correct_exact / len(test_samples)) * 100
    avg_iou = np.mean(iou_scores) if iou_scores else 0
    time_per_100 = total_time * (100 / len(test_samples))
    grade = calculate_final_grade(accuracy, time_per_100)
    
    print("\n" + "="*50)
    print("RAPORT EWALUACJI")
    print("="*50)
    print(f"Czas przetwarzania (100 próbek): {time_per_100:.2f} s")
    print(f"Średnie IoU:                     {avg_iou:.4f}")
    print("-" * 50)
    print(f"Dokładność: {accuracy:.1f}% ({correct_exact}/{len(test_samples)})")
    print("-" * 50)
    print(f"Ostateczna ocena: {grade}")
    print("="*50)


if __name__ == "__main__":
    main()