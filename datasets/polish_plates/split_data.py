import os
import shutil
import random
from pathlib import Path


def split_dataset(base_dir, val_ratio=0.3):
    base_path = Path(base_dir)
    images_dir = base_path / "images"
    labels_dir = base_path / "labels"
    
    for split in ['train', 'val']:
        (base_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    all_images = list(images_dir.glob("*.jpg"))
    pairs = []
    
    for img_path in all_images:
        txt_path = labels_dir / f"{img_path.stem}.txt"
        if txt_path.exists():
            pairs.append((img_path, txt_path))
        else:
            print(f"Skipped (no label): {img_path.name}")
    
    if not pairs:
        print("No image-label pairs found.")
        return
    
    random.shuffle(pairs)
    split_index = int(len(pairs) * (1 - val_ratio))
    
    train_set = pairs[:split_index]
    val_set = pairs[split_index:]
    
    print(f"Split: {len(train_set)} training, {len(val_set)} validation")
    
    def move_files(file_pairs, split_name):
        for img, txt in file_pairs:
            shutil.move(str(img), str(base_path / "images" / split_name / img.name))
            shutil.move(str(txt), str(base_path / "labels" / split_name / txt.name))
    
    move_files(train_set, 'train')
    move_files(val_set, 'val')
    
    print("Dataset split completed.")


if __name__ == "__main__":
    split_dataset("datasets/polish_plates", val_ratio=0.3)