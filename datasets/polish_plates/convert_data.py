import xml.etree.ElementTree as ET
import os
from pathlib import Path


def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    
    xtl, ytl, xbr, ybr = box
    
    x = (xtl + xbr) / 2.0
    y = (ytl + ybr) / 2.0
    w = xbr - xtl
    h = ybr - ytl
    
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    
    return (x, y, w, h)


def convert_cvat_to_yolo(xml_file, output_dir='labels'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    count = 0
    for image in root.findall('image'):
        file_name = image.get('name')
        width = int(image.get('width'))
        height = int(image.get('height'))
        
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_name)
        
        with open(txt_path, 'w') as f:
            for box in image.findall('box'):
                if box.get('label') == 'plate':
                    xtl = float(box.get('xtl'))
                    ytl = float(box.get('ytl'))
                    xbr = float(box.get('xbr'))
                    ybr = float(box.get('ybr'))
                    
                    x, y, w, h = convert_to_yolo((width, height), (xtl, ytl, xbr, ybr))
                    f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        count += 1
    
    print(f"Converted {count} images. Labels saved to {output_dir}")


if __name__ == "__main__":
    convert_cvat_to_yolo('annotations.xml', 'labels')