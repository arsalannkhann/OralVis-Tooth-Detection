import os
import glob
from pathlib import Path
import random
import shutil

def pair_exists(img_path: str) -> bool:
    txt = Path(img_path).with_suffix('.txt').as_posix().replace('/images/','/labels/')
    return os.path.exists(txt)

def split_dataset(images_root: str, labels_root: str, out_root: str, train=0.8, val=0.1, test=0.1, seed=42):
    random.seed(seed)
    all_images = []
    for ext in ('*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff'):
        all_images += glob.glob(os.path.join(images_root, ext))
    all_images = [p for p in all_images if pair_exists(p)]
    random.shuffle(all_images)
    n = len(all_images)
    n_train = int(n*train)
    n_val = int(n*val)
    train_imgs = all_images[:n_train]
    val_imgs = all_images[n_train:n_train+n_val]
    test_imgs = all_images[n_train+n_val:]
    for split, imgs in [('train',train_imgs),('val',val_imgs),('test',test_imgs)]:
        out_img = os.path.join(out_root,'images',split)
        out_lbl = os.path.join(out_root,'labels',split)
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)
        for ip in imgs:
            lp = Path(ip).with_suffix('.txt').as_posix().replace('/images/','/labels/')
            shutil.copy2(ip, os.path.join(out_img, os.path.basename(ip)))
            shutil.copy2(lp, os.path.join(out_lbl, os.path.basename(lp)))
    return {'train': len(train_imgs), 'val': len(val_imgs), 'test': len(test_imgs)}

def yolo_txt_to_boxes(txt_path: str):
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = parts
            boxes.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return boxes
