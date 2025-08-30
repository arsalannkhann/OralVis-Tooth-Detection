import argparse, os, glob
import numpy as np
import pandas as pd
from pathlib import Path
import cv2

# Anatomically-aware post-processing:
# 1) Separate upper vs lower arches by Y (image-relative)
# 2) Split left vs right by midline (X)
# 3) Sort within quadrant horizontally and assign FDI sequentially
# 4) Detect large gaps to skip missing teeth IDs

FDI_CLASSES = [
  'Canine (13)', 'Canine (23)', 'Canine (33)', 'Canine (43)',
  'Central Incisor (21)', 'Central Incisor (41)', 'Central Incisor (31)', 'Central Incisor (11)',
  'First Molar (16)', 'First Molar (26)', 'First Molar (36)', 'First Molar (46)',
  'First Premolar (14)', 'First Premolar (34)', 'First Premolar (44)', 'First Premolar (24)',
  'Lateral Incisor (22)', 'Lateral Incisor (32)', 'Lateral Incisor (42)', 'Lateral Incisor (12)',
  'Second Molar (17)', 'Second Molar (27)', 'Second Molar (37)', 'Second Molar (47)',
  'Second Premolar (15)', 'Second Premolar (25)', 'Second Premolar (35)', 'Second Premolar (45)',
  'Third Molar (18)', 'Third Molar (28)', 'Third Molar (38)', 'Third Molar (48)'
]

def yolo_to_xyxy(box, W, H):
    cls, cx, cy, w, h = box
    x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
    x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
    return cls, max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', required=True, help='Directory containing YOLO predictions (.txt) and source images')
    ap.add_argument('--out', default='results/sample_predictions')
    ap.add_argument('--gap_frac', type=float, default=0.08, help='Fraction of image width for missing-tooth gap detection')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img_exts = ('*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff')
    images = []
    for e in img_exts:
        images += glob.glob(os.path.join(args.pred_dir, e))

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None: 
            continue
        H, W = img.shape[:2]
        txt = Path(img_path).with_suffix('.txt').as_posix()
        if not os.path.exists(txt):
            continue
        # YOLO predict .txt: cls x_center y_center w h [conf?] -> handle 5+ fields
        boxes = []
        with open(txt,'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0]); cx, cy, w, h = map(float, parts[1:5])
                boxes.append((cls, cx, cy, w, h))

        # Convert to pixel xyxy
        px_boxes = [yolo_to_xyxy(b, W, H) for b in boxes]
        # Separate by upper/lower using median Y
        ys = np.array([(y1+y2)/2 for _,_,y1,_,y2 in px_boxes])
        mid_y = np.median(ys) if len(ys) else H//2
        upper = [b for b in px_boxes if (b[2]+b[4])/2 < mid_y]
        lower = [b for b in px_boxes if (b[2]+b[4])/2 >= mid_y]

        # Split each arch by X midline
        def process_arch(arch_boxes, quadrant_codes):
            xs = np.array([(x1+x2)/2 for _,x1,_,x2,_ in arch_boxes])
            mid_x = np.median(xs) if len(xs) else W//2
            left = [b for b in arch_boxes if (b[1]+b[3])/2 < mid_x]
            right = [b for b in arch_boxes if (b[1]+b[3])/2 >= mid_x]
            # Sort horizontally outward from midline
            left_sorted = sorted(left, key=lambda b: (b[1]+b[3])/2)  # left to right
            right_sorted = sorted(right, key=lambda b: (b[1]+b[3])/2) # left to right
            # Within each, compute gaps to skip IDs where large spacing
            def assign_ids(sorted_boxes, quadrant):
                # Teeth positions should run incisor(1) to molar(8)
                positions = []
                if not sorted_boxes:
                    return positions
                centers = [ (b[1]+b[3])/2 for b in sorted_boxes ]
                diffs = np.diff(sorted(centers))
                gap_thr = args.gap_frac * W
                pos = 1
                positions.append((sorted_boxes[0], quadrant*10 + pos))
                for i in range(1, len(sorted_boxes)):
                    if diffs[i-1] > gap_thr:
                        pos += 1  # skip one due to large gap
                    pos += 1
                    pos = min(pos, 8)
                    positions.append((sorted_boxes[i], quadrant*10 + pos))
                return positions
            # Map quadrants: for upper arch: left->2, right->1 ; for lower: left->3, right->4
            return (
                assign_ids(left_sorted, quadrant_codes[0]) +
                assign_ids(right_sorted, quadrant_codes[1])
            )

        # Upper: left=2, right=1 ; Lower: left=3, right=4
        assignments = process_arch(upper, (2,1)) + process_arch(lower, (3,4))

        # Draw
        vis = img.copy()
        for (cls,x1,y1,x2,y2), fdi in assignments:
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, str(fdi), (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        out_path = os.path.join(args.out, os.path.basename(img_path))
        cv2.imwrite(out_path, vis)

if __name__ == '__main__':
    main()
