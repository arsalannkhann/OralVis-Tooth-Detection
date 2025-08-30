# OralVis Tooth Detection (YOLOv5/YOLOv8/YOLOv11)

End-to-end implementation to detect and number teeth on panoramic radiographs using YOLO and FDI numbering.

## Repo Structure
```
OralVis-Tooth-Detection/
├── configs/
│   └── data.yaml
├── data/
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
├── notebooks/
│   └── OralVis_Tooth_Detection.ipynb
├── results/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── sample_predictions/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── post_process.py
│   └── utils.py
├── submission/
│   └── OralVis_Submission_Report_Template.docx
├── requirements.txt
└── README.md
```

## Quickstart
1. **Install**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place Dataset**
   Organize to `data/images/...` and `data/labels/...` with 80/10/10 split (images paired to YOLO `.txt` labels).

3. **Train**
   ```bash
   # YOLOv8 small; change model to yolov8n.pt/yolov8m.pt/yolov8l.pt or YOLOv5/YOLOv11 via Ultralytics
   yolo detect train data=configs/data.yaml model=yolov8s.pt imgsz=640 epochs=100 batch=16 project=runs name=fdiyolo
   ```

4. **Validate**
   ```bash
   yolo detect val model=runs/fdiyolo/weights/best.pt data=configs/data.yaml
   ```

5. **Predict Samples**
   ```bash
   yolo detect predict model=runs/fdiyolo/weights/best.pt source=data/images/test save_txt save_conf
   ```

6. **Post-Process (FDI Ordering & Missing Teeth)**
   ```bash
   python src/post_process.py --pred_dir runs/predict --out results/sample_predictions
   ```

7. **Export Confusion Matrix & Training Curves**
   Use the notebook `notebooks/OralVis_Tooth_Detection.ipynb` (Section 5 & 6).

## Notes
- Recommended image size: 640×640, pretrained weights (e.g., `yolov8s.pt`).
- Keep **class order identical** to `configs/data.yaml`.
- Include confusion matrix, precision/recall, mAP@50, mAP@50-95, 3+ sample images, and training curves in submission.
- See `submission/OralVis_Submission_Report_Template.docx` for the required report structure.
