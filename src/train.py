import argparse, os
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='configs/data.yaml')
    ap.add_argument('--model', default='yolov8s.pt')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--project', default='runs')
    ap.add_argument('--name', default='fdiyolo')
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        pretrained=True
    )

if __name__ == '__main__':
    main()
