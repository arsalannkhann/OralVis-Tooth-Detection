import argparse, os, json
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='runs/fdiyolo/weights/best.pt')
    ap.add_argument('--data', default='configs/data.yaml')
    ap.add_argument('--save_json', action='store_true')
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data)
    # metrics results object has attributes like results_dict
    results = getattr(metrics, 'results_dict', None)
    if args.save_json and results:
        with open('results/metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
    print(results)

if __name__ == '__main__':
    main()
