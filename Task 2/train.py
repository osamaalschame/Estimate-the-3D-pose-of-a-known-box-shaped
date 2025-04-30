import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO segmentation model")
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model .pt file')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=512, help='Image size for training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--name', type=str, default='Carton-seg-s', help='Run name')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')

    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        cos_lr=True,
        epochs=args.epochs,
        imgsz=args.imgsz,
        pretrained=True,
        augment=True,
        batch=args.batch,
        overlap_mask=True,
        patience=args.patience,
        cache=True,
        optimizer='auto',
        plots=True,
        exist_ok=True,
        scale=0.5,
        name=args.name
    )

if __name__ == "__main__":
    main()

