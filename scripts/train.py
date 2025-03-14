import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imports import *
from data import create_yolo_dataset
from models import train_yolo


def train_main():
    if not os.path.exists(config.YOLO_DATAESET_DIR):
        print(f"YOLO Dataset not found at {config.YOLO_DATAESET_DIR}")
        create_yolo_dataset.create_yolo_dataset()
        sys.exit(1)

    print("Starting YOLO training process...")
    yaml_path = os.path.join(config.YOLO_DATAESET_DIR, 'dataset.yaml')
    print(f"Using YAML file: {yaml_path}")
    with open(yaml_path, 'r') as f:
        print(f"YAML contents:\n{f.read()}")
    
    print("\nStarting YOLO training...")
    yolo_pretrained_weights = os.path.join(config.YOLO_WEIGHTS_DIR, 'yolov8m.pt')
    model, results = train_yolo.train_yolo_model(
        yaml_path,
        pretrained_weights_path=yolo_pretrained_weights,
        epochs=100,  # For demonstration, using 30 epochs
        batch_size=16,
    )

def inference_main():
    model = YOLO(model=os.path.join(config.YOLO_WEIGHTS_DIR, 'yolov8_motor_detector_epoch100_batchsize16', 'weights', 'best.pt'))

    print("\nTraining complete!")
    print("\nRunning predictions on sample images...")
    train_yolo.predict_on_samples(model, num_samples=4)

if __name__ == '__main__':
    # train_main()
    inference_main()