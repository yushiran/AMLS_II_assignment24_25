import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imports import *
from utils import visualization

def train_yolo_model(yaml_path, pretrained_weights_path, epochs=50, batch_size=16, img_size=640):
    """
    Train a YOLO model on the prepared dataset with optimized accuracy settings.

    Args:
        yaml_path (str): Path to the dataset YAML file.
        pretrained_weights_path (str): Path to pre-downloaded weights file.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        img_size (int): Image size for training.

    Returns:
        model (YOLO): Trained YOLO model.
        results: Training results.
    """
    print(f"Loading pre-trained weights from: {pretrained_weights_path}")
    model = YOLO(pretrained_weights_path)

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=config.YOLO_WEIGHTS_DIR,
        name=f'yolov8_motor_detector_epoch{epochs}_batchsize{batch_size}',
        exist_ok=True,
        patience=30,  # Stop training if no improvement after 10 epochs
        save_period=5,  # Save model every 5 epochs
        val=True,
        verbose=True,
        optimizer="AdamW",  # AdamW optimizer for stability
        lr0=0.001,  # Initial learning rate
        lrf=0.01,  # Final learning rate factor
        cos_lr=True,  # Use cosine learning rate decay
        weight_decay=0.0005,  # Prevent overfitting
        momentum=0.937,  # Momentum for better gradient updates
        close_mosaic=10,  # Disable mosaic augmentation after 10 epochs
        mixup=0.2,  # Apply mixup augmentation
        workers=4,  # Speed up data loading
        augment=True,  # Enable additional augmentations
        amp=True,  # Mixed precision training for faster performance
        dropout=0.1,
    )

    run_dir = os.path.join(config.YOLO_WEIGHTS_DIR, 'motor_detector')
    
    # If function is defined, plot loss curves for better insights
    if 'plot_dfl_loss_curve' in globals():
        best_epoch_info = visualization.plot_dfl_loss_curve(run_dir)
        if best_epoch_info:
            best_epoch, best_val_loss = best_epoch_info
            print(f"\nBest model found at epoch {best_epoch} with validation DFL loss: {best_val_loss:.4f}")

    return model, results

def predict_on_samples(model, num_samples=4):
    """
    Run predictions on random validation samples and display results.
    
    Args:
        model: Trained YOLO model.
        num_samples (int): Number of random samples to test.
    """
    val_dir = os.path.join(config.YOLO_DATAESET_DIR, 'images', 'val')
    if not os.path.exists(val_dir):
        print(f"Validation directory not found at {val_dir}")
        val_dir = os.path.join(config.YOLO_DATAESET_DIR, 'images', 'train')
        print(f"Using train directory for predictions instead: {val_dir}")
        
    if not os.path.exists(val_dir):
        print("No images directory found for predictions")
        return
    
    val_images = os.listdir(val_dir)
    if len(val_images) == 0:
        print("No images found for prediction")
        return
    
    num_samples = min(num_samples, len(val_images))
    samples = random.sample(val_images, num_samples)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, img_file in enumerate(samples):
        if i >= len(axes):
            break
            
        img_path = os.path.join(val_dir, img_file)
        results = model.predict(img_path, conf=0.25)[0]
        img = Image.open(img_path)
        axes[i].imshow(np.array(img), cmap='gray')
        
        # Draw ground truth box if available (extracted from filename)
        try:
            parts = img_file.split('_')
            y_part = [p for p in parts if p.startswith('y')]
            x_part = [p for p in parts if p.startswith('x')]
            if y_part and x_part:
                y_gt = int(y_part[0][1:])
                x_gt = int(x_part[0][1:].split('.')[0])
                box_size = 24
                rect_gt = Rectangle((x_gt - box_size//2, y_gt - box_size//2), box_size, box_size,
                                      linewidth=1, edgecolor='g', facecolor='none')
                axes[i].add_patch(rect_gt)
        except:
            pass
        
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box
                rect_pred = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
                axes[i].add_patch(rect_pred)
        axes[i].set_axis_off()
        # axes[i].set_title(f"Image: {img_file}\nGT (green) vs Pred (red)")
    
    plt.tight_layout()
    save_path = f'{config.OUTPUT_DIR}/predict_samples'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'predictions.png'))


