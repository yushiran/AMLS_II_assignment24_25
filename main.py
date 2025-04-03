import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.imports import *

if __name__ == '__main__':
    print('Starting YOLO dataset creation...')
    create_yolo_dataset.create_yolo_dataset()
    
    print('Starting YOLO inference...')
    infer_yolo.yolo_inference_main()

    print("Starting visualization process...")
    train_labels = pd.read_csv(config.TRAIN_CSV)
    visualization.create_animation(df=train_labels,tomo_id= "tomo_0a8f05",save_as_gif=True, save_dir=f"{config.BASE_DIR}/outputs/tomo_gif")
    visualization.create_animation_images(df=train_labels, tomo_id="tomo_00e463", n_images=9,save_dir=f"{config.BASE_DIR}/outputs/motor_visualization")
    visualization.visualize_random_training_samples(num_samples=4,images_train_dir=config.YOLO_IMAGES_TRAIN,labels_train_dir=config.YOLO_LABELS_TRAIN)
