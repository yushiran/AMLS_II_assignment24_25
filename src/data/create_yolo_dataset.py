import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imports import *

# Define a helper function for image normalization using percentile-based contrast enhancement.
def normalize_slice(slice_data):
    """
    Normalize slice data using the 2nd and 98th percentiles.
    
    Args:
        slice_data (numpy.array): Input image slice.
    
    Returns:
        np.uint8: Normalized image in the range [0, 255].
    """
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    clipped_data = np.clip(slice_data, p2, p98)
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    return np.uint8(normalized)


# Define the preprocessing function to extract slices, normalize, and generate YOLO annotations.
def prepare_yolo_dataset(trust=4, train_split=0.8, box_size = 24, 
                         data_path = config.DATA_DIR, 
                         train_dir = config.TRAIN_DIR, 
                         yolo_dataset_dir=config.YOLO_DATAESET_DIR,
                         yolo_images_train=config.YOLO_IMAGES_TRAIN,
                         yolo_images_val=config.YOLO_IMAGES_VAL,
                         yolo_labels_train=config.YOLO_LABELS_TRAIN,
                         yolo_labels_val=config.YOLO_LABELS_VAL):
    """
    Extract slices containing motors and save images with corresponding YOLO annotations.
    
    Steps:
    - Load the motor labels.
    - Perform a train/validation split by tomogram.
    - For each motor, extract slices in a range (± trust parameter).
    - Normalize each slice and save it.
    - Generate YOLO format bounding box annotations with a fixed box size.
    - Create a YAML configuration file for YOLO training.
    
    Returns:
        dict: A summary containing dataset statistics and file paths.
    """
    # Load the labels CSV
    labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    
    total_motors = labels_df['Number of motors'].sum()
    print(f"Total number of motors in the dataset: {total_motors}")
    
    # Consider only tomograms with at least one motor
    tomo_df = labels_df[labels_df['Number of motors'] > 0].copy()
    unique_tomos = tomo_df['tomo_id'].unique()
    print(f"Found {len(unique_tomos)} unique tomograms with motors")
    
    # Shuffle and split tomograms into train and validation sets
    np.random.shuffle(unique_tomos)
    split_idx = int(len(unique_tomos) * train_split)
    train_tomos = unique_tomos[:split_idx]
    val_tomos = unique_tomos[split_idx:]
    print(f"Split: {len(train_tomos)} tomograms for training, {len(val_tomos)} tomograms for validation")
    
    # Helper function to process a list of tomograms
    def process_tomogram_set(tomogram_ids, images_dir, labels_dir, set_name):
        motor_counts = []
        for tomo_id in tomogram_ids:
            # Get motor annotations for the current tomogram
            tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
            for _, motor in tomo_motors.iterrows():
                if pd.isna(motor['Motor axis 0']):
                    continue
                motor_counts.append(
                    (tomo_id, 
                     int(motor['Motor axis 0']), 
                     int(motor['Motor axis 1']), 
                     int(motor['Motor axis 2']),
                     int(motor['Array shape (axis 0)']))
                )
        
        print(f"Will process approximately {len(motor_counts) * (2 * trust + 1)} slices for {set_name}")
        processed_slices = 0
        
        # Loop over each motor annotation
        for tomo_id, z_center, y_center, x_center, z_max in tqdm(motor_counts, desc=f"Processing {set_name} motors"):
            z_min = max(0, z_center - trust)
            z_max_bound = min(z_max - 1, z_center + trust)
            for z in range(z_min, z_max_bound + 1):
                # Create the slice filename and source path
                slice_filename = f"slice_{z:04d}.jpg"
                src_path = os.path.join(train_dir, tomo_id, slice_filename)
                if not os.path.exists(src_path):
                    print(f"Warning: {src_path} does not exist, skipping.")
                    continue
                
                # Load, normalize, and save the image slice
                img = Image.open(src_path)
                img_array = np.array(img)
                normalized_img = normalize_slice(img_array)
                dest_filename = f"{tomo_id}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
                dest_path = os.path.join(images_dir, dest_filename)
                Image.fromarray(normalized_img).save(dest_path)
                
                # Prepare YOLO bounding box annotation (normalized values)
                img_width, img_height = img.size
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                box_width_norm = box_size / img_width
                box_height_norm = box_size / img_height
                label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")
                
                processed_slices += 1
        
        return processed_slices, len(motor_counts)
    
    # Process training tomograms
    train_slices, train_motors = process_tomogram_set(train_tomos, yolo_images_train, yolo_labels_train, "training")
    # Process validation tomograms
    val_slices, val_motors = process_tomogram_set(val_tomos, yolo_images_val, yolo_labels_val, "validation")
    
    # Generate YAML configuration for YOLO training
    yaml_content = {
        'path': yolo_dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'motor'}
    }
    with open(os.path.join(yolo_dataset_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\nProcessing Summary:")
    print(f"- Train set: {len(train_tomos)} tomograms, {train_motors} motors, {train_slices} slices")
    print(f"- Validation set: {len(val_tomos)} tomograms, {val_motors} motors, {val_slices} slices")
    print(f"- Total: {len(train_tomos) + len(val_tomos)} tomograms, {train_motors + val_motors} motors, {train_slices + val_slices} slices")
    
    return {
        "dataset_dir": yolo_dataset_dir,
        "yaml_path": os.path.join(yolo_dataset_dir, 'dataset.yaml'),
        "train_tomograms": len(train_tomos),
        "val_tomograms": len(val_tomos),
        "train_motors": train_motors,
        "val_motors": val_motors,
        "train_slices": train_slices,
        "val_slices": val_slices
    }


def create_yolo_dataset():
    # Create necessary directories
    for dir_path in [config.YOLO_IMAGES_TRAIN, config.YOLO_IMAGES_VAL, config.YOLO_LABELS_TRAIN, config.YOLO_LABELS_VAL]:
        os.makedirs(dir_path, exist_ok=True)

    # Define constants for processing
    TRUST = 4       # Number of slices above and below center slice (total slices = 2*TRUST + 1)
    BOX_SIZE = 24   # Bounding box size (in pixels)
    TRAIN_SPLIT = 0.8  # 80% training, 20% validation

    # Run the preprocessing
    summary = prepare_yolo_dataset(trust=TRUST,
                                   train_split=TRAIN_SPLIT,
                                   box_size=BOX_SIZE,
                                   data_path=config.DATA_DIR,
                                   train_dir=config.TRAIN_DIR,
                                   yolo_dataset_dir=config.YOLO_DATAESET_DIR,
                                   yolo_images_train=config.YOLO_IMAGES_TRAIN,
                                   yolo_images_val=config.YOLO_IMAGES_VAL,
                                   yolo_labels_train=config.YOLO_LABELS_TRAIN,
                                   yolo_labels_val=config.YOLO_LABELS_VAL)

    print(f"\nPreprocessing Complete:")
    print(f"- Training data: {summary['train_tomograms']} tomograms, {summary['train_motors']} motors, {summary['train_slices']} slices")
    print(f"- Validation data: {summary['val_tomograms']} tomograms, {summary['val_motors']} motors, {summary['val_slices']} slices")
    print(f"- Dataset directory: {summary['dataset_dir']}")
    print(f"- YAML configuration: {summary['yaml_path']}")
    print("\nReady for YOLO training!")
