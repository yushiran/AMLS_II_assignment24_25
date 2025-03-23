import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imports import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

def generate_heatmap(shape, coords, sigma=2):
    heatmap = np.zeros(shape, dtype=np.float32)
    for coord in coords:
        heatmap[tuple(coord)] = 1
    heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=sigma)
    return heatmap

def prepare_3dunet_dataset(images, coords):
    # Define transforms for image
    imtrans = Compose(
        [
            LoadImage(image_only=True),
            ScaleIntensity(),
            EnsureChannelFirst(),
            Resize((96, 96, 96)),  # Resize to a fixed shape
        ]
    )
    
    # Define transforms for heatmaps
    heatmaptrans = Compose(
        [
            EnsureType(),  # Ensure the heatmaps are in the correct type
            Resize((96, 96, 96)),  # Resize to a fixed shape
        ]
    )

    # Generate heatmaps from coordinates
    heatmaps = [generate_heatmap(images[i].shape, coord) for i, coord in enumerate(coords)]

    # Define dataset and dataloader
    ds = ArrayDataset(images, imtrans, heatmaps, heatmaptrans)
    loader = DataLoader(ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
    
    return loader

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

def normalize_loc(location, shape):
    normalized_location_list = []
    for i in location:
        normalized_location_list.append([float(i[0])/float(shape[0]), float(i[1]) / float(shape[1]), float(i[2]) / float(shape[2])])
    return normalized_location_list

def import_dataset(data_path = config.DATA_DIR, train_dir = config.TRAIN_DIR,train_split=0.8,
                   unet_dataset_dir=config.UNET_DATAESET_DIR, unet_images_train=config.UNET_IMAGES_TRAIN,
                   unet_images_val=config.UNET_IMAGES_VAL,
                   unet_labels_train=config.UNET_LABELS_TRAIN, unet_labels_val=config.UNET_LABELS_VAL):
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
    
    # Create directories if they do not exist
    for i in [unet_dataset_dir,unet_images_train,unet_images_val,unet_labels_train,unet_labels_val]:
        os.makedirs(i, exist_ok=True)

    # Process training tomograms
    process_tomogram_set(train_tomos, unet_images_train, unet_labels_train, "training", labels_df)
    # Process validation tomograms
    process_tomogram_set(val_tomos, unet_images_val, unet_labels_val, "validation", labels_df)

# Helper function to process a list of tomograms
def process_tomogram_set(tomogram_ids,images_dir, labels_dir, set_name,labels_df):
    motor_counts = []
    for tomo_id in tomogram_ids:
        now_motor_counts = [] #用来保存location的数组
        # Get motor annotations for the current tomogram
        tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
        for _, motor in tomo_motors.iterrows():
            if pd.isna(motor['Motor axis 0']):
                continue
            now_motor_counts.append(
                    [int(motor['Motor axis 0']), 
                    int(motor['Motor axis 1']), 
                    int(motor['Motor axis 2'])])                
        tomo_shape = [int(motor['Array shape (axis 0)']),
                      int(motor['Array shape (axis 1)']),
                      int(motor['Array shape (axis 2)'])]
        motor_counts.append(
            (tomo_id, now_motor_counts, tomo_shape))
    
    with ProcessPoolExecutor(max_workers=32) as executor:  # 用多进程代替多线程
        futures = [executor.submit(process_single_tomogram, tomo_id, location, shape, images_dir, labels_dir) 
                for tomo_id, location, shape in motor_counts]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tomograms"):
            try:
                future.result()  # 手动释放内存
                gc.collect()
            except Exception as e:
                print(f"Error in processing tomogram: {e}")

def process_single_tomogram(tomo_id, location, shape, images_dir, labels_dir):
    slice_images = []
    for z in range(shape[0]):
        slice_filename = f"slice_{z:04d}.jpg"
        src_path = os.path.join(config.TRAIN_DIR, tomo_id, slice_filename)
        if not os.path.exists(src_path):
            print(f"Warning: {src_path} does not exist, skipping.")
            continue
        img = Image.open(src_path)
        img_array = np.array(img)
        normalized_img = normalize_slice(img_array)
        normalized_location = normalize_loc(location, shape)
        slice_images.append(normalized_img)
        img.close()  # Close the image to release memory

    image_stack = np.stack(slice_images, axis=0)
    nifti_image = nib.Nifti1Image(image_stack, np.eye(4))
    nib.save(nifti_image, os.path.join(images_dir, f"{tomo_id}.nii.gz"))
    np.savetxt(os.path.join(labels_dir, f"{tomo_id}_coords.txt"), np.array(normalized_location), fmt='%f')
    del slice_images,nifti_image, normalized_location
    gc.collect()

if __name__ == "__main__":
    import_dataset()