import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imports import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

def generate_heatmap(shape, coords, sigma=2):
    heatmap = np.zeros(shape, dtype=np.float32)
    for coord in coords:
        # Convert normalized coordinates to actual pixel coordinates
        pixel_coord = [int(coord[i] * shape[i]) for i in range(len(coord))]
        heatmap[tuple(pixel_coord)] = 1
    heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=sigma)
    return heatmap

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
    
    with ProcessPoolExecutor(max_workers=16) as executor:  # 用多进程代替多线程
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
    heatmap = generate_heatmap(shape, normalized_location) 
    
       
    for loc in normalized_location:
        z, x, y = [int(coord * dim) for coord, dim in zip(loc, shape)]
        crop_size = 96
        def crop_image(axis_value,axis_shape,crop_size = crop_size):
            if axis_value-int(crop_size/2) <= 0:
                axis_start = 0
                axis_end = crop_size
            elif axis_value+int(crop_size/2) >= axis_shape:
                axis_start = axis_shape-crop_size
                axis_end = axis_shape
            else:
                axis_start = axis_value-int(crop_size/2)
                axis_end = axis_value+int(crop_size/2)
            return axis_start,axis_end

        z_start, z_end = crop_image(z,shape[0], crop_size=crop_size)
        x_start, x_end = crop_image(x,shape[1], crop_size=crop_size)
        y_start,y_end = crop_image(y,shape[2], crop_size=crop_size)

        cropped_images = image_stack[z_start:z_end, x_start:x_end, y_start:y_end]
        cropped_heatmap = heatmap[z_start:z_end, x_start:x_end, y_start:y_end]

        if cropped_images.shape == (crop_size, crop_size, crop_size):
            cropped_nifti_image = nib.Nifti1Image(cropped_images, np.eye(4))
            nib.save(cropped_nifti_image, os.path.join(images_dir, f"{tomo_id}_cropped{z}_{x}_{y}.nii.gz"))
            cropped_nifti_heatmap = nib.Nifti1Image(cropped_heatmap, np.eye(4))
            nib.save(cropped_nifti_heatmap, os.path.join(labels_dir, f"{tomo_id}_cropped{z}_{x}_{y}_heatmap.nii.gz"))

    del slice_images, normalized_location,heatmap, image_stack, cropped_images, cropped_heatmap
    gc.collect()

def nifti_dataloader():
    train_transforms = Compose(
        [
            # LoadImaged(keys=["image"]),
            # EnsureChannelFirstd(keys=["image"]),
            # RandSpatialCropd(keys=["image"], roi_size=(96, 96, 96), random_center=True, random_size=False),
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            RandSpatialCropd(keys=["image"], roi_size=(96, 96, 96), random_center=True, random_size=False),
        ]
    )

    datalist = load_data_list(dataset_dir=config.UNET_DATAESET_DIR,data_list_key="train")
    val_files = load_data_list(dataset_dir=config.UNET_DATAESET_DIR,data_list_key="val")
    train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=2,
    cache_rate=1.0,
    num_workers=2,
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=2, cache_rate=1.0, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

    return train_loader, val_loader


def load_data_list(dataset_dir = config.UNET_DATAESET_DIR,data_list_key="train"):
    dataset = []
    image_files = sorted(os.listdir(os.path.join(dataset_dir, "images", data_list_key)))
    for image_file in image_files:
        image_path = os.path.join(dataset_dir, "images", data_list_key, image_file)
        label_file = image_file.replace(".nii.gz", "_heatmap.nii.gz")
        label_path = os.path.join(dataset_dir, "labels", data_list_key, label_file)
        if os.path.exists(label_path):
            dataset.append({'image': image_path, 'label': label_path})
    
    return dataset


def unet_train_main():
    train_loader, val_loader = nifti_dataloader()
    model = UNETR(
        in_channels=1,
        out_channels=1,  # Output should be 1 channel for heatmap
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(config.DEVICE)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    def validation(epoch_iterator_val):
        model.eval()
        with torch.no_grad():
            for batch in epoch_iterator_val:
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                val_labels = torch.stack([torch.tensor(generate_heatmap(val_inputs.shape, coords)).cuda() for coords in val_labels])
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
            mean_dice_val = dice_metric.aggregate().item()
            dice_metric.reset()
        return mean_dice_val

    def train(global_step, train_loader, dice_val_best, global_step_best):
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x = batch["image"].cuda()
            y = batch["label"].cuda()

            logit_map = model(x)
            loss = loss_function(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            epoch_iterator.set_description(  # noqa: B038
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
            )
            if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
                epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
                dice_val = validation(epoch_iterator_val)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(model.state_dict(), os.path.join(config.UNET_MODEL_DIR, "best_metric_model.pth"))
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                    )
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
            global_step += 1
        return global_step, dice_val_best, global_step_best
    
    max_iterations = 25000
    eval_num = 500
    post_label = AsDiscrete(to_onehot=14)
    post_pred = AsDiscrete(argmax=True, to_onehot=14)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)

if __name__ == "__main__":
    import_dataset()
    # nifti_dataloader()
    # unet_train_main()
