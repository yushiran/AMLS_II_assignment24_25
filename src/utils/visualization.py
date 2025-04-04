import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imports import *
from glob import glob

def create_animation(df, tomo_id, fps=20, imsize=384, frame_skip=1, save_as_gif=False, save_dir="./"):
    """
    Creates an animation from a series of images and annotations for a given tomogram ID.
    Args:
        df (pd.DataFrame): DataFrame containing metadata and annotations for the tomograms.
            Must include columns such as 'tomo_id', 'Array shape (axis 1)', 'Array shape (axis 2)', 
            'Voxel spacing', and motor axis coordinates ('Motor axis 0', 'Motor axis 1', 'Motor axis 2').
        tomo_id (str): Identifier for the tomogram to process.
        fps (int, optional): Frames per second for the animation. Defaults to 20.
        imsize (int, optional): Size (in pixels) to resize the images to. Defaults to 384.
        frame_skip (int, optional): Number of frames to skip when creating the animation. 
            If set to 0, it will automatically skip frames to limit the animation to 100 frames. Defaults to 1.
        save_as_gif (bool, optional): Whether to save the animation as a GIF. Defaults to False.
        save_dir (str, optional): Directory to save the GIF if `save_as_gif` is True. Defaults to "./".
    Returns:
        matplotlib.animation.FuncAnimation: The generated animation object.
    Notes:
        - The function reads image and annotation files from directories specified in `config.DATA_DIR`.
        - Images and annotations are resized to the specified `imsize` while maintaining aspect ratio.
        - Motor annotations are visualized as circles on the images, with their size determined by voxel spacing.
        - If `save_as_gif` is True, the animation is saved as a GIF using the ImageMagick writer.
    Raises:
        FileNotFoundError: If image or annotation files are not found in the specified directories.
        ValueError: If the DataFrame does not contain the required columns.
    """
   
    # Get paths
    img_paths = sorted(glob(f"{config.DATA_DIR}/train/{tomo_id}/*"))
    annot_paths = sorted(glob(f"{config.DATA_DIR}/train/{tomo_id}/*"))

    pdf = df[df['tomo_id']==tomo_id]
    orig_y = pdf["Array shape (axis 1)"].values[0]
    orig_x = pdf["Array shape (axis 2)"].values[0]
    ratio_x = imsize/orig_x
    ratio_y = imsize/orig_y
    ratio_m = min(ratio_x, ratio_y)

    vox_sp = pdf["Voxel spacing"].values[0]
    
    
    # Get images
    images = np.stack([cv2.imread(path) for path in img_paths],axis=0)
    images = [np.array(cv2.resize(img, (imsize,imsize))) for img in images]
    images = (images-np.min(images))/(np.max(images)-np.min(images)+1e-6)
    images = (images*255).astype(np.uint8)

    annot = np.stack([cv2.imread(path) for path in annot_paths],axis=0)
    annot = [np.array(cv2.resize(img, (imsize,imsize))) for img in annot]
    annot = (annot-np.min(annot))/(np.max(annot)-np.min(annot)+1e-6)
    annot = (annot*255).astype(np.uint8)   

    mat_imgs = np.zeros_like(annot)
    
    for k, row in pdf.iterrows():
        m_ax_z = row["Motor axis 0"]
        m_ax_y = row["Motor axis 1"]
        m_ax_x = row["Motor axis 2"]
        tmp_mask = annot
        for j, (i, a) in enumerate((zip(images, annot))):
            base_rad = 1000 / vox_sp * ratio_m
            rad = (base_rad**2 - abs(m_ax_z-j)**2)
            rad = max(0, rad) **0.5
            rad = int(np.round(rad))
            cv2.circle(tmp_mask[j], (int(m_ax_x*ratio_x), int(m_ax_y*ratio_y)), rad, (255*(1&k), 255*(2&k), 255*(4&k)), thickness=-1)

            
            if j==m_ax_z:
                cv2.putText(tmp_mask[j], 'GT_slice', (imsize-145, imsize-20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255*(1&k), 255*(2&k), 255*(4&k)), thickness=2)
        for j in range(len(images)):
            mat_imgs[j] = cv2.addWeighted(tmp_mask[j], 0.5, images[j], 0.5, 0)
            cv2.putText(mat_imgs[j], f'{j:03d}/{len(images)-1:03d}', (5, imsize-20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), thickness=2)

    ims_sgs = [np.concatenate([images[i], mat_imgs[i]], axis=1) for i in range(len(images))]
    
    # Stack images
    if frame_skip==0:
        frame_skip = len(images)//100
    animation_arr = np.stack(ims_sgs, axis=0)[::frame_skip]

    del images, ims_sgs
    gc.collect()
    
    # Initialise plot
    fig = plt.figure(figsize=(6,3), dpi=128)  # if size is too big then gif gets truncated

    im = plt.imshow(animation_arr[0], cmap='bone')
    plt.axis('off')
    plt.title(f"{tomo_id}", fontweight="bold")
    
    # Load next frame
    def animate_func(i):
        im.set_array(animation_arr[i])
        return [im]
    plt.close()
    
    # Animation function
    anim = animation.FuncAnimation(fig, animate_func, frames = animation_arr.shape[0], interval = 200//fps)
   
    # Save
    if save_as_gif:
        os.makedirs(save_dir, exist_ok=True)
        anim.save(os.path.join(save_dir, f"patient_{tomo_id}.gif"), fps=fps, writer='imagemagick')
        print(f'Animation saved to {os.path.join(save_dir, f"patient_{tomo_id}.gif")}')
        
    return anim

def create_animation_images(df, tomo_id, n_images=16, imsize=384, save_dir="./"):
    """
    Generate and save an image matrix visualization for a given tomogram ID.
    This function processes a set of images corresponding to a tomogram ID, 
    annotates motor positions on the images, and arranges a subset of the 
    images into a grid layout. The resulting image matrix is saved as a PNG file.
    Args:
        df (pd.DataFrame): A DataFrame containing metadata about the tomograms, 
            including motor positions and voxel spacing.
        tomo_id (str): The identifier for the tomogram to process.
        n_images (int, optional): The number of images to include in the matrix. 
            Defaults to 16.
        imsize (int, optional): The size (width and height) to which each image 
            will be resized. Defaults to 384.
        save_dir (str, optional): The directory where the resulting image matrix 
            will be saved. Defaults to "./".
    Returns:
        None: The function saves the image matrix to the specified directory 
        and does not return any value.
    Notes:
        - The function ensures that frames with motor annotations are included 
          in the image matrix.
        - Images are normalized and resized before being processed.
        - Motor positions are annotated as circles on the images, with the 
          radius calculated based on voxel spacing and resizing ratios.
    Example:
        create_animation_images(df, tomo_id="tomo_001", n_images=9, imsize=256, save_dir="./output")
    """
    
    # Get paths
    img_paths = sorted(glob(f"{config.DATA_DIR}/train/{tomo_id}/*"))

    pdf = df[df['tomo_id'] == tomo_id]
    orig_y = pdf["Array shape (axis 1)"].values[0]
    orig_x = pdf["Array shape (axis 2)"].values[0]
    ratio_x = imsize / orig_x
    ratio_y = imsize / orig_y
    ratio_m = min(ratio_x, ratio_y)

    vox_sp = pdf["Voxel spacing"].values[0]

    # Get images
    images = np.stack([cv2.imread(path) for path in img_paths], axis=0)
    images = [np.array(cv2.resize(img, (imsize, imsize))) for img in images]
    images = (images - np.min(images)) / (np.max(images) - np.min(images) + 1e-6)
    images = (images * 255).astype(np.uint8)

    mat_imgs = np.copy(images)

    for k, row in pdf.iterrows():
        m_ax_z = row["Motor axis 0"]
        m_ax_y = row["Motor axis 1"]
        m_ax_x = row["Motor axis 2"]
        for j in range(len(images)):
            if j == m_ax_z:
                base_rad = 1000 / vox_sp * ratio_m
                rad = (base_rad**2 - abs(m_ax_z - j)**2)
                rad = max(0, rad)**0.5
                rad = int(np.round(rad))
                cv2.circle(mat_imgs[j], (int(m_ax_x * ratio_x), int(m_ax_y * ratio_y)), rad, (255, 0, 0), thickness=2)
                # cv2.putText(mat_imgs[j], 'Motor', (int(m_ax_x * ratio_x), int(m_ax_y * ratio_y) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), thickness=1)

    # Calculate the number of rows and columns for the image matrix
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))

    # Determine the indices of the frames to be displayed
    frame_indices = np.linspace(0, len(images) - 1, n_images).astype(int)
    
    # Ensure frames with motor annotations are included
    motor_frames = pdf["Motor axis 0"].values.astype(int)
    frame_indices = np.unique(np.concatenate((frame_indices, motor_frames)))
    frame_indices = frame_indices[:n_images]  # Limit to n_images

    # Create the image matrix
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < len(frame_indices):
            frame_idx = frame_indices[idx]
            img = mat_imgs[frame_idx]
            ax.imshow(img, cmap='bone')
            ax.axis('off')
            ax.set_title(f"Frame {frame_idx}", fontweight="bold")
        else:
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{tomo_id}_image_matrix.png"))
    print(f'Image matrix saved to {os.path.join(save_dir, f'{tomo_id}_image_matrix.png')}')
    plt.close()

def visualize_random_training_samples(num_samples=4,images_train_dir=config.YOLO_IMAGES_TRAIN,labels_train_dir=config.YOLO_LABELS_TRAIN):
    """
    Visualize random training samples with YOLO annotations.
    
    Args:
        num_samples (int): Number of random images to display.
    """
    # Get all image files from the train directory (support multiple image extensions)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob(os.path.join(images_train_dir, "**", ext), recursive=True))
    
    if len(image_files) == 0:
        print("No image files found in the train directory!")
        return
        
    num_samples = min(num_samples, len(image_files))
    random_images = random.sample(image_files, num_samples)
    
    # Create subplots for visualization
    rows = int(np.ceil(num_samples / 2))
    cols = min(num_samples, 2)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, img_path in enumerate(random_images):
        try:
            # Determine corresponding label file (YOLO format)
            relative_path = os.path.relpath(img_path, images_train_dir)
            label_path = os.path.join(labels_train_dir, os.path.splitext(relative_path)[0] + '.txt')
            
            # Load and normalize image for display
            img = Image.open(img_path)
            img_width, img_height = img.size
            img_array = np.array(img)
            p2 = np.percentile(img_array, 2)
            p98 = np.percentile(img_array, 98)
            normalized = np.clip(img_array, p2, p98)
            normalized = 255 * (normalized - p2) / (p98 - p2)
            img_normalized = Image.fromarray(np.uint8(normalized))
            
            # Convert to RGB for annotation drawing
            img_rgb = img_normalized.convert('RGB')
            overlay = Image.new('RGBA', img_rgb.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Load YOLO annotations if available
            annotations = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        # YOLO format: class x_center y_center width height (normalized values)
                        values = line.strip().split()
                        class_id = int(values[0])
                        x_center = float(values[1]) * img_width
                        y_center = float(values[2]) * img_height
                        width = float(values[3]) * img_width
                        height = float(values[4]) * img_height
                        annotations.append({
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
            
            # Draw annotations on the overlay
            for ann in annotations:
                x_center = ann['x_center']
                y_center = ann['y_center']
                width = ann['width']
                height = ann['height']
                x1 = max(0, int(x_center - width/2))
                y1 = max(0, int(y_center - height/2))
                x2 = min(img_width, int(x_center + width/2))
                y2 = min(img_height, int(y_center + height/2))
                draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 64), outline=(255, 0, 0, 200))
                draw.text((x1, y1-10), f"Class {ann['class_id']}", fill=(255, 0, 0, 255))
            
            # Indicate if no annotations were found
            if not annotations:
                draw.text((10, 10), "No annotations found", fill=(255, 0, 0, 255))
            
            # Composite overlay and display image
            img_rgb = Image.alpha_composite(img_rgb.convert('RGBA'), overlay).convert('RGB')
            axes[i].imshow(np.array(img_rgb))
            # img_name = os.path.basename(img_path)
            # axes[i].set_title(f"Image: {img_name}\nAnnotations: {len(annotations)}")
            # axes[i].axis('on')
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading image: {os.path.basename(img_path)}",
                         horizontalalignment='center', verticalalignment='center')
            axes[i].axis('off')
        
        axes[i].set_axis_off()
    
    # Turn off any extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/motor_visualization/random_training_samples.png")
    print(f"Displayed {num_samples} random images with YOLO annotations, saved at {config.OUTPUT_DIR}/motor_visualization/random_training_samples.png")

def plot_dfl_loss_curve(run_dir):
    """
    Plot the DFL loss curves for training and validation, marking the best model.
    
    Args:
        run_dir (str): Directory where the training results are stored.
    """
    results_csv = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(results_csv):
        print(f"Results file not found at {results_csv}")
        return
    
    results_df = pd.read_csv(results_csv)
    train_dfl_col = [col for col in results_df.columns if 'train/dfl_loss' in col]
    val_dfl_col = [col for col in results_df.columns if 'val/dfl_loss' in col]
    
    if not train_dfl_col or not val_dfl_col:
        print("DFL loss columns not found in results CSV")
        print(f"Available columns: {results_df.columns.tolist()}")
        return
    
    train_dfl_col = train_dfl_col[0]
    val_dfl_col = val_dfl_col[0]
    
    best_epoch = results_df[val_dfl_col].idxmin()
    best_val_loss = results_df.loc[best_epoch, val_dfl_col]
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['epoch'], results_df[train_dfl_col], label='Train DFL Loss')
    plt.plot(results_df['epoch'], results_df[val_dfl_col], label='Validation DFL Loss')
    plt.axvline(x=results_df.loc[best_epoch, 'epoch'], color='r', linestyle='--', 
                label=f'Best Model (Epoch {int(results_df.loc[best_epoch, "epoch"])}, Val Loss: {best_val_loss:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('DFL Loss')
    plt.title('Training and Validation DFL Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(run_dir, 'dfl_loss_curve.png')
    plt.savefig(plot_path)
    # plt.savefig(os.path.join('/kaggle/working', 'dfl_loss_curve.png'))
    
    print(f"Loss curve saved to {plot_path}")
    plt.close()
    
    return best_epoch, best_val_loss

if __name__ == '__main__':
    train_labels = pd.read_csv(config.TRAIN_CSV)
    # create_animation(df=train_labels,tomo_id= "tomo_0a8f05",save_as_gif=True, save_dir=f"{config.BASE_DIR}/outputs/tomo_gif")
    # create_animation_images(df=train_labels, tomo_id="tomo_00e463", n_images=9,save_dir=f"{config.BASE_DIR}/outputs/motor_visualization")
    # visualize_random_training_samples(num_samples=4,images_train_dir=config.YOLO_IMAGES_TRAIN,labels_train_dir=config.YOLO_LABELS_TRAIN)
