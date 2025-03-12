import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imports import *
from glob import glob

def create_animation(df, tomo_id, fps=20, imsize=384, frame_skip=1, save_as_gif=False, save_dir="./"):
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
        
    return anim

def create_animation_images(df, tomo_id, n_images=16, imsize=384, save_dir="./"):
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
    plt.close()



if __name__ == '__main__':
    train_labels = pd.read_csv(config.TRAIN_CSV)
    # create_animation(df=train_labels,tomo_id= "tomo_0a8f05",save_as_gif=True, save_dir=f"{config.BASE_DIR}/outputs/tomo_gif")
    create_animation_images(df=train_labels, tomo_id="tomo_00e463", n_images=9,save_dir=f"{config.BASE_DIR}/outputs/motor_visualization")