# BYU_Locating_Bacterial_Flagellar_Motors_2025# BYU Locating Bacterial Flagellar Motors 2025

## Project Overview
This project aims to detect bacterial flagellar motors in tomography data using a combination of deep learning approaches, specifically YOLOv8 for object detection and 3D UNet for segmentation. The workflow includes data preprocessing, model training, and inference processes to accurately locate bacterial flagellar motors in 3D tomographic images.

## Project Organization

### Core Directories
- `byu-locating-bacterial-flagellar-motors-2025/`: Contains the raw competition data
  - `train/`: Training tomography data
  - `test/`: Test tomography data for prediction
  - `train_labels.csv`: Ground truth coordinates for training
  - `sample_submission.csv`: Submission format template

### Data Processing
- `data/`: Data processing utilities
  - `__init__.py`: Package initialization
  - `create_yolo_dataset.py`: Creates YOLO-compatible dataset from raw data
  - `dataset.py`: Dataset classes and data loading utilities

### Models
- `models/`: Contains model training scripts and weights
  - `__init__.py`: Package initialization 
  - `train_yolo.py`: YOLOv8 training script
  - `train_3dunet.py`: 3D UNet training for segmentation
  - `infer_yolo.py`: Inference utilities for YOLOv8
  - `yolo_v8/`: YOLOv8 model files and weights
  - `3dunet/`: 3D UNet model files and weights

### Datasets
- `yolo_dataset/`: Processed dataset for YOLO training
  - `dataset.yaml`: YOLO dataset configuration
  - `images/`: Preprocessed images
  - `labels/`: Corresponding labels
- `3dunet_dataset/`: Dataset for 3D UNet
  - `images/train/`: Training images
  - `images/val/`: Validation images
  - `labels/train/`: Training labels
  - `labels/val/`: Validation labels
- `3dunet_dataset_heatmap/`: Heatmap-based dataset for 3D UNet

### Configuration
- `config/`: Configuration files
  - `config.py`: Python configuration module
  - `config.yaml`: YAML configuration file for project settings

### Scripts
- `scripts/`: Executable scripts
  - `train.py`: Main training script
  - `infer.py`: Inference script for generating predictions
  - `test.py`: Testing and evaluation script

### Utilities
- `utils/`: Utility functions
  - `__init__.py`: Package initialization
  - `gpu_profiler.py`: GPU usage monitoring
  - `score.py`: Evaluation metrics
  - `visualization.py`: Visualization utilities

### Output
- `outputs/`: Results and visualizations
  - `submissions/`: Competition submissions
  - `3dunet/`: 3D UNet outputs
  - `data_description/`: Data analysis
  - `motor_visualization/`: Visualization of detected motors
  - `predict_samples/`: Sample predictions
  - `tomo_gif/`: Animated visualizations of tomography data

### Reference
- `reference/`: Reference notebooks and code
  - Jupyter notebooks with exploratory data analysis and model implementation examples

## Dependencies
This project requires the following packages:

### Main Dependencies
- Deep Learning & Computer Vision:
  - `torch` (>=2.0.1)
  - `torchvision` (>=0.15.0)
  - `ultralytics` (>=8.1.0) - YOLOv8
  - `monai` (>=1.4.0) - Medical imaging library
  - `opencv-python` (>=4.8.0)
  - `ignite` (>=1.1.0)

- Data Processing & Analysis:
  - `numpy` (>=1.24.0)
  - `pandas` (>=2.0.0)
  - `scikit-learn` (>=1.3.0)
  - `scipy` (>=1.11.0)
  - `scikit-image` (>=0.25.2)
  - `nibabel` (>=5.3.2) - Neuroimaging file formats

- Visualization:
  - `matplotlib` (>=3.7.0)
  - `plotly` (>=5.18.0)
  - `seaborn` (>=0.13.0)
  - `pillow` (>=10.0.0)

- 3D Geometry & Processing:
  - `plyfile` (>=1.1)
  - `trimesh` (>=4.6.4)
  - `networkx` (>=3.4.2)

- Utilities:
  - `pyyaml` (>=6.0.0)
  - `tqdm` (>=4.66.0)
  - `einops` (>=0.8.1)

### Development Dependencies
- `jupyterlab` (>=4.0.0)
- `ipywidgets` (>=8.0.0)
- `ipykernel` (>=6.29.5)
- `pytest` (>=7.4.0)

## Installation
All dependencies can be installed using the provided `pyproject.toml` file:

```bash
# Clone the repository
git clone https://github.com/yourusername/BYU_Locating_Bacterial_Flagellar_Motors_2025.git
cd BYU_Locating_Bacterial_Flagellar_Motors_2025

# Install dependencies
pip install -e .