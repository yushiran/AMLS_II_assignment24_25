import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import random
import numpy as np
import yaml

class Config:
    """
    Config class for managing project configurations and paths.
    This class is responsible for loading configuration settings from a YAML file,
    initializing directory paths, setting up random seeds, and configuring device
    settings for training and inference.
    Attributes:
        BASE_DIR (str): Base directory for the project.
        DATA_DIR (str): Directory for storing data.
        TRAIN_CSV (str): Path to the training CSV file.
        TRAIN_DIR (str): Directory for training data.
        TEST_DIR (str): Directory for test data.
        OUTPUT_DIR (str): Directory for storing output files.
        MODEL_DIR (str): Directory for storing models.
        SUBMISSION_DIR (str): Directory for storing submission files.
        YOLO_DATAESET_DIR (str): Directory for YOLO dataset.
        YOLO_IMAGES_TRAIN (str): Directory for YOLO training images.
        YOLO_IMAGES_VAL (str): Directory for YOLO validation images.
        YOLO_LABELS_TRAIN (str): Directory for YOLO training labels.
        YOLO_LABELS_VAL (str): Directory for YOLO validation labels.
        YOLO_MODEL_DIR (str): Directory for YOLO model files.
        YOLO_WEIGHTS_DIR (str): Directory for YOLO weights.
        YOLO_BEST_MODEL_DIR (str): Directory for the best YOLO model.
        UNET_MODEL_DIR (str): Directory for 3D UNet model files.
        UNET_DATAESET_DIR (str): Directory for 3D UNet dataset.
        UNET_IMAGES_TRAIN (str): Directory for 3D UNet training images.
        UNET_IMAGES_VAL (str): Directory for 3D UNet validation images.
        UNET_LABELS_TRAIN (str): Directory for 3D UNet training labels.
        UNET_LABELS_VAL (str): Directory for 3D UNet validation labels.
        UNET_OUTPUT_DIR (str): Directory for 3D UNet output files.
        RANDOM_SEED (int): Random seed for reproducibility.
        DEVICE (str): Device to use for computation ('cuda:0' or 'cpu').
        INFER_BATCH_SIZE (int): Batch size for inference.
        CONFIFENCE_THRESHOLD (float): Confidence threshold for inference.
        MAX_DETECTIONS_PER_TOMO (int): Maximum detections per tomogram.
        NMS_IOU_THRESHOLD (float): Non-maximum suppression IoU threshold.
        CONCENTRATION (float): Fraction of slices to process for fast submission.
    Methods:
        __init__(config_path='src/config/config.yaml'):
            Initializes the Config object by loading the YAML configuration file,
            setting up paths, creating necessary directories, and configuring device
            and random seed settings.
        set_seed(seed):
            Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Note:
        Ensure that the YAML configuration file exists at the specified path and
        contains the required keys for paths and parameters.
    """

    def __init__(self, config_path='src/config/config.yaml'):
        # Load configuration file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize paths
        self.BASE_DIR = self.config['paths']['base_dir']
        self.DATA_DIR = os.path.join(self.BASE_DIR, self.config['paths']['data_dir'])
        self.TRAIN_CSV = os.path.join(self.DATA_DIR, self.config['paths']['train_csv'])
        self.TRAIN_DIR = os.path.join(self.DATA_DIR, self.config['paths']['train_dir'])
        self.TEST_DIR = os.path.join(self.DATA_DIR, self.config['paths']['test_dir'])
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR,self.config['paths']['output_dir'])
        self.MODEL_DIR = os.path.join(self.BASE_DIR,self.config['paths']['model_dir'])
        self.SUBMISSION_DIR = os.path.join(self.BASE_DIR,self.config['paths']['submission_dir'])
        # YOLO dataset directory
        self.YOLO_DATAESET_DIR = os.path.join(self.BASE_DIR,self.config['paths']['yolo_dataset_dir'])
        self.YOLO_IMAGES_TRAIN = os.path.join(self.YOLO_DATAESET_DIR, "images", "train")
        self.YOLO_IMAGES_VAL = os.path.join(self.YOLO_DATAESET_DIR, "images", "val")
        self.YOLO_LABELS_TRAIN = os.path.join(self.YOLO_DATAESET_DIR, "labels", "train")
        self.YOLO_LABELS_VAL = os.path.join(self.YOLO_DATAESET_DIR, "labels", "val")
        # YOLO model location
        self.YOLO_MODEL_DIR = os.path.join(self.BASE_DIR, self.config['paths']['yolo_model_dir'])
        self.YOLO_WEIGHTS_DIR = os.path.join(self.YOLO_MODEL_DIR, self.config['paths']['yolo_weights'])
        self.YOLO_BEST_MODEL_DIR = os.path.join(self.BASE_DIR, self.config['paths']['yolo_best_model_dir'])
        # UNET model location
        self.UNET_MODEL_DIR = os.path.join(self.BASE_DIR, self.config['paths']['3dunet_model_dir'])
        self.UNET_DATAESET_DIR = os.path.join(self.BASE_DIR, self.config['paths']['3dunet_dataset_dir'])
        self.UNET_IMAGES_TRAIN = os.path.join(self.UNET_DATAESET_DIR, "images", "train")
        self.UNET_IMAGES_VAL = os.path.join(self.UNET_DATAESET_DIR, "images", "val")
        self.UNET_LABELS_TRAIN = os.path.join(self.UNET_DATAESET_DIR, "labels", "train")
        self.UNET_LABELS_VAL = os.path.join(self.UNET_DATAESET_DIR, "labels", "val")
        self.UNET_OUTPUT_DIR = os.path.join(self.BASE_DIR, self.config['paths']['3dunet_output_dir'])

            # Create necessary directories
        dirs_to_create = [
            self.OUTPUT_DIR,
            self.MODEL_DIR,
            self.YOLO_DATAESET_DIR,
            self.YOLO_IMAGES_TRAIN,
            self.YOLO_IMAGES_VAL,
            self.YOLO_LABELS_TRAIN,
            self.YOLO_LABELS_VAL,
            self.YOLO_MODEL_DIR,
            self.YOLO_WEIGHTS_DIR,
            self.YOLO_BEST_MODEL_DIR,
            self.SUBMISSION_DIR,
            self.UNET_MODEL_DIR,
            self.UNET_DATAESET_DIR,
            self.UNET_IMAGES_TRAIN,
            self.UNET_IMAGES_VAL,
            self.UNET_LABELS_TRAIN,
            self.UNET_LABELS_VAL,
            self.UNET_OUTPUT_DIR,
        ]
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
        
        # Set random seed
        self.RANDOM_SEED = self.config['random_seed']
        self.set_seed(self.RANDOM_SEED)

        # Set device and dynamic batch size
        self.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.INFER_BATCH_SIZE = 8
        if self.DEVICE.startswith('cuda'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
            free_mem = gpu_mem - torch.cuda.memory_allocated(0) / 1e9
            self.INFER_BATCH_SIZE = max(8, min(32, int(free_mem * 4)))
            print(f"Dynamic batch size set to {self.INFER_BATCH_SIZE} based on {free_mem:.2f}GB free memory")
        else:
            print("GPU not available, using CPU")
            self.INFER_BATCH_SIZE = 4
        
        # set inference parameters
        self.CONFIFENCE_THRESHOLD = self.config['infer_params']['confidence_threshold']
        self.MAX_DETECTIONS_PER_TOMO = self.config['infer_params']['max_detections_per_tomo']   
        self.NMS_IOU_THRESHOLD = self.config['infer_params']['nms_iou_threshold']       
        self.CONCENTRATION = self.config['infer_params']['concentration']          # Process a fraction of slices for fast submission


    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

# Create a global configuration object
config = Config()