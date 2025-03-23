import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imports import *

class Config:
    def __init__(self, config_path='config/config.yaml'):
        # 加载配置文件
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 初始化路径
        self.BASE_DIR = self.config['paths']['base_dir']
        self.DATA_DIR = os.path.join(self.BASE_DIR, self.config['paths']['data_dir'])
        self.TRAIN_CSV = os.path.join(self.DATA_DIR, self.config['paths']['train_csv'])
        self.TRAIN_DIR = os.path.join(self.DATA_DIR, self.config['paths']['train_dir'])
        self.TEST_DIR = os.path.join(self.DATA_DIR, self.config['paths']['test_dir'])
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR,self.config['paths']['output_dir'])
        self.MODEL_DIR = os.path.join(self.BASE_DIR,self.config['paths']['model_dir'])
        self.SUBMISSION_DIR = os.path.join(self.BASE_DIR,self.config['paths']['submission_dir'])
        # YOLO数据集目录
        self.YOLO_DATAESET_DIR = os.path.join(self.BASE_DIR,self.config['paths']['yolo_dataset_dir'])
        self.YOLO_IMAGES_TRAIN = os.path.join(self.YOLO_DATAESET_DIR, "images", "train")
        self.YOLO_IMAGES_VAL = os.path.join(self.YOLO_DATAESET_DIR, "images", "val")
        self.YOLO_LABELS_TRAIN = os.path.join(self.YOLO_DATAESET_DIR, "labels", "train")
        self.YOLO_LABELS_VAL = os.path.join(self.YOLO_DATAESET_DIR, "labels", "val")
        # YOLO模型位置
        self.YOLO_MODEL_DIR = os.path.join(self.BASE_DIR, self.config['paths']['yolo_model_dir'])
        self.YOLO_WEIGHTS_DIR = os.path.join(self.YOLO_MODEL_DIR, self.config['paths']['yolo_weights'])
        self.YOLO_BEST_MODEL_DIR = os.path.join(self.BASE_DIR, self.config['paths']['yolo_best_model_dir'])
        # UNET模型位置
        self.UNET_MODEL_DIR = os.path.join(self.BASE_DIR, self.config['paths']['3dunet_model_dir'])
        self.UNET_DATAESET_DIR = os.path.join(self.BASE_DIR, self.config['paths']['3dunet_dataset_dir'])
        self.UNET_IMAGES_TRAIN = os.path.join(self.UNET_DATAESET_DIR, "images", "train")
        self.UNET_IMAGES_VAL = os.path.join(self.UNET_DATAESET_DIR, "images", "val")
        self.UNET_LABELS_TRAIN = os.path.join(self.UNET_DATAESET_DIR, "labels", "train")
        self.UNET_LABELS_VAL = os.path.join(self.UNET_DATAESET_DIR, "labels", "val")

        # 创建必要的目录
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
            self.SUBMISSION_DIR
        ]
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
        
        # 设置随机种子
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

# 创建全局配置对象
config = Config()