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
        # YOLO数据集目录
        self.YOLO_DATAESET_DIR = os.path.join(self.BASE_DIR,self.config['paths']['yolo_dataset_dir'])
        self.YOLO_IMAGES_TRAIN = os.path.join(self.YOLO_DATAESET_DIR, "images", "train")
        self.YOLO_IMAGES_VAL = os.path.join(self.YOLO_DATAESET_DIR, "images", "val")
        self.YOLO_LABELS_TRAIN = os.path.join(self.YOLO_DATAESET_DIR, "labels", "train")
        self.YOLO_LABELS_VAL = os.path.join(self.YOLO_DATAESET_DIR, "labels", "val")
        # YOLO模型位置
        self.YOLO_MODEL_DIR = os.path.join(self.BASE_DIR, self.config['paths']['yolo_model_dir'])
        self.YOLO_WEIGHTS_DIR = os.path.join(self.YOLO_MODEL_DIR, self.config['paths']['yolo_weights'])

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
            self.YOLO_WEIGHTS_DIR
        ]
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)

        # 初始化设备
        self.DEVICE = torch.device(self.config['device'])
        print(f"Using device: {self.DEVICE}")

        # 设置随机种子
        self.RANDOM_SEED = self.config['random_seed']
        self.set_seed(self.RANDOM_SEED)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

# 创建全局配置对象
config = Config()