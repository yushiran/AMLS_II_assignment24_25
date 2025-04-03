import gc
gc.collect()
# import plotly.express as px
from PIL import Image, ImageDraw
import random
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib import animation, rc; rc('animation', html='jshtml')
from ultralytics import YOLO
import yaml
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
import threading
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import math
import gc
import sys
import sklearn.metrics
import shutil
import tempfile
import nibabel as nib
import numpy as np
from monai.config import print_config
from monai.data import ArrayDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.handlers import (
    MeanDice,
    MLFlowHandler,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
)
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    Transposed,
    RandSpatialCropd
)
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.inferers import sliding_window_inference

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from monai.utils import first
from monai.losses import DiceCELoss
import scipy.ndimage
import ignite
from monai.metrics import DiceMetric
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config.config import config
from src.models import *
from src.utils import *
from src.data import *
print_config()