import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.imports import *

if __name__ == '__main__':
    infer_yolo.yolo_inference_main()