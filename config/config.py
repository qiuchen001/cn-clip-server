import os
from dotenv import load_dotenv
load_dotenv()


class Config:
    # 模型配置
    MODEL_BASE_DIR = os.getenv('MODEL_BASE_DIR', 'models')
    CN_CLIP_MODEL_PATH = os.getenv('CN_CLIP_MODEL_PATH', '/models/clip_cn_vit-l-14-336.pt')

    # 其他配置项
    IMAGE_SIZE = 336  # ViT-L-14-336的输入分辨率
    DEVICE = os.getenv('DEVICE', 'cuda')  # 可以通过环境变量配置使用CPU或GPU
