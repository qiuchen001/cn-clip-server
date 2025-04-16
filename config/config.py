import os
from dotenv import load_dotenv
load_dotenv()


class Config:
    # 模型配置
    MODEL_BASE_DIR = os.getenv('MODEL_BASE_DIR', 'models')
    CN_CLIP_MODEL_PATH = os.path.join(
        MODEL_BASE_DIR,
        'embedding',
        'cn-clip',
        'clip_cn_vit-l-14-336.pt'
    )
