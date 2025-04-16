from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Union
import torch
import cn_clip.clip as clip
from PIL import Image
import io
import requests
import base64
from config.config import Config
import json
from pathlib import Path

app = FastAPI(
    title="CN-CLIP API",
    description="CN-CLIP模型的HTTP API服务",
    version="1.0.0"
)


# 请求模型
class TextRequest(BaseModel):
    text: str


class MatchRequest(BaseModel):
    texts: List[str]
    image_url: Optional[str] = None
    image_base64: Optional[str] = None

    @property
    def has_valid_input(self):
        return bool(self.image_url) != bool(self.image_base64)  # 确保只提供了一种输入


class ImageRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None

    @property
    def has_valid_input(self):
        return bool(self.image_url) != bool(self.image_base64)  # 确保只提供了一种输入


# CN-CLIP服务类
class CNClipService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = clip.load_from_name(
            name=Config.CN_CLIP_MODEL_PATH,
            device=self.device,
            vision_model_name="ViT-L-14-336",
            text_model_name="RoBERTa-wwm-ext-base-chinese",
            input_resolution=336
        )
        self.model.eval()
        self.tokenizer = clip.tokenize

    async def process_image(self, image: Image.Image):
        process_image = self.processor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(process_image)
            return image_features[0].detach().cpu().numpy().tolist()

    async def process_text(self, text: str):
        text = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            return text_features[0].detach().cpu().numpy().tolist()

    async def match(self, image: Image.Image, texts: List[str]):
        """计算图文匹配相似度,使用CN-CLIP原生的计算方式"""
        processed_image = self.processor(image).unsqueeze(0).to(self.device)
        text = self.tokenizer(texts).to(self.device)

        with torch.no_grad():
            # 使用模型原生的相似度计算方法
            logits_per_image, logits_per_text = self.model.get_similarity(processed_image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            # 将概率值与文本对应
            return {text: float(prob) for text, prob in zip(texts, probs[0])}


# 创建服务实例
clip_service = CNClipService()


# API路由
@app.post("/embeddings/image")
async def image_embedding(request: ImageRequest):
    """生成图像的embedding向量
    支持两种图片输入方式:
    1. 图片URL
    2. Base64编码
    """
    try:
        if not request.has_valid_input:
            raise HTTPException(status_code=400, detail="必须且只能提供一种图片输入方式: image_url 或 image_base64")

        # 获取图像
        if request.image_url:
            try:
                response = requests.get(request.image_url)
                response.raise_for_status()  # 检查请求是否成功
                image = Image.open(io.BytesIO(response.content))
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=400, detail=f"获取图片URL失败: {str(e)}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"处理URL图片失败: {str(e)}")
        else:  # 使用base64
            try:
                image_data = base64.b64decode(request.image_base64)
                image = Image.open(io.BytesIO(image_data))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"处理base64图片失败: {str(e)}")

        # 生成embedding
        embedding = await clip_service.process_image(image)
        return JSONResponse({
            "success": True,
            "embedding": embedding
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@app.post("/embeddings/text")
async def text_embedding(request: TextRequest):
    """生成文本的embedding向量"""
    try:
        embedding = await clip_service.process_text(request.text)
        return JSONResponse({
            "success": True,
            "embedding": embedding
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/match")
async def match(request: MatchRequest):
    """计算图文匹配相似度
    支持两种图片输入方式:
    1. 图片URL
    2. Base64编码
    """
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="texts不能为空")

        if not request.has_valid_input:
            raise HTTPException(status_code=400, detail="必须且只能提供一种图片输入方式: image_url 或 image_base64")

        # 获取图像
        if request.image_url:
            try:
                response = requests.get(request.image_url)
                response.raise_for_status()  # 检查请求是否成功
                image = Image.open(io.BytesIO(response.content))
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=400, detail=f"获取图片URL失败: {str(e)}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"处理URL图片失败: {str(e)}")
        else:  # 使用base64
            try:
                image_data = base64.b64decode(request.image_base64)
                image = Image.open(io.BytesIO(image_data))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"处理base64图片失败: {str(e)}")

        # 计算匹配度
        scores = await clip_service.match(image, request.texts)
        return JSONResponse({
            "success": True,
            "scores": scores
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
