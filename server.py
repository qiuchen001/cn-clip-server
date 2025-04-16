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


# 新增请求模型
# class ImageRequest(BaseModel):
#     image_url: Optional[str] = None
#     image_base64: Optional[str] = None

class ImageRequest(BaseModel):
    image_url: str


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
        processed_image = self.processor(image).unsqueeze(0).to(self.device)
        text = self.tokenizer(texts).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(processed_image)
            text_features = self.model.encode_text(text)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = image_features @ text_features.t()
            similarity = (similarity + 1) / 2
            similarity = similarity.softmax(dim=-1)

            return {text: float(score) for text, score in zip(texts, similarity[0].cpu().numpy())}


# 创建服务实例
clip_service = CNClipService()


# API路由
@app.post("/embeddings/image")
# async def image_embedding(request: Optional[ImageRequest] = None, file: Optional[UploadFile] = File(None)):
async def image_embedding(request: Optional[ImageRequest] = None):
    """生成图像的embedding向量
    支持三种图片输入方式:
    1. 文件上传
    2. 图片URL
    3. Base64编码
    """
    try:
        # 获取图像
        if file:
            content = await file.read()
            image = Image.open(io.BytesIO(content))
        elif request and request.image_url:
            response = requests.get(request.image_url)
            image = Image.open(io.BytesIO(response.content))
        elif request and request.image_base64:
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data))
        else:
            raise HTTPException(status_code=400, detail="需要提供图像(支持文件上传、URL或Base64)")
            
        # 生成embedding
        embedding = await clip_service.process_image(image)
        return JSONResponse({
            "success": True,
            "embedding": embedding
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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
async def match(request: Optional[ImageRequest] = None, file: Optional[UploadFile] = File(None)):
    """计算图文匹配相似度"""
    try:
        # 获取图像
        # if file:
        #     content = await file.read()
        #     image = Image.open(io.BytesIO(content))
        # elif request.image_url:
        #     response = requests.get(request.image_url)
        #     image = Image.open(io.BytesIO(response.content))
        # elif request.image_base64:
        #     image_data = base64.b64decode(request.image_base64)
        #     image = Image.open(io.BytesIO(image_data))
        # else:
        #     raise HTTPException(status_code=400, detail="需要提供图像")

        if request.image_url:
            response = requests.get(request.image_url)
            image = Image.open(io.BytesIO(response.content))
        else:
            raise HTTPException(status_code=400, detail="需要提供图像")



        # 计算匹配度
        scores = await clip_service.match(image, request.texts)
        return JSONResponse({
            "success": True,
            "scores": scores
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def compare_image_texts(texts: List[str],
                        image_path: Union[str, Path] = None,
                        image_url: str = None,
                        image_base64: str = None) -> dict:
    """
    计算一张图片与多个文本的相似度

    参数:
        texts: 文本列表
        image_path: 本地图片路径
        image_url: 图片URL
        image_base64: 图片的base64编码

    返回:
        dict: 每个文本对应的相似度分数
    """
    if not texts:
        raise ValueError("texts不能为空")

    # 准备请求数据
    data = {
        "texts": texts,
        "image_url": image_url,
        "image_base64": image_base64
    }
    
    try:
        if image_path:
            # 使用文件上传时的请求
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    'http://localhost:8000/match',
                    data={'request': json.dumps(data)},  # 将json数据作为form字段发送
                    files=files
                )
        else:
            # 使用URL或base64时的请求
            response = requests.post(
                'http://localhost:8000/match',
                json=data
            )

        response.raise_for_status()
        result = response.json()
        
        if result.get("success"):
            return result["scores"]
        else:
            raise Exception(result.get("detail", "未知错误"))

    except requests.exceptions.RequestException as e:
        raise Exception(f"请求失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)