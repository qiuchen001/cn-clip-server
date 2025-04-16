import requests
from typing import List, Union
from pathlib import Path
import base64
import json


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

    try:
        if image_path:
            # 使用文件上传时的请求
            with open(image_path, 'rb') as f:
                files = {'file': f}
                # 直接将texts作为form字段发送
                response = requests.post(
                    'http://localhost:8000/match',
                    data={'texts': json.dumps(texts)},  # 将texts列表转为JSON字符串
                    files=files
                )
        else:
            # 使用URL或base64时的请求
            data = {
                "texts": texts,
                "image_url": image_url,
                "image_base64": image_base64
            }
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


# 使用示例
if __name__ == "__main__":
    # # 示例1: 使用本地图片
    # try:
    #     scores = compare_image_texts(
    #         texts=["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"],
    #         image_path="pokemon.jpeg"
    #     )
    #     print("本地图片的相似度分数:", scores)
    # except Exception as e:
    #     print(f"错误: {e}")

    # 示例2: 使用图片URL
    try:
        scores = compare_image_texts(
            texts=[
                "猫咪", 
                "狗狗", 
                "宠物", 
                "可爱动物",
                "户外风景",
                "室内场景"
            ],
            image_url="https://t7.baidu.com/it/u=1595072465,3644073269&fm=193&f=GIF"
        )
        print("\n在线图片的相似度分数:", scores)
    except Exception as e:
        print(f"错误: {e}")