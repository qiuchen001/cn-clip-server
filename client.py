# 图像embedding
import requests
import base64
from pathlib import Path

# # 方式1: 通过图片URL调用
# print("=== 通过URL调用embedding ===")
# response = requests.post('http://localhost:8000/embeddings/image',
#                          json={
#                              'image_url': 'http://10.66.8.51:9000/perception-mining/876e8f38-9ee4-48eb-bdf7-5937d9dcc5ad_frame_9.jpg'
#                          })
# print(response.json())
#
# # 方式2: 通过base64调用
# print("\n=== 通过base64调用embedding ===")
# 读取本地图片文件并转换为base64
with open('pokemon.jpeg', 'rb') as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
#
# response = requests.post('http://localhost:8000/embeddings/image',
#                         json={
#                             'image_base64': image_base64
#                         })
# print(response.json())

# # match接口调用示例
# print("\n=== 通过URL调用match ===")
# response = requests.post('http://localhost:8000/match',
#                         json={
#                             'texts': ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"],
#                             'image_url': 'https://img2.baidu.com/it/u=2204846193,1434649112&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=980'
#                             # 'image_url': 'http://10.66.8.51:9000/perception-mining/876e8f38-9ee4-48eb-bdf7-5937d9dcc5ad_frame_9.jpg'
#                         })
# print(response.json())

# print("\n=== 通过base64调用match ===")
# response = requests.post('http://localhost:8000/match',
#                         json={
#                             'texts': ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"],
#                             'image_base64': image_base64
#                         })
# print(response.json())

# 统一的embeddings接口示例
import requests
import base64

# 准备base64图片数据
with open('pokemon.jpeg', 'rb') as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

# 混合输入示例
request_data = {
    "inputs": [
        # 文本输入
        {
            "text": "杰尼龟"
        },
        {
            "text": "皮卡丘"
        },
        # URL图片输入
        {
            "image": {
                "image_url": "https://img2.baidu.com/it/u=2204846193,1434649112&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=980"
            }
        },
        # base64图片输入
        {
            "image": {
                "image_base64": image_base64
            }
        }
    ]
}

# 调用统一的embeddings接口
print("=== 调用统一的embeddings接口 ===")
response = requests.post('http://localhost:8000/embeddings',
                         json=request_data)
print(response.json())
