# 图像embedding
import requests
import base64

# 方式1: 通过图片URL调用
print("=== 通过URL调用 ===")
response = requests.post('http://localhost:8000/embeddings/image',
                         json={
                             'image_url': 'http://10.66.8.51:9000/perception-mining/876e8f38-9ee4-48eb-bdf7-5937d9dcc5ad_frame_9.jpg'
                         })
print(response.json())

# 方式2: 通过base64调用
print("\n=== 通过base64调用 ===")
# 读取本地图片文件并转换为base64
with open('first_frame.png', 'rb') as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

response = requests.post('http://localhost:8000/embeddings/image',
                        json={
                            'image_base64': image_base64
                        })
print(response.json())
