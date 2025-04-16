# 图像embedding
import requests
# files = {'file': open('first_frame.png', 'rb')}
# response = requests.post('http://localhost:8000/embeddings/image', files=files)
# print(response.json())

response = requests.post('http://localhost:8000/embeddings/image',
                         json={
                             'image_url': 'http://10.66.8.51:9000/perception-mining/876e8f38-9ee4-48eb-bdf7-5937d9dcc5ad_frame_9.jpg'
                         })
print(response.json())

# 文本embedding
response = requests.post('http://localhost:8000/embeddings/text',
                        json={'text': '一只猫'})
print(response.json())

# 图文匹配
response = requests.post('http://localhost:8000/match',
                        json={
                            'texts': ['猫', '狗', '鸟'],
                            'image_url': 'http://10.66.8.51:9000/perception-mining/876e8f38-9ee4-48eb-bdf7-5937d9dcc5ad_frame_9.jpg'
                        })
print(response.json())