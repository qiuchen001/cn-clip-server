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
