# 图像embedding
import requests
import base64

with open('pokemon.jpeg', 'rb') as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

response = requests.post('http://localhost:8000/match',
                        json={
                            'texts': ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"],
                            'image_base64': image_base64
                        })
print(response.json())