import torch
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = r"E:\playground\ai\projects\vision-perception-sample\models\embedding\cn-clip\clip_cn_vit-l-14-336.pt"

model, preprocess = load_from_name(
            name=model_path,
            device=device,
            vision_model_name="ViT-L-14-336",
            text_model_name="RoBERTa-wwm-ext-base-chinese",
            input_resolution=336)
model.eval()
image = preprocess(Image.open("./pokemon.jpeg")).unsqueeze(0).to(device)
text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]