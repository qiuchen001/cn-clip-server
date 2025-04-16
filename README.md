# CN-CLIP Server

基于CN-CLIP的图文特征提取和匹配服务。

## 功能特性

- 图像特征提取
- 文本特征提取
- 图文匹配度计算
- 支持URL和base64两种图片输入方式

## Docker部署

### 1. 准备工作

1. 安装Docker和NVIDIA Container Toolkit
2. 准备CN-CLIP模型文件
3. 创建模型目录:
```bash
mkdir -p /path/to/models
# 将模型文件clip_cn_vit-l-14-336.pt复制到该目录
```

### 2. 构建镜像

```bash
# 在项目根目录下执行
docker build -t cn-clip-server .
```

### 3. 运行容器

```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v /path/to/models:/models \
  -e CN_CLIP_MODEL_PATH=/models/clip_cn_vit-l-14-336.pt \
  --name cn-clip-server \
  cn-clip-server
```

参数说明:
- `--gpus all`: 启用GPU支持
- `-p 8000:8000`: 端口映射
- `-v /path/to/models:/models`: 挂载模型目录
- `-e CN_CLIP_MODEL_PATH`: 设置模型路径
- `--name`: 容器名称

### 4. 验证服务

```bash
# 测试服务是否正常运行
curl -X POST "http://localhost:8000/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"text": "测试文本"}
    ]
  }'
```

## 环境变量配置

- `CN_CLIP_MODEL_PATH`: 模型文件路径
- `DEVICE`: 计算设备 (cuda/cpu)

## API文档

服务启动后,访问 http://localhost:8000/docs 查看完整的API文档。

## 注意事项

1. 确保模型文件正确挂载
2. 使用GPU时需要安装NVIDIA Container Toolkit
3. 可以根据需要调整配置文件中的参数
