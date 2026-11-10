"""单图推理:用 EXP-005 正式基线的冻结 ConvNeXt-T 特征 + linear_head.pt 分类一张图。
用法: python infer_one.py <图片路径>"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "data" / "results" / "convnext" / "EXP-005-final"
IMG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\20443\AppData\Local\Temp\QQ_1789389438170.png")

head = nn.Linear(768, 2)
head.load_state_dict(torch.load(RUN / "linear_head.pt", map_location="cpu"))
head.eval()

model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(model.classifier[0], model.classifier[1])  # 768 维特征
model.eval()

pre = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

x = pre(Image.open(IMG).convert("RGB")).unsqueeze(0)
with torch.no_grad():
    feat = model(x)
    prob = torch.softmax(head(feat), 1)[0]
print(f"P(正=泥石流) = {prob[1].item():.4f}")
print(f"P(负=非泥石流) = {prob[0].item():.4f}")
print(f"判定(阈值0.5): {'正样本' if prob[1] >= 0.5 else '负样本'}")
