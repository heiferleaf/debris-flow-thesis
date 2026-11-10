"""ConvNeXt-T 迁移冒烟验证:只验证"预训练模型能加载、2 类头前向能出结果",不训练。

按 docs/process/experiments/复现分析-ConvNeXt.md 的方案执行:
  1. 加载 torchvision 自带 convnext_tiny(ImageNet-1k 预训练权重,首次运行自动下载约 110 MB);
  2. 换 2 类头,打印参数量;
  3. 从 train 侧抽 8 张(4 正:scene1 带框帧;4 负:dataset2 空标签图)做前向,
     打印概率与单张耗时。
输出只是链路验证,不是任何评估指标。

用法(仓库根目录):
    .venv\\Scripts\\python.exe experiments\\common\\convnext\\smoke_convnext.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

WEIGHTS = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def pick_positives(root: Path, n: int) -> list[Path]:
    files = sorted((root / "scene1" / "Annotated" / "images").glob("*.png"))
    return files[:n]


def pick_negatives(root: Path, n: int) -> list[Path]:
    """从 dataset2 的 labels 里找空 txt,映射回图片路径。"""
    out: list[Path] = []
    for txt in sorted((root / "labels").rglob("*.txt")):
        if txt.stat().st_size == 0:
            img = root / "images" / txt.relative_to(root / "labels").with_suffix(".jpg")
            if not img.exists():
                img = img.with_suffix(".png")
            if img.exists():
                out.append(img)
            if len(out) == n:
                break
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    ds3 = repo / "data" / "raw" / "dataset3"
    ds2 = repo / "data" / "raw" / "dataset2"

    print("[1/3] 加载 convnext_tiny(ImageNet-1k 预训练,首次运行会下载权重)…", flush=True)
    t0 = time.perf_counter()
    model = models.convnext_tiny(weights=WEIGHTS)
    n_full = sum(p.numel() for p in model.parameters())
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 2)  # 换 2 类头
    n_2cls = sum(p.numel() for p in model.parameters())
    model.eval()
    print(f"      1000 类头 {n_full:,} 参数 → 2 类头 {n_2cls:,} 参数,加载+换头 {time.perf_counter() - t0:.1f}s")

    pre = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    samples = [(p, 1) for p in pick_positives(ds3, 4)] + [(p, 0) for p in pick_negatives(ds2, 4)]
    if len(samples) != 8:
        raise SystemExit(f"抽样不足 8 张(实际 {len(samples)}),检查数据路径")

    print(f"[2/3] CPU 前向 8 张(4 正 + 4 负)…", flush=True)
    lat = []
    with torch.no_grad():
        for path, expect in samples:
            t0 = time.perf_counter()
            x = pre(Image.open(path).convert("RGB")).unsqueeze(0)
            prob = torch.softmax(model(x), dim=1)[0]
            dt = (time.perf_counter() - t0) * 1000
            lat.append(dt)
            pred = int(prob.argmax())
            mark = "✓" if pred == expect else "·"
            print(f"   {mark} 正={prob[1]:.3f} 负={prob[0]:.3f} 预测={'正' if pred else '负'}"
                  f"(期望{'正' if expect else '负'}) {dt:5.0f}ms  {path.name}", flush=True)

    print(f"[3/3] 单张前向平均 {sum(lat) / len(lat):.0f}ms(CPU,含预处理)")
    print("冒烟通过:预训练加载、2 类头、前向链路全部可用。")
    print("注意:以上概率来自未训练的随机 2 类头,无判别意义;本输出不是评估指标。")


if __name__ == "__main__":
    main()
