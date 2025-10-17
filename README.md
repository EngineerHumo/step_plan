# Permutation-Invariant Surgical Sub-Region Segmentation

（中文简介在下，English follows.）

## 项目简介（Chinese Overview）
本仓库实现了一个面向手术分期规划的 2D 医学图像子区域自动划分框架。模型采用 **集合预测 (Set Prediction)** 思路，利用固定数量的可学习查询直接预测若干子区域掩膜与存在性得分，无需显式血管分割即可隐式学习血管走向。训练阶段通过匈牙利匹配保证标签置换不变，仅在“可手术区域”内计算损失，同时引入互斥、平滑、边界对齐、禁区惩罚等先验约束，从而得到解剖合理、互不重叠的子区域。

## Project Overview (English)
This repository provides a permutation-invariant set prediction baseline for surgical sub-region planning on 2D medical images. A SegFormer-style encoder and a lightweight query-to-mask decoder predict up to `K_max` instance masks together with existence confidences. Hungarian matching keeps training permutation-free while losses emphasise intra-region compactness, inter-region exclusivity, smooth boundaries aligned with vasculature cues, and penalties for forbidden areas. All inference operators are ONNX-friendly, enabling direct export to deployment environments.

---

## 数据格式 / Data Format
- 数据以 JSON (`items.json`) 描述，每个元素形如：
  ```json
  {
    "image": "path/to/image.png",
    "aux_masks": [
      "path/to/operable.png",
      "path/to/operated.png",
      "path/to/forbidden.png",
      "path/to/irrelevant.png"
    ],
    "gt_masks": [
      "path/to/subregion_0.png",
      "path/to/subregion_1.png"
    ]
  }
  ```
- 上游掩膜顺序必须是：可手术 / 已手术 / 禁止 / 无关。
- 子区域标签按照质心从左到右的顺序存储；训练过程中通过匈牙利算法自动匹配，无需语义编号。
- 所有图像与掩膜在加载时会被调整到 `Config.img_size`（默认 512×512）。

## 快速开始 / Quick Start
1. 安装依赖：`pip install -r requirements.txt`
2. 准备 `train_items.json` 与 `val_items.json`（或使用 `--synthetic` 生成随机数据进行冒烟测试）。
3. 运行训练脚本：
   ```bash
   bash scripts/train.sh
   ```
4. 导出 ONNX 模型：
   ```bash
   bash scripts/export_onnx.sh
   ```

### Synthetic Smoke Test
使用 `--synthetic` 可以自动生成随机样本，验证训练与导出流程不会报错：
```bash
python -u train.py --synthetic --synthetic_samples 8 --epochs 1 --save_dir runs/debug
```

## 模型结构 / Model Architecture
- **InputFusion**：将原始图像与 4 通道上游掩膜融合为 3 通道输入。
- **Backbone**：默认使用 Hugging Face Transformers 的 SegFormer-B3 (`nvidia/mit-b3`)，若不可用则回退到轻量卷积金字塔。
- **Pixel Decoder**：多尺度特征自顶向下融合，输出低分辨率像素嵌入。
- **Mask Decoder**：`K` 个可学习查询通过标准 Multi-Head Attention 获取上下文，并预测掩膜原型与存在性得分。
- **Inference Output**：`K` 张子区域概率图 + `K` 个存在性概率，可根据阈值灵活选择实际子区域数。

## 训练要点 / Training Highlights
- 损失仅在“可手术 ROI”内累积：Dice + BCE、互斥惩罚、总变差平滑、边界对齐、禁区重叠惩罚、面积先验、存在性与基数约束。
- 匹配代价由 Dice 与 BCE 组合，通过匈牙利算法保持置换不变。
- 训练脚本支持断点续训（`--resume`）与固定随机种子（`--seed`）。

## 推理与后处理 / Inference & Post-processing
- ONNX 模型输出 `mask_probs` 与 `exist_scores`，可在部署端按 0.5 阈值二值化并去除 ROI 内较小的连通域（如 <0.2% ROI 面积）。
- 显示或存储时，可按照子区域质心的 x 坐标重新排序，便于医生按“从左到右”查看。

## ONNX 导出 / ONNX Export
- `export_onnx.py` 使用 `torch.onnx.export`（opset >= 17），确保推理图仅包含 Conv/Linear/LayerNorm/Softmax/MatMul/Upsample 等标准算子。
- 脚本自动用 `onnxruntime` 载入导出的模型并进行一次前向推理 sanity check。
- 默认固定输入尺寸（512×512）；如需动态尺寸，可自行修改导出脚本的 `dynamic_axes`。

## 目录结构 / Repository Layout
```
.
├─ README.md
├─ requirements.txt
├─ config.py
├─ dataset.py
├─ model_parts.py
├─ matcher.py
├─ losses.py
├─ train.py
├─ export_onnx.py
├─ utils/
│  ├─ metrics.py
│  └─ seed.py
└─ scripts/
   ├─ train.sh
   └─ export_onnx.sh
```

## 备注 / Notes
- 推理路径完全不依赖训练阶段的匈牙利匹配或损失计算，保证 ONNX 导出的可行性。
- 若环境中无法安装 `transformers`，自动切换到内置的 CNN 备选骨干网络（无预训练）。
- 项目遵循 PyTorch ≥ 2.0、Python ≥ 3.10，且避免使用自定义 CUDA 运算。

欢迎根据具体医疗场景调整损失权重与后处理策略。
