"""Export trained model to ONNX and validate with onnxruntime."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import onnxruntime as ort

from config import Config
from train import FullModel, _select_device


class InferenceWrapper(torch.nn.Module):
    def __init__(self, model: FullModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, aux_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_logits, exist_logits, _ = self.model(image, aux_masks)
        return torch.sigmoid(mask_logits), torch.sigmoid(exist_logits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--onnx_out", type=str, required=True, help="Output ONNX file")
    parser.add_argument("--device", type=str, default=None, help="Device for export")
    parser.add_argument("--img_size", type=int, nargs=2, default=None, help="Image size H W")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.img_size is not None:
        cfg.img_size = (int(args.img_size[0]), int(args.img_size[1]))
    if args.device is not None:
        cfg.device = args.device
    cfg.device = _select_device(cfg.device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    model = FullModel(cfg).to(cfg.device)
    checkpoint = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    wrapper = InferenceWrapper(model).to(cfg.device)
    H, W = cfg.img_size
    dummy_img = torch.randn(1, cfg.in_channels, H, W, device=cfg.device)
    dummy_aux = torch.randn(1, cfg.aux_mask_channels, H, W, device=cfg.device)
    onnx_path = Path(args.onnx_out)
    torch.onnx.export(
        wrapper,
        (dummy_img, dummy_aux),
        onnx_path.as_posix(),
        input_names=["image", "aux_masks"],
        output_names=["mask_probs", "exist_scores"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"Exported ONNX model to {onnx_path}")

    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    outputs = session.run(
        None,
        {
            "image": dummy_img.cpu().numpy(),
            "aux_masks": dummy_aux.cpu().numpy(),
        },
    )
    print("ONNXRuntime outputs:", [tuple(o.shape) for o in outputs])


if __name__ == "__main__":
    main()
