# src/export_onnx.py
import argparse
import torch
from src.lit_model import LitKeypoint2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="학습된 Lightning ckpt")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="bead_kp2.onnx")
    return ap.parse_args()

def main():
    args = parse_args()
    model = LitKeypoint2.load_from_checkpoint(args.ckpt)
    net = model.net.eval().cpu()  # 네트워크만 ONNX로

    dummy = torch.randn(1, 3, args.size, args.size)
    torch.onnx.export(
        net, dummy, args.out,
        input_names=["images"], output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14
    )
    print(f"exported: {args.out}")

if __name__ == "__main__":
    main()
