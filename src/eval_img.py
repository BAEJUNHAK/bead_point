# src/eval_img.py
import os
import argparse
import cv2
import torch
from torch.utils.data import DataLoader

from src.data.dataset_img import BeadImageKeypointDataset
from src.models import LitKeypoint2
from src.losses.heatmap_utils import heatmaps_to_coords_argmax


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--pkl_dir", required=True)
    ap.add_argument("--ann_csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", default="runs/eval_vis")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds = BeadImageKeypointDataset(
        img_dir=args.img_dir, pkl_dir=args.pkl_dir,
        size=args.size, annotation_csv=args.ann_csv,
        auto_blue_min=70, auto_diff_min=30
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    model = LitKeypoint2.load_from_checkpoint(args.ckpt)
    model.eval().cuda() if torch.cuda.is_available() else model.eval()

    all_mae = []
    all_hit = []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            img = batch["image"]
            gt = batch["coord_px"].float()
            polyline_mask = batch["polyline_mask"]  # [B,H,W] - 폴리라인 제약조건
            dev = next(model.parameters()).device
            img = img.to(dev)
            polyline_mask = polyline_mask.to(dev)

            logit = model(img)
            
            # 폴리라인 영역만 활성화 (제약조건 적용)
            masked_logit = logit * polyline_mask.unsqueeze(1)
            prob = torch.sigmoid(masked_logit)

            # argmax (훈련과 동일한 방식)
            pred = heatmaps_to_coords_argmax(prob).cpu()  # [B,2,2]

            mae_per_point = torch.linalg.norm(pred - gt, dim=-1)  # [B,2] - 유클리드 거리
            mae_per_sample = mae_per_point.mean(dim=-1)  # [B] - 각 샘플의 평균 MAE
            all_mae.extend(mae_per_sample.numpy().tolist())

            dist = torch.linalg.norm(pred - gt, dim=-1)  # [B,2]
            hit = (dist <= model.pck_thresh_px).float().mean(dim=-1)
            all_hit.extend(hit.numpy().tolist())

            # 선택: 앞 몇 장 오버레이 저장
            if i == 0:
                for b in range(min(8, img.size(0))):
                    rgb = (img[b].cpu().numpy().transpose(1,2,0) * 255).astype("uint8")
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    for (x,y), color in [(gt[b,0].numpy(), (0,0,255)), (gt[b,1].numpy(), (0,255,0))]:
                        cv2.drawMarker(bgr, (int(x),int(y)), color, cv2.MARKER_TILTED_CROSS, 18, 2)
                    for (x,y), color in [(pred[b,0].numpy(), (255,0,0)), (pred[b,1].numpy(), (0,165,255))]:
                        cv2.circle(bgr, (int(x),int(y)), 4, color, 2)
                    cv2.imwrite(os.path.join(args.out, f"overlay_{b}.png"), bgr)

    import numpy as np
    print(f"MAE(px): mean={np.mean(all_mae):.2f} | PCK@{model.pck_thresh_px}px: {np.mean(all_hit)*100:.1f}%")
    print(f"overlays saved to {args.out}")


if __name__ == "__main__":
    main()
