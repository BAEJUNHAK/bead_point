# MAE, PCK
import numpy as np

def mae_px(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    # [N,2,2]
    d = np.linalg.norm(pred_xy - gt_xy, axis=-1)
    return d.mean().item()

def pck_score(pred_xy: np.ndarray, gt_xy: np.ndarray, thresh_px: float) -> float:
    d = np.linalg.norm(pred_xy - gt_xy, axis=-1)
    return (d <= thresh_px).mean().item()
