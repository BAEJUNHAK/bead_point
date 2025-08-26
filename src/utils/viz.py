# 이미지 위 GT/Pred 점/히트맵 시각화
import os, cv2, numpy as np

def draw_points(img_bgr: np.ndarray, pts_xy, color_gt=(0,0,255), color_pred=(0,255,0)):
    out = img_bgr.copy()
    if "gt" in pts_xy:
        for (x,y) in pts_xy["gt"]:
            cv2.circle(out, (int(x),int(y)), 4, color_gt, -1)
    if "pred" in pts_xy:
        for (x,y) in pts_xy["pred"]:
            cv2.circle(out, (int(x),int(y)), 4, color_pred, -1)
    return out

def save_overlay(path, img_rgb, pts_xy):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    over = draw_points(bgr, pts_xy)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, over)
