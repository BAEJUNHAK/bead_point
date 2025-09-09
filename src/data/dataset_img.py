# src/data/dataset_img.py
import os, glob, csv, pickle, re
from typing import Tuple, Dict, Any, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torch.nn.utils.rnn import pad_sequence

# 데이터 증강 모듈 import
from .augmentation import apply_augmentation_pipeline, create_polyline_mask_from_points


# ------------------ Blue (bead) ROI detection ------------------
def detect_blue_roi(img_bgr: np.ndarray,
                    blue_min: int = 80,
                    diff_min: int = 40,
                    min_area: int = 3000,
                    pad: int = 8) -> Tuple[int,int,int,int]:
    B, G, R = cv2.split(img_bgr)
    m1 = (B > blue_min).astype(np.uint8)
    m2 = (B - np.maximum(R, G) > diff_min).astype(np.uint8)
    mask = (m1 & m2) * 255
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, 1)
    mask = cv2.dilate(mask, ker, 1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img_bgr.shape[:2]
    if not cnts: return (pad, pad, W-pad, H-pad)
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area: return (pad, pad, W-pad, H-pad)
    x,y,w,h = cv2.boundingRect(cnt)
    return (max(0,x-pad), max(0,y-pad), min(W-1,x+w+pad), min(H-1,y+h+pad))

def blue_mask_roi(img_bgr: np.ndarray, roi: Tuple[int,int,int,int],
                  blue_min: int, diff_min: int) -> np.ndarray:
    B, G, R = cv2.split(img_bgr)
    m1 = (B > blue_min).astype(np.uint8)
    m2 = (B - np.maximum(R, G) > diff_min).astype(np.uint8)
    mask_full = (m1 & m2) * 255
    x0,y0,x1,y1 = roi
    return mask_full[y0:y1, x0:x1]


# ------------- Data→pixel mapping (with explicit range/flip) -------------
def map_with_range(pts_data: np.ndarray,
                   roi: Tuple[int,int,int,int],
                   data_range: Tuple[float,float,float,float],
                   flip_x: bool, flip_y: bool) -> np.ndarray:
    x0,y0,x1,y1 = roi
    w, h = max(1, x1-x0), max(1, y1-y0)
    x_min,x_max,y_min,y_max = data_range
    fx = (pts_data[:,0] - x_min) / (x_max - x_min + 1e-12)
    fy = (pts_data[:,1] - y_min) / (y_max - y_min + 1e-12)
    fx = np.clip(fx, 0, 1); fy = np.clip(fy, 0, 1)
    if flip_x: fx = 1.0 - fx
    if flip_y: fy = 1.0 - fy
    xs = x0 + fx * w
    ys = y0 + fy * h
    return np.stack([xs, ys], axis=1)

def best_flip_from_polyline(poly_pts: np.ndarray,
                            roi: Tuple[int,int,int,int],
                            mask_roi: np.ndarray) -> Tuple[bool,bool,Tuple[float,float,float,float]]:
    # 데이터 범위는 폴리라인(해당 샘플)의 min/max를 사용
    x_min, x_max = float(poly_pts[:,0].min()), float(poly_pts[:,0].max())
    y_min, y_max = float(poly_pts[:,1].min()), float(poly_pts[:,1].max())
    if abs(x_max-x_min) < 1e-9: x_max = x_min + 1.0
    if abs(y_max-y_min) < 1e-9: y_max = y_min + 1.0
    dr = (x_min,x_max,y_min,y_max)

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dil = cv2.dilate(mask_roi, ker, 1)

    def score(fx, fy):
        # 라인을 그려서 마스크 겹침 점수 계산
        pts = map_with_range(poly_pts, roi, dr, fx, fy)
        H, W = dil.shape[:2]
        canvas = np.zeros((H, W), np.uint8)
        for i in range(len(pts)-1):
            p0 = tuple(np.clip(np.round(pts[i   ] - [roi[0],roi[1]]).astype(int), 0, [W-1,H-1]))
            p1 = tuple(np.clip(np.round(pts[i+1] - [roi[0],roi[1]]).astype(int), 0, [W-1,H-1]))
            cv2.line(canvas, p0, p1, 255, 2)
        ys, xs = np.where(canvas>0)
        if len(xs)==0: return -1.0
        return float((dil[ys, xs] > 0).sum()) / len(xs)

    candidates = []
    for fx in (False, True):
        for fy in (True, False):  # 보통 y축은 뒤집힘
            candidates.append((score(fx, fy), fx, fy))
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, fx_best, fy_best = candidates[0]
    return fx_best, fy_best, dr


# ------------------- Polyline → two points (strategies) -------------------
def _ends_first_last(pts: np.ndarray) -> Tuple[float,float,float,float]:
    p1, p2 = pts[0], pts[-1]
    if p1[0] <= p2[0]: return float(p1[0]),float(p1[1]),float(p2[0]),float(p2[1])
    else:              return float(p2[0]),float(p2[1]),float(p1[0]),float(p1[1])

def _x_minmax(pts: np.ndarray) -> Tuple[float,float,float,float]:
    i0, i1 = int(pts[:,0].argmin()), int(pts[:,0].argmax())
    p1, p2 = pts[i0], pts[i1]
    if p1[0] <= p2[0]: return float(p1[0]),float(p1[1]),float(p2[0]),float(p2[1])
    else:              return float(p2[0]),float(p2[1]),float(p1[0]),float(p1[1])

def _extrema_far(pts: np.ndarray) -> Tuple[float,float,float,float]:
    xs, ys = pts[:,0], pts[:,1]
    cands = [int(xs.argmin()), int(xs.argmax()), int(ys.argmin()), int(ys.argmax())]
    cands = list(dict.fromkeys(cands))
    best, pair = -1.0, (cands[0], cands[-1])
    for i in range(len(cands)):
        for j in range(i+1, len(cands)):
            d = float(np.linalg.norm(pts[cands[i]] - pts[cands[j]]))
            if d > best: best, pair = d, (cands[i], cands[j])
    p1, p2 = pts[pair[0]], pts[pair[1]]
    if p1[0] <= p2[0]: return float(p1[0]),float(p1[1]),float(p2[0]),float(p2[1])
    else:              return float(p2[0]),float(p2[1]),float(p1[0]),float(p1[1])

def _arc_percent(pts: np.ndarray, low:float=0.10, high:float=0.90) -> Tuple[float,float,float,float]:
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    if seg.size==0: return _ends_first_last(pts)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1] if cum[-1]>0 else 1.0
    t1, t2 = low*total, high*total
    i1 = int(np.searchsorted(cum, t1, side="right") - 1)
    i2 = int(np.searchsorted(cum, t2, side="right") - 1)
    i1 = np.clip(i1, 0, len(pts)-1)
    i2 = np.clip(i2, 0, len(pts)-1)
    p1, p2 = pts[i1], pts[i2]
    if p1[0] <= p2[0]: return float(p1[0]),float(p1[1]),float(p2[0]),float(p2[1])
    else:              return float(p2[0]),float(p2[1]),float(p1[0]),float(p1[1])

STRATEGIES = {"ends":_ends_first_last, "x_minmax":_x_minmax, "extrema_far":_extrema_far, "arc_percent":_arc_percent}

def choose_two_points_from_polyline(pts_xy: np.ndarray,
                                    strategy:str="extrema_far",
                                    low:float=0.10, high:float=0.90) -> Tuple[float,float,float,float]:
    if pts_xy.shape[0]==2:
        i0,i1 = (0,1) if pts_xy[0,0] <= pts_xy[1,0] else (1,0)
        p0,p1 = pts_xy[i0], pts_xy[i1]
        return float(p0[0]),float(p0[1]),float(p1[0]),float(p1[1])
    if strategy=="arc_percent": return _arc_percent(pts_xy, low, high)
    return STRATEGIES.get(strategy, _extrema_far)(pts_xy)


# ------------------------ PKL & CSV loaders ------------------------
def load_points_from_pkl_raw(p: str) -> np.ndarray:
    with open(p, "rb") as f: obj = pickle.load(f)
    if isinstance(obj, dict):
        if "points" in obj:
            arr = np.array(obj["points"], np.float32)
            if arr.ndim==1: arr = arr.reshape(-1,2)
            return arr
        if "p1" in obj and "p2" in obj:
            return np.array([obj["p1"], obj["p2"]], np.float32)
        raise ValueError(f"{p}: dict but no 'points' or ('p1','p2').")
    if isinstance(obj, list):
        seq = obj[0] if (len(obj)>0 and isinstance(obj[0], list)) else obj
        return np.array([[float(a),float(b)] for (a,b) in seq], np.float32)
    raise ValueError(f"{p}: unsupported PKL structure.")

_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
def parse_point_str(s: str) -> Tuple[float,float]:
    if not isinstance(s, str): raise ValueError(f"point is not string: {s}")
    nums = _num_re.findall(s)
    if len(nums) < 2: raise ValueError(f"cannot parse point: {s}")
    return float(nums[0]), float(nums[1])

def load_annotation_csv_pointpair(csv_path: str) -> Dict[str, np.ndarray]:
    """
    CSV 형식:
      image,point1,point2
      left_191013_012.png,"(13.38, 201.41)","(14.75, 202.39)"
    반환: { "left_191013_012": np.array([[x1,y1],[x2,y2]], float32), ... }  # 데이터좌표
    """
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        cols = [c.strip().lower() for c in rdr.fieldnames]
        req = ["image","point1","point2"]
        for r in req:
            if r not in cols:
                raise ValueError(f"{csv_path}: missing column '{r}', columns={rdr.fieldnames}")
        # 실제 키 이름 원형 찾기
        col_index = {c.lower(): c for c in rdr.fieldnames}
        for row in rdr:
            img = row[col_index["image"]].strip()
            base = os.path.splitext(os.path.basename(img))[0]
            p1 = parse_point_str(row[col_index["point1"]])
            p2 = parse_point_str(row[col_index["point2"]])
            mapping[base] = np.array([p1, p2], dtype=np.float32)
    return mapping


# ----------------------------- Dataset -----------------------------
class BeadImageKeypointDataset(Dataset):
    """
    GT 우선순위:
      1) annotation.csv 의 point1/point2 (데이터좌표) → (PKL로 추정한 데이터축 범위 & flip)으로 픽셀 변환
      2) 없을 경우, PKL 폴리라인을 규칙으로 2점 축약하여 사용

    반환:
      image   : [3,H,W] float32 (0..1)
      coord_px: [2,2]   float32 픽셀 좌표 (리사이즈 해상도 기준)
      meta    : dict
    """
    def __init__(self,
                 img_dir: str,
                 pkl_dir: str,
                 size: int = 512,
                 strategy: str = "extrema_far",
                 arc_low: float = 0.10,
                 arc_high: float = 0.90,
                 annotation_csv: Optional[str] = None,
                 auto_blue_min: int = 80,
                 auto_diff_min: int = 40,
                 # 새로운 증강 관련 파라미터
                 enable_augmentation: bool = True,
                 augmentation_strength: float = 1.0,
                 training: bool = True):
        self.size = int(size)
        self.strategy = strategy
        self.arc_low = float(arc_low)
        self.arc_high = float(arc_high)
        self.auto_blue_min = int(auto_blue_min)
        self.auto_diff_min = int(auto_diff_min)
        self.ann_map = load_annotation_csv_pointpair(annotation_csv) if annotation_csv else None
        
        # 증강 관련 설정 저장
        self.enable_augmentation = enable_augmentation
        self.augmentation_strength = augmentation_strength
        self.training = training

        self.imgs: List[str] = sorted(
            glob.glob(os.path.join(img_dir, "*.png")) +
            glob.glob(os.path.join(img_dir, "*.jpg")) +
            glob.glob(os.path.join(img_dir, "*.jpeg"))
        )
        if len(self.imgs)==0:
            raise FileNotFoundError(f"No images in {img_dir}")
        self.pkl_dir = pkl_dir

    def __len__(self): return len(self.imgs)

    def _match_pkl(self, img_path: str) -> str:
        base = os.path.splitext(os.path.basename(img_path))[0]
        p = os.path.join(self.pkl_dir, base + ".pkl")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing PKL for {img_path} -> {p}")
        return p
    
    def _create_polyline_mask(self, polyline_points: np.ndarray, H: int, W: int, thickness: int = 5) -> np.ndarray:
        """폴리라인을 따라 마스크 생성 - 키포인트 제약조건"""
        mask = np.zeros((H, W), dtype=np.uint8)
        
        # 폴리라인을 이미지에 그리기
        pts = polyline_points.astype(np.int32)
        pts = np.clip(pts, 0, [W-1, H-1])  # 이미지 경계 내로 제한
        
        for i in range(len(pts)-1):
            cv2.line(mask, tuple(pts[i]), tuple(pts[i+1]), 255, thickness)
        
        # 각 폴리라인 점 주변도 마킹 (점 끝부분 보강)
        for pt in pts:
            cv2.circle(mask, tuple(pt), thickness//2, 255, -1)
        
        return (mask > 0).astype(np.float32)
    
    def _normalize_polyline_points(self, polyline_points: np.ndarray, target_points: int = 128) -> np.ndarray:
        """
        폴리라인 포인트를 고정 개수로 정규화 (리샘플링)
        
        Args:
            polyline_points: [N, 2] - 원본 폴리라인 포인트들
            target_points: 목표 포인트 개수
            
        Returns:
            [target_points, 2] - 정규화된 폴리라인 포인트들
        """
        if len(polyline_points) <= 1:
            # 포인트가 1개 이하면 복제하여 채움
            return np.tile(polyline_points[0] if len(polyline_points) > 0 else [0, 0], 
                          (target_points, 1)).astype(np.float32)
        
        # 누적 거리 계산 (폴리라인을 따라)
        distances = np.zeros(len(polyline_points))
        for i in range(1, len(polyline_points)):
            distances[i] = distances[i-1] + np.linalg.norm(polyline_points[i] - polyline_points[i-1])
        
        # 전체 길이로 정규화 (0~1)
        if distances[-1] > 0:
            normalized_distances = distances / distances[-1]
        else:
            normalized_distances = np.linspace(0, 1, len(polyline_points))
        
        # 목표 포인트 개수에 맞게 균등 분할
        target_distances = np.linspace(0, 1, target_points)
        
        # 보간을 통해 새로운 포인트들 생성
        new_x = np.interp(target_distances, normalized_distances, polyline_points[:, 0])
        new_y = np.interp(target_distances, normalized_distances, polyline_points[:, 1])
        
        return np.column_stack([new_x, new_y]).astype(np.float32)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        img_path = self.imgs[i]
        base = os.path.splitext(os.path.basename(img_path))[0]
        pkl_path = self._match_pkl(img_path)

        # 1) 이미지/ROI/마스크
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None: raise FileNotFoundError(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = img_rgb.shape[:2]

        roi = detect_blue_roi(img_bgr, self.auto_blue_min, self.auto_diff_min)
        mask_roi = blue_mask_roi(img_bgr, roi, self.auto_blue_min, self.auto_diff_min)

        # 2) 해당 샘플의 PKL 폴리라인 → flip과 데이터범위 결정
        poly = load_points_from_pkl_raw(pkl_path)  # 데이터좌표
        flip_x, flip_y, data_range = best_flip_from_polyline(poly, roi, mask_roi)
        
        # 폴리라인을 픽셀 좌표로 변환 (마스크 생성용)
        poly_px = map_with_range(poly, roi, data_range, flip_x, flip_y)

        # 3) GT 좌표 결정
        if self.ann_map is not None and base in self.ann_map:
            # CSV 두 점(데이터좌표)을, (PKL에서 얻은 data_range & flip)으로 픽셀 변환
            ann_pts_data = self.ann_map[base]
            pts_px = map_with_range(ann_pts_data, roi, data_range, flip_x, flip_y)
            pts2_px = pts_px.astype(np.float32)
            # 포인트 순서 일관성을 위해 정렬하지 않음 - CSV 원본 순서 유지
        else:
            # CSV 없으면 폴리라인 자체를 픽셀로 맵핑 → 2점 축약
            x1,y1,x2,y2 = choose_two_points_from_polyline(poly_px, self.strategy, self.arc_low, self.arc_high)
            pts2_px = np.array([[x1,y1],[x2,y2]], dtype=np.float32)

        # 4) 리사이즈 & 스케일
        img_r = cv2.resize(img_rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        sx, sy = self.size / float(W0), self.size / float(H0)
        pts2_r = pts2_px.copy()
        pts2_r[:,0] *= sx; pts2_r[:,1] *= sy
        # 포인트 순서 일관성을 위해 정렬하지 않음 - 원본 순서 유지
        
        poly_r = poly_px.copy()
        poly_r[:,0] *= sx; poly_r[:,1] *= sy
        
        # 4.5) 데이터 증강 적용 (training 시에만)
        if self.enable_augmentation and self.training:
            try:
                img_r, pts2_r, poly_r = apply_augmentation_pipeline(
                    img_r, pts2_r, poly_r,
                    training=True,
                    strength=self.augmentation_strength
                )
            except Exception as e:
                # 증강 실패 시 원본 데이터 사용 (안전장치)
                print(f"⚠️ 증강 실패, 원본 사용: {e}")
                pass
        else:
            # 증강 비활성화 시에도 통계 기록
            apply_augmentation_pipeline(
                img_r, pts2_r, poly_r,
                training=False,  # 증강 비활성화
                strength=0.0
            )
        
        # 5) 폴리라인 마스크 생성 (제약조건 적용) - 증강된 좌표 사용
        polyline_mask = create_polyline_mask_from_points(poly_r, self.size, self.size)

        return {
            "image": to_tensor(img_r),
            "coord_px": torch.from_numpy(pts2_r.astype(np.float32)),
            "polyline_mask": torch.from_numpy(polyline_mask.astype(np.float32)),
            # 폴리라인 포인트는 제거 - 마스크만 사용
            "meta": {
                "img_path": img_path, "pkl_path": pkl_path,
                "orig_size": (H0,W0), "resize": (self.size,self.size),
                "scale_xy": (sx,sy), "roi_xyxy": roi,
                "flip": (flip_x, flip_y), "data_range": data_range
            }
        }


# 이제 모든 폴리라인이 동일한 크기(128개 포인트)로 정규화되므로 
# 기본 DataLoader collate 함수를 사용할 수 있음

# ---------- legacy compatibility ----------
def load_points_from_pkl(p: str) -> Tuple[float,float,float,float]:
    pts = load_points_from_pkl_raw(p)
    x1,y1,x2,y2 = choose_two_points_from_polyline(pts, strategy="extrema_far")
    return x1,y1,x2,y2
