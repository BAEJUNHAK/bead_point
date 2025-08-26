# src/losses/heatmap_utils.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def make_heatmaps_torch(coords_xy: torch.Tensor, H: int, W: int, sigma: float):
    """
    coords_xy: [B, 2, 2]  (x,y) in pixel (리사이즈 해상도 기준)
    return   : [B, 2, H, W], 값 0..1
    """
    B = coords_xy.shape[0]
    device = coords_xy.device
    
    # 모든 텐서를 명시적으로 float32로 생성
    Y = torch.arange(H, device=device, dtype=torch.float32).view(1,1,H,1)
    X = torch.arange(W, device=device, dtype=torch.float32).view(1,1,1,W)
    cx = coords_xy[..., 0].float().view(B,2,1,1)  # float32 강제 변환
    cy = coords_xy[..., 1].float().view(B,2,1,1)  # float32 강제 변환
    sigma = float(sigma)  # sigma도 float32로 변환
    
    dist2 = (X - cx)**2 + (Y - cy)**2
    hm = torch.exp(- dist2 / (2.0 * (sigma**2)))
    return hm.clamp_(0, 1)

@torch.no_grad()
def make_polyline_heatmaps_from_mask(polyline_mask: torch.Tensor, sigma: float):
    """
    폴리라인 마스크에서 가우시안 히트맵을 생성
    원본 폴리라인 형태를 그대로 보존하면서 부드러운 히트맵 생성
    
    Args:
        polyline_mask: [B, H, W] - 폴리라인 마스크 (0 또는 1)
        sigma: 가우시안 표준편차
    
    Returns:
        [B, H, W] - 폴리라인에 대한 가우시안 히트맵
    """
    import cv2
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    
    B, H, W = polyline_mask.shape
    device = polyline_mask.device
    sigma = float(sigma)  # sigma를 float32로 변환
    
    heatmaps = []
    
    for b in range(B):
        mask_b = polyline_mask[b].cpu().numpy()  # [H, W]
        
        # 마스크에서 거리 변환 사용
        # 폴리라인(1) 영역에서의 거리를 계산
        binary_mask = (mask_b > 0.5).astype(np.uint8)
        
        if binary_mask.sum() == 0:
            # 마스크가 비어있으면 빈 히트맵
            heatmap = np.zeros((H, W), dtype=np.float32)
        else:
            # 거리 변환: 각 픽셀에서 가장 가까운 폴리라인까지의 거리
            distance_map = distance_transform_edt(1 - binary_mask).astype(np.float32)
            
            # 가우시안 적용
            heatmap = np.exp(- distance_map**2 / (2.0 * (sigma**2))).astype(np.float32)
        
        # 텐서로 변환
        heatmap_tensor = torch.from_numpy(heatmap).to(device)
        heatmaps.append(heatmap_tensor)
    
    polyline_heatmap = torch.stack(heatmaps, dim=0)  # [B, H, W]
    return polyline_heatmap.clamp_(0, 1)

def point_to_line_segment_distance(point: torch.Tensor, line_start: torch.Tensor, line_end: torch.Tensor) -> torch.Tensor:
    """
    점에서 선분까지의 최단 거리 계산 (단일 점 버전)
    
    Args:
        point: [2] - 점 좌표 (x, y)
        line_start: [2] - 선분 시작점 (x, y)
        line_end: [2] - 선분 끝점 (x, y)
    
    Returns:
        scalar - 점에서 선분까지의 거리
    """
    # 선분 벡터
    line_vec = line_end - line_start  # [2]
    line_length_sq = torch.sum(line_vec**2)
    
    if line_length_sq < 1e-8:  # 점과 같은 경우 (선분 길이가 0)
        return torch.sqrt(torch.sum((point - line_start)**2))
    
    # 점에서 선분 시작점으로의 벡터
    point_to_start = point - line_start  # [2]
    
    # 선분 위의 가장 가까운 점까지의 매개변수 t 계산
    t = torch.sum(point_to_start * line_vec) / line_length_sq  # scalar
    t = torch.clamp(t, 0.0, 1.0)  # 선분 범위로 제한
    
    # 선분 위의 가장 가까운 점
    closest_point = line_start + t * line_vec  # [2]
    
    # 거리 계산
    distance = torch.sqrt(torch.sum((point - closest_point)**2))
    
    return distance

@torch.no_grad()
def make_combined_heatmaps_from_mask(coords_xy: torch.Tensor, polyline_mask: torch.Tensor, 
                                    H: int, W: int, sigma_points: float, sigma_polyline: float,
                                    polyline_weight: float = 0.3):
    """
    키포인트와 폴리라인 마스크를 결합한 히트맵 생성
    
    Args:
        coords_xy: [B, 2, 2] - 키포인트 좌표
        polyline_mask: [B, H, W] - 폴리라인 마스크
        H, W: 히트맵 해상도
        sigma_points: 키포인트용 가우시안 표준편차
        sigma_polyline: 폴리라인용 가우시안 표준편차
        polyline_weight: 폴리라인 히트맵의 가중치 (0~1)
    
    Returns:
        [B, 2, H, W] - 결합된 히트맵 (채널 0,1은 각각 키포인트용)
    """
    # 키포인트 히트맵 생성
    keypoint_heatmaps = make_heatmaps_torch(coords_xy, H, W, sigma_points)  # [B, 2, H, W]
    
    # 폴리라인 히트맵 생성 (마스크에서)
    polyline_heatmap = make_polyline_heatmaps_from_mask(polyline_mask, sigma_polyline)  # [B, H, W]
    
    # 데이터 타입 강제 변환 (float32 보장)
    keypoint_heatmaps = keypoint_heatmaps.float()
    polyline_heatmap = polyline_heatmap.float()
    
    # 폴리라인 히트맵을 두 채널에 복사하여 키포인트 히트맵과 결합
    polyline_expanded = polyline_heatmap.unsqueeze(1).expand(-1, 2, -1, -1)  # [B, 2, H, W]
    
    # 가중 합성: 키포인트 히트맵 + 폴리라인 히트맵 (float32로 계산)
    polyline_weight = float(polyline_weight)  # 가중치도 float32로 변환
    combined_heatmaps = keypoint_heatmaps + polyline_weight * polyline_expanded
    
    return combined_heatmaps.clamp_(0, 1)

@torch.no_grad()
def heatmaps_to_coords_argmax(hm: torch.Tensor):
    """
    hm: [B, 2, H, W]
    return: coords_xy [B, 2, 2]  (x,y)
    """
    B, C, H, W = hm.shape
    flat = hm.view(B, C, -1)
    idx = flat.argmax(dim=-1)           # [B,2]
    y = (idx // W).float()
    x = (idx %  W).float()
    return torch.stack([x, y], dim=-1)  # [B,2,2]

def softargmax_2d(prob: torch.Tensor) -> torch.Tensor:
    """
    prob: [B,2,H,W] sigmoid 출력 (0~1 범위, 가우시안 분포)
    반환: [B,2,2]  (각 점의 [x,y])
    """
    B, C, H, W = prob.shape
    device = prob.device
    
    # 연속적 그리드 생성 (0.0 ~ W-1, 0.0 ~ H-1)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xx = xs[None, None, None, :].expand(B, C, H, W)
    yy = ys[None, None, :, None].expand(B, C, H, W)
    
    # 각 채널별로 독립적으로 가중 평균 계산
    prob_sum = prob.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)  # [B,C,1,1]
    
    # 가중 평균으로 좌표 계산 (브로드캐스팅 활용)
    ex = (prob * xx).sum(dim=(2, 3)) / prob_sum.view(B, C)  # [B,C]
    ey = (prob * yy).sum(dim=(2, 3)) / prob_sum.view(B, C)  # [B,C]
    
    # [B,2,2] -> [[x1,y1],[x2,y2]]
    coords = torch.stack([ex, ey], dim=-1)
    return coords
