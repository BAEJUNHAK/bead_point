# src/data/augmentation.py
"""
이미지 및 키포인트 데이터 증강 모듈
"""
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import threading
import time


class AugmentationStats:
    """데이터 증강 통계를 추적하는 클래스"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()
    
    def reset(self):
        """통계 초기화"""
        with self.lock:
            self.total_samples = 0
            self.augmented_samples = 0
            self.augmentation_types = {
                'rotation': 0,
                'scaling': 0,
                'translation': 0,
                'brightness': 0,
                'contrast': 0,
                'noise': 0,
                'gamma': 0,
                'color_shift': 0,
            }
            self.start_time = time.time()
    
    def log_sample(self, augmented: bool, aug_params: Dict[str, Any] = None):
        """샘플 처리 로그"""
        with self.lock:
            self.total_samples += 1
            if augmented:
                self.augmented_samples += 1
                if aug_params:
                    # 적용된 증강 유형 카운트
                    if aug_params.get('rotation', 0) != 0:
                        self.augmentation_types['rotation'] += 1
                    if aug_params.get('scale', 1.0) != 1.0:
                        self.augmentation_types['scaling'] += 1
                    if aug_params.get('translate_x', 0) != 0 or aug_params.get('translate_y', 0) != 0:
                        self.augmentation_types['translation'] += 1
                    if aug_params.get('brightness', 1.0) != 1.0:
                        self.augmentation_types['brightness'] += 1
                    if aug_params.get('contrast', 1.0) != 1.0:
                        self.augmentation_types['contrast'] += 1
                    if aug_params.get('add_noise', False):
                        self.augmentation_types['noise'] += 1
                    if aug_params.get('gamma_correction', False):
                        self.augmentation_types['gamma'] += 1
                    if aug_params.get('color_shift', False):
                        self.augmentation_types['color_shift'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """현재 통계 반환"""
        with self.lock:
            if self.total_samples == 0:
                return {
                    "total_samples": 0, 
                    "augmented_samples": 0, 
                    "original_samples": 0,
                    "augmentation_ratio": 0.0,
                    "data_multiplier": 1.0,
                    "augmentation_types": {},
                    "elapsed_time": 0.0,
                    "samples_per_second": 0.0
                }
            
            augmentation_ratio = self.augmented_samples / self.total_samples
            effective_data_multiplier = 1.0 + augmentation_ratio
            
            elapsed_time = time.time() - self.start_time
            
            return {
                "total_samples": self.total_samples,
                "augmented_samples": self.augmented_samples,
                "original_samples": self.total_samples - self.augmented_samples,
                "augmentation_ratio": augmentation_ratio,
                "data_multiplier": effective_data_multiplier,
                "augmentation_types": self.augmentation_types.copy(),
                "elapsed_time": elapsed_time,
                "samples_per_second": self.total_samples / elapsed_time if elapsed_time > 0 else 0
            }
    
    def print_summary(self, epoch: int = None):
        """통계 요약 출력"""
        stats = self.get_stats()
        
        if stats.get("total_samples", 0) == 0:
            return
        
        prefix = f"[Epoch {epoch}] " if epoch is not None else ""
        print(f"\n📊 {prefix}Data Augmentation Statistics:")
        print(f"  📈 Total samples processed: {stats.get('total_samples', 0):,}")
        print(f"  🔄 Augmented samples: {stats.get('augmented_samples', 0):,}")
        print(f"  📦 Original samples: {stats.get('original_samples', 0):,}")
        print(f"  📊 Augmentation ratio: {stats.get('augmentation_ratio', 0.0):.1%}")
        print(f"  🚀 Effective data multiplier: {stats.get('data_multiplier', 1.0):.2f}x")
        print(f"  ⏱️  Processing speed: {stats.get('samples_per_second', 0.0):.1f} samples/sec")
        
        print(f"  🎯 Augmentation breakdown:")
        aug_types = stats.get('augmentation_types', {})
        total_samples = stats.get('total_samples', 1)  # 0으로 나누기 방지
        for aug_type, count in aug_types.items():
            if count > 0:
                percentage = count / total_samples * 100
                print(f"    - {aug_type}: {count:,} ({percentage:.1f}%)")


# 전역 통계 객체
_global_aug_stats = AugmentationStats()


def sample_augmentation_params(training: bool = True, strength: float = 1.0) -> Dict[str, Any]:
    """
    증강 파라미터를 랜덤 샘플링
    
    Args:
        training: 학습 모드인지 여부 (False면 증강 안함)
        strength: 증강 강도 (0.0~1.0)
    
    Returns:
        증강 파라미터 딕셔너리
    """
    if not training:
        return {"apply_aug": False}
    
    # 더 자주 증강 적용 (더 강화된 버전)
    apply_prob = min(0.9, 0.5 + strength * 0.4)  # strength가 높을수록 더 자주 적용
    if np.random.random() > apply_prob:
        return {"apply_aug": False}
    
    params = {"apply_aug": True}
    
    # 회전: ±90도까지 (완전 빡세게!)
    max_rotation = min(90.0, 60.0 * strength)
    params['rotation'] = np.random.uniform(-max_rotation, max_rotation)
    
    # 가끔씩 90, 180, 270도 정확한 회전도 추가 (10% 확률)
    if np.random.random() < 0.1:
        params['rotation'] = np.random.choice([90, 180, 270, -90])
    
    # 스케일: 0.8~1.3 범위 (배경 최소화)
    scale_range = min(0.2, 0.15 * strength)
    params['scale'] = np.random.uniform(1.0 - scale_range, 1.0 + scale_range)
    
    # 평행이동: ±40픽셀까지 (배경 최소화)
    max_translate = min(40, 30 * strength)
    params['translate_x'] = np.random.uniform(-max_translate, max_translate)
    params['translate_y'] = np.random.uniform(-max_translate, max_translate)
    
    # 광학 증강 파라미터 (더 강하게)
    params['brightness'] = np.random.uniform(0.6, 1.4) if np.random.random() < 0.5 else 1.0
    params['contrast'] = np.random.uniform(0.6, 1.4) if np.random.random() < 0.5 else 1.0
    params['add_noise'] = np.random.random() < 0.3
    
    # Task에 특화된 추가 효과 (안전한 것들만)
    params['gamma_correction'] = np.random.random() < 0.3  # 30% 확률로 감마 보정
    params['color_shift'] = np.random.random() < 0.2  # 20% 확률로 색상 이동
    params['elastic'] = False  # 탄성 변형 일시 비활성화
    params['perspective'] = False  # 원근 변형 일시 비활성화
    
    return params


def create_transform_matrix(H: int, W: int, aug_params: Dict[str, Any]) -> np.ndarray:
    """
    어파인 변환 매트릭스 생성
    
    Args:
        H, W: 이미지 높이, 너비
        aug_params: 증강 파라미터
    
    Returns:
        2x3 어파인 변환 매트릭스
    """
    if not aug_params.get("apply_aug", False):
        # 항등 변환 반환
        return np.array([[1.0, 0.0, 0.0], 
                        [0.0, 1.0, 0.0]], dtype=np.float32)
    
    center_x, center_y = W / 2.0, H / 2.0
    
    # 회전 + 스케일링
    angle = aug_params.get('rotation', 0.0)
    scale = aug_params.get('scale', 1.0)
    
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
    
    # 평행이동 추가
    tx = aug_params.get('translate_x', 0.0)
    ty = aug_params.get('translate_y', 0.0)
    
    M[0, 2] += tx  # x 이동
    M[1, 2] += ty  # y 이동
    
    return M.astype(np.float32)


def transform_points(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    어파인 변환 매트릭스로 좌표들 변환
    
    Args:
        points: [N, 2] 좌표 배열 (x, y)
        M: [2, 3] 어파인 변환 매트릭스
    
    Returns:
        [N, 2] 변환된 좌표 배열
    """
    if len(points) == 0:
        return points.copy()
    
    # 동차좌표로 변환 [x,y] -> [x,y,1]
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homogeneous = np.hstack([points.astype(np.float32), ones])  # [N,3]
    
    # 변환 적용: [N,3] @ [3,2]^T -> [N,2]
    transformed = homogeneous @ M.T
    
    return transformed.astype(np.float32)


def clip_points(points: np.ndarray, H: int, W: int, margin: int = 2) -> np.ndarray:
    """
    변환된 좌표를 이미지 경계 내로 클리핑
    
    Args:
        points: [N, 2] 좌표 배열
        H, W: 이미지 높이, 너비
        margin: 경계로부터의 최소 거리
    
    Returns:
        [N, 2] 클리핑된 좌표 배열
    """
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], margin, W - margin - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], margin, H - margin - 1)
    return clipped


def apply_photometric_augmentation(image: np.ndarray, aug_params: Dict[str, Any]) -> np.ndarray:
    """
    색상 및 광학적 증강 적용 (좌표 변환 불필요)
    
    Args:
        image: [H, W, 3] RGB 이미지 (0~255)
        aug_params: 증강 파라미터
    
    Returns:
        [H, W, 3] 증강된 이미지
    """
    if not aug_params.get("apply_aug", False):
        return image.copy()
    
    aug_image = image.astype(np.float32)
    
    # 밝기 조정
    brightness = aug_params.get('brightness', 1.0)
    if brightness != 1.0:
        aug_image = aug_image * brightness
    
    # 대비 조정
    contrast = aug_params.get('contrast', 1.0)
    if contrast != 1.0:
        mean_val = aug_image.mean()
        aug_image = (aug_image - mean_val) * contrast + mean_val
    
    # 가우시안 노이즈 추가 (적당한 강도로 조정)
    if aug_params.get('add_noise', False):
        noise_strength = np.random.uniform(1.0, 4.0)  # 적당한 노이즈
        noise = np.random.normal(0, noise_strength, aug_image.shape)
        aug_image = aug_image + noise
    
    # 감마 보정 (조명 변화 시뮬레이션) - 안전하게 처리
    if aug_params.get('gamma_correction', False):
        gamma = np.random.uniform(0.5, 2.0)  # 감마 값
        # 음수 값 방지를 위해 클리핑
        normalized = np.clip(aug_image / 255.0, 0.0, 1.0)
        aug_image = np.power(normalized, gamma) * 255.0
    
    # 색상 이동 (RGB 채널별 변화)
    if aug_params.get('color_shift', False):
        # 각 채널에 독립적인 shift 적용
        shift_r = np.random.uniform(-20, 20)
        shift_g = np.random.uniform(-20, 20)
        shift_b = np.random.uniform(-20, 20)
        
        aug_image[:, :, 0] += shift_r  # R 채널
        aug_image[:, :, 1] += shift_g  # G 채널
        aug_image[:, :, 2] += shift_b  # B 채널
    
    # 값 범위 클리핑
    aug_image = np.clip(aug_image, 0, 255)
    
    return aug_image.astype(np.uint8)


def apply_geometric_augmentation(
    image: np.ndarray, 
    keypoints: np.ndarray, 
    polyline_points: np.ndarray, 
    aug_params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    기하학적 증강을 이미지와 모든 좌표에 동시 적용
    
    Args:
        image: [H, W, 3] RGB 이미지
        keypoints: [2, 2] 키포인트 좌표 (x, y)
        polyline_points: [N, 2] 폴리라인 좌표 (x, y)
        aug_params: 증강 파라미터
    
    Returns:
        (aug_image, aug_keypoints, aug_polyline)
    """
    H, W = image.shape[:2]
    
    # 변환 매트릭스 생성
    M = create_transform_matrix(H, W, aug_params)
    
    # 이미지 변환 (이미지 경계값으로 채우기)
    aug_image = cv2.warpAffine(
        image, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE  # 경계 픽셀을 복제하여 자연스럽게
    )
    
    # 키포인트 변환
    aug_keypoints = transform_points(keypoints, M)
    aug_keypoints = clip_points(aug_keypoints, H, W)
    
    # 폴리라인 변환
    aug_polyline = transform_points(polyline_points, M)
    aug_polyline = clip_points(aug_polyline, H, W)
    
    # 추가 고급 변형 적용
    if aug_params.get('perspective', False):
        aug_image, aug_keypoints, aug_polyline = apply_perspective_transform(
            aug_image, aug_keypoints, aug_polyline, H, W
        )
    
    if aug_params.get('elastic', False):
        aug_image, aug_keypoints, aug_polyline = apply_elastic_transform(
            aug_image, aug_keypoints, aug_polyline, H, W
        )
    
    return aug_image, aug_keypoints, aug_polyline


def apply_augmentation_pipeline(
    image: np.ndarray,
    keypoints: np.ndarray, 
    polyline_points: np.ndarray,
    training: bool = True,
    strength: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    완전한 데이터 증강 파이프라인
    
    Args:
        image: [H, W, 3] RGB 이미지
        keypoints: [2, 2] 키포인트 좌표 (x, y)
        polyline_points: [N, 2] 폴리라인 좌표 (x, y)
        training: 학습 모드인지 여부
        strength: 증강 강도 (0.0~1.0)
    
    Returns:
        (aug_image, aug_keypoints, aug_polyline)
    """
    # 증강 파라미터 샘플링
    aug_params = sample_augmentation_params(training, strength)
    
    # 기하학적 증강 적용
    aug_image, aug_keypoints, aug_polyline = apply_geometric_augmentation(
        image, keypoints, polyline_points, aug_params
    )
    
    # 광학적 증강 적용
    aug_image = apply_photometric_augmentation(aug_image, aug_params)
    
    # 통계 기록
    augmented = aug_params.get("apply_aug", False)
    _global_aug_stats.log_sample(augmented, aug_params if augmented else None)
    
    return aug_image, aug_keypoints, aug_polyline


def get_augmentation_stats() -> Dict[str, Any]:
    """현재 증강 통계 반환"""
    return _global_aug_stats.get_stats()


def print_augmentation_summary(epoch: int = None):
    """증강 통계 요약 출력"""
    _global_aug_stats.print_summary(epoch)


def reset_augmentation_stats():
    """증강 통계 초기화"""
    _global_aug_stats.reset()


def apply_perspective_transform(
    image: np.ndarray, 
    keypoints: np.ndarray, 
    polyline_points: np.ndarray, 
    H: int, W: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """원근 변형 적용"""
    # 원근 변형 강도
    max_offset = 30
    
    # 원본 모서리 점들
    src_points = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    
    # 변형된 모서리 점들 (랜덤 offset) - 안전하게 생성
    offsets = np.random.uniform(-max_offset, max_offset, (4, 2))
    dst_points = src_points + offsets
    
    # 경계 내에서 유지하되, 최소 간격 보장
    margin = 10
    dst_points = np.clip(dst_points, [margin, margin], [W-margin, H-margin])
    dst_points = dst_points.astype(np.float32)
    
    # 원근 변환 매트릭스
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 이미지 변환
    aug_image = cv2.warpPerspective(
        image, perspective_matrix, (W, H),
        borderMode=cv2.BORDER_REPLICATE  # 경계 픽셀 복제
    )
    
    # 좌표 변환 (homogeneous coordinates)
    def transform_points_perspective(points, matrix):
        if len(points) == 0:
            return points
        
        # 동차좌표로 변환
        ones = np.ones((len(points), 1))
        homogeneous = np.hstack([points, ones])
        
        # 원근 변환 적용
        transformed = homogeneous @ matrix.T
        
        # 정규화 (w로 나누기)
        transformed[:, 0] /= transformed[:, 2]
        transformed[:, 1] /= transformed[:, 2]
        
        return transformed[:, :2].astype(np.float32)
    
    aug_keypoints = transform_points_perspective(keypoints, perspective_matrix)
    aug_polyline = transform_points_perspective(polyline_points, perspective_matrix)
    
    # 경계 클리핑
    aug_keypoints = clip_points(aug_keypoints, H, W)
    aug_polyline = clip_points(aug_polyline, H, W)
    
    return aug_image, aug_keypoints, aug_polyline


def apply_elastic_transform(
    image: np.ndarray, 
    keypoints: np.ndarray, 
    polyline_points: np.ndarray, 
    H: int, W: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """탄성 변형 적용 (간단한 버전)"""
    # 탄성 변형 강도
    sigma = 10.0  # 변형 강도
    alpha = 100.0  # 변형 스케일
    
    # 랜덤 변위 필드 생성
    dx = np.random.uniform(-1, 1, (H//8, W//8)) * alpha
    dy = np.random.uniform(-1, 1, (H//8, W//8)) * alpha
    
    # 가우시안 필터로 부드럽게
    dx = cv2.GaussianBlur(dx, (0, 0), sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigma)
    
    # 원본 크기로 리사이즈
    dx = cv2.resize(dx, (W, H))
    dy = cv2.resize(dy, (W, H))
    
    # 격자 생성
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    
    # 이미지 변형
    aug_image = cv2.remap(
        image, map_x, map_y, 
        cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_REPLICATE  # 경계 픽셀 복제
    )
    
    # 좌표 변형 (근사적으로 보간)
    def transform_points_elastic(points, dx_map, dy_map):
        if len(points) == 0:
            return points
        
        transformed = points.copy()
        for i, (x, y) in enumerate(points):
            x_int, y_int = int(x), int(y)
            if 0 <= x_int < W and 0 <= y_int < H:
                transformed[i, 0] += dx_map[y_int, x_int]
                transformed[i, 1] += dy_map[y_int, x_int]
        
        return transformed.astype(np.float32)
    
    aug_keypoints = transform_points_elastic(keypoints, dx, dy)
    aug_polyline = transform_points_elastic(polyline_points, dx, dy)
    
    # 경계 클리핑
    aug_keypoints = clip_points(aug_keypoints, H, W)
    aug_polyline = clip_points(aug_polyline, H, W)
    
    return aug_image, aug_keypoints, aug_polyline


def create_polyline_mask_from_points(polyline_points: np.ndarray, H: int, W: int, thickness: int = 5) -> np.ndarray:
    """
    폴리라인 좌표로부터 마스크 생성 (증강된 좌표용)
    
    Args:
        polyline_points: [N, 2] 폴리라인 좌표 (x, y)
        H, W: 이미지 높이, 너비
        thickness: 라인 두께
    
    Returns:
        [H, W] 마스크 (0.0 또는 1.0)
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    
    if len(polyline_points) < 2:
        return mask.astype(np.float32)
    
    # 좌표를 정수로 변환
    points = polyline_points.astype(np.int32)
    
    # 폴리라인 그리기
    cv2.polylines(mask, [points], isClosed=False, color=1, thickness=thickness)
    
    return mask.astype(np.float32)
