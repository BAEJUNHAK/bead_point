#!/usr/bin/env python3
"""
실제 데이터로 폴리라인 히트맵 기능 테스트
실제 데이터 로더와 실제 PKL 파일을 사용하여 테스트
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader

from src.data.dataset_img import BeadImageKeypointDataset
from src.losses.heatmap_utils import make_heatmaps_torch, make_polyline_heatmaps_from_mask, make_combined_heatmaps_from_mask

def test_real_polyline_heatmap(img_dir, pkl_dir, ann_csv=None, num_samples=3):
    """실제 데이터로 폴리라인 히트맵 테스트"""
    print("=== 실제 데이터로 폴리라인 히트맵 테스트 ===")
    
    # 실제 데이터셋 로드
    dataset = BeadImageKeypointDataset(
        img_dir=img_dir,
        pkl_dir=pkl_dir,
        size=512,
        annotation_csv=ann_csv,
        auto_blue_min=70,
        auto_diff_min=30
    )
    
    # 배치 크기를 1로 설정하여 크기 불일치 문제 해결
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    print(f"데이터셋 크기: {len(dataset)}")
    
    # 실제 배치 데이터 가져오기 (여러 샘플 테스트)
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_samples:  # 지정된 샘플 수만큼 테스트
            break
            
        print(f"\n=== 배치 {batch_idx} 분석 ===")
        
        # 배치 데이터 추출
        images = batch["image"]  # [B, 3, H, W]
        coords_xy = batch["coord_px"]  # [B, 2, 2] - 키포인트 좌표
        polyline_mask = batch["polyline_mask"]  # [B, H, W] - 폴리라인 마스크
        
        B, _, H, W = images.shape
        
        print(f"배치 크기: {B}")
        print(f"이미지 크기: {H}x{W}")
        print(f"키포인트 좌표 범위:")
        for b in range(B):
            print(f"  배치 {b}: 점1({coords_xy[b,0,0]:.1f},{coords_xy[b,0,1]:.1f}), 점2({coords_xy[b,1,0]:.1f},{coords_xy[b,1,1]:.1f})")
            mask_active_ratio = polyline_mask[b].mean().item()
            print(f"  폴리라인 마스크 활성 비율: {mask_active_ratio:.4f}")
        
        # 히트맵 생성 파라미터
        sigma_points = 5.0
        sigma_polyline = 3.0
        polyline_weight = 0.3
        
        # 각각의 히트맵 생성
        print("\n히트맵 생성 중...")
        keypoint_heatmaps = make_heatmaps_torch(coords_xy, H, W, sigma_points)
        polyline_heatmap = make_polyline_heatmaps_from_mask(polyline_mask, sigma_polyline)
        combined_heatmaps = make_combined_heatmaps_from_mask(
            coords_xy, polyline_mask, H, W, 
            sigma_points, sigma_polyline, polyline_weight
        )
        
        print(f"키포인트 히트맵 형태: {keypoint_heatmaps.shape}")
        print(f"폴리라인 히트맵 형태: {polyline_heatmap.shape}")
        print(f"결합 히트맵 형태: {combined_heatmaps.shape}")
        
        # 통계 정보
        print(f"\n=== 히트맵 통계 ===")
        print(f"키포인트 히트맵 - 최대값: {keypoint_heatmaps.max():.4f}, 평균: {keypoint_heatmaps.mean():.6f}")
        print(f"폴리라인 히트맵 - 최대값: {polyline_heatmap.max():.4f}, 평균: {polyline_heatmap.mean():.6f}")
        print(f"결합 히트맵 - 최대값: {combined_heatmaps.max():.4f}, 평균: {combined_heatmaps.mean():.6f}")
        print(f"폴리라인 마스크 - 활성 픽셀 비율: {polyline_mask.mean():.4f}")
        
        # 시각화
        visualize_real_heatmaps(
            images, coords_xy, polyline_mask,
            keypoint_heatmaps, polyline_heatmap, combined_heatmaps,
            batch_idx
        )
        
        # 다음 샘플로 계속
    
    print("\n✅ 실제 데이터 폴리라인 히트맵 테스트 완료!")

def visualize_real_heatmaps(images, coords_xy, polyline_mask,
                           keypoint_heatmaps, polyline_heatmap, combined_heatmaps, batch_idx):
    """실제 데이터 히트맵 시각화"""
    
    B = images.shape[0]
    
    for b in range(B):  # 배치 크기가 1이므로 모든 샘플 시각화
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'실제 데이터 폴리라인 히트맵 테스트 - 배치 {batch_idx}, 샘플 {b}', fontsize=16)
        
        # 원본 이미지
        img_np = images[b].permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('원본 이미지')
        
        # 키포인트 표시
        axes[0, 0].scatter([coords_xy[b, 0, 0]], [coords_xy[b, 0, 1]], 
                          c='red', s=100, marker='x', linewidth=3, label='키포인트 1')
        axes[0, 0].scatter([coords_xy[b, 1, 0]], [coords_xy[b, 1, 1]], 
                          c='blue', s=100, marker='x', linewidth=3, label='키포인트 2')
        
        # 폴리라인 마스크 윤곽선 표시
        mask_np = polyline_mask[b].numpy()
        contours, _ = cv2.findContours((mask_np > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and contour.shape[0] > 1:
                axes[0, 0].plot(contour[:, 0], contour[:, 1], 'g-', linewidth=2, alpha=0.7, label='폴리라인')
        axes[0, 0].legend()
        
        # 폴리라인 마스크
        axes[0, 1].imshow(polyline_mask[b].numpy(), cmap='gray')
        axes[0, 1].set_title('폴리라인 마스크')
        # 마스크 윤곽선 표시
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and contour.shape[0] > 1:
                axes[0, 1].plot(contour[:, 0], contour[:, 1], 'r-', linewidth=1, alpha=0.8)
        
        # 키포인트 히트맵 (채널 0, 1)
        axes[0, 2].imshow(keypoint_heatmaps[b, 0].numpy(), cmap='hot')
        axes[0, 2].set_title('키포인트 히트맵 - 채널 0')
        axes[0, 2].scatter([coords_xy[b, 0, 0]], [coords_xy[b, 0, 1]], 
                          c='cyan', s=50, marker='x')
        
        axes[0, 3].imshow(keypoint_heatmaps[b, 1].numpy(), cmap='hot')
        axes[0, 3].set_title('키포인트 히트맵 - 채널 1')
        axes[0, 3].scatter([coords_xy[b, 1, 0]], [coords_xy[b, 1, 1]], 
                          c='cyan', s=50, marker='x')
        
        # 폴리라인 히트맵
        axes[1, 0].imshow(polyline_heatmap[b].numpy(), cmap='hot')
        axes[1, 0].set_title('폴리라인 히트맵 (마스크 기반)')
        # 마스크 윤곽선 표시
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and contour.shape[0] > 1:
                axes[1, 0].plot(contour[:, 0], contour[:, 1], 'cyan', linewidth=1, alpha=0.8)
        
        # 폴리라인 히트맵 + 마스크 오버레이
        axes[1, 1].imshow(polyline_heatmap[b].numpy(), cmap='hot')
        axes[1, 1].contour(polyline_mask[b].numpy(), levels=[0.5], colors=['cyan'], linewidths=1)
        axes[1, 1].set_title('폴리라인 히트맵 + 마스크')
        
        # 결합 히트맵 (채널 0, 1)
        axes[1, 2].imshow(combined_heatmaps[b, 0].numpy(), cmap='hot')
        axes[1, 2].set_title('결합 히트맵 - 채널 0')
        axes[1, 2].scatter([coords_xy[b, 0, 0]], [coords_xy[b, 0, 1]], 
                          c='cyan', s=50, marker='x')
        # 마스크 윤곽선 표시
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and contour.shape[0] > 1:
                axes[1, 2].plot(contour[:, 0], contour[:, 1], 'cyan', linewidth=1, alpha=0.5)
        
        axes[1, 3].imshow(combined_heatmaps[b, 1].numpy(), cmap='hot')
        axes[1, 3].set_title('결합 히트맵 - 채널 1')
        axes[1, 3].scatter([coords_xy[b, 1, 0]], [coords_xy[b, 1, 1]], 
                          c='cyan', s=50, marker='x')
        # 마스크 윤곽선 표시
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and contour.shape[0] > 1:
                axes[1, 3].plot(contour[:, 0], contour[:, 1], 'cyan', linewidth=1, alpha=0.5)
        
        # 히트맵 비교 (가로 프로파일)
        mid_y = polyline_heatmap.shape[1] // 2
        axes[2, 0].plot(keypoint_heatmaps[b, 0, mid_y, :].numpy(), label='키포인트 채널 0')
        axes[2, 0].plot(keypoint_heatmaps[b, 1, mid_y, :].numpy(), label='키포인트 채널 1')
        axes[2, 0].plot(polyline_heatmap[b, mid_y, :].numpy(), label='폴리라인')
        axes[2, 0].set_title(f'히트맵 프로파일 (y={mid_y})')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 히트맵 비교 (세로 프로파일)
        mid_x = polyline_heatmap.shape[2] // 2
        axes[2, 1].plot(keypoint_heatmaps[b, 0, :, mid_x].numpy(), label='키포인트 채널 0')
        axes[2, 1].plot(keypoint_heatmaps[b, 1, :, mid_x].numpy(), label='키포인트 채널 1')
        axes[2, 1].plot(polyline_heatmap[b, :, mid_x].numpy(), label='폴리라인')
        axes[2, 1].set_title(f'히트맵 프로파일 (x={mid_x})')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # 폴리라인 마스크 분포
        axes[2, 2].imshow(polyline_mask[b].numpy(), cmap='gray', alpha=0.7)
        axes[2, 2].scatter([coords_xy[b, 0, 0]], [coords_xy[b, 0, 1]], 
                          c='red', s=100, marker='x', linewidth=3)
        axes[2, 2].scatter([coords_xy[b, 1, 0]], [coords_xy[b, 1, 1]], 
                          c='blue', s=100, marker='x', linewidth=3)
        # 마스크 윤곽선 표시
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and contour.shape[0] > 1:
                axes[2, 2].plot(contour[:, 0], contour[:, 1], 'yellow', linewidth=2, alpha=0.8)
        axes[2, 2].set_title('폴리라인 마스크 + 키포인트')
        axes[2, 2].set_xlim(0, polyline_heatmap.shape[2])
        axes[2, 2].set_ylim(polyline_heatmap.shape[1], 0)
        
        # 통계 정보 텍스트
        stats_text = f"""
        키포인트 1: ({coords_xy[b,0,0]:.1f}, {coords_xy[b,0,1]:.1f})
        키포인트 2: ({coords_xy[b,1,0]:.1f}, {coords_xy[b,1,1]:.1f})
        
        히트맵 최대값:
        - 키포인트: {keypoint_heatmaps[b].max():.4f}
        - 폴리라인: {polyline_heatmap[b].max():.4f}
        - 결합: {combined_heatmaps[b].max():.4f}
        
        활성 픽셀 비율:
        - 마스크: {polyline_mask[b].mean():.4f}
        - 폴리라인 히트맵 (>0.1): {(polyline_heatmap[b] > 0.1).float().mean():.4f}
        
        마스크 기반 접근:
        - 원본 폴리라인 형태 보존
        - 배치 처리 문제 해결
        - 정확한 키포인트-폴리라인 관계
        """
        axes[2, 3].text(0.05, 0.95, stats_text, transform=axes[2, 3].transAxes, 
                        verticalalignment='top', fontsize=10, fontfamily='monospace')
        axes[2, 3].set_xlim(0, 1)
        axes[2, 3].set_ylim(0, 1)
        axes[2, 3].axis('off')
        axes[2, 3].set_title('통계 정보')
        
        plt.tight_layout()
        plt.savefig(f'real_polyline_heatmap_test_batch{batch_idx}_sample{b}.png', 
                   dpi=150, bbox_inches='tight')
        print(f"시각화 결과 저장: real_polyline_heatmap_test_batch{batch_idx}_sample{b}.png")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True, help="이미지 디렉토리")
    parser.add_argument("--pkl_dir", required=True, help="PKL 디렉토리")
    parser.add_argument("--ann_csv", default=None, help="어노테이션 CSV 파일")
    parser.add_argument("--num_samples", type=int, default=3, help="테스트할 샘플 수")
    
    args = parser.parse_args()
    
    test_real_polyline_heatmap(
        img_dir=args.img_dir,
        pkl_dir=args.pkl_dir,
        ann_csv=args.ann_csv,
        num_samples=args.num_samples
    )
