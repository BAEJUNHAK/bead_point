#!/usr/bin/env python3
"""
데이터 증강 기능 테스트 스크립트
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, '.')

from src.data.dataset_img import BeadImageKeypointDataset


def test_augmentation():
    """데이터 증강 기능을 테스트하고 시각화"""
    
    # 테스트할 데이터 경로 (실제 경로에 맞게 수정)
    img_dir = "data/train/img"
    pkl_dir = "data/train/pkl" 
    ann_csv = "data/train/annotations.csv"
    
    # 경로 존재 확인
    if not os.path.exists(img_dir):
        print(f"❌ 이미지 디렉토리가 없습니다: {img_dir}")
        print("실제 데이터 경로에 맞게 수정해주세요.")
        return
    
    print("🔍 Starting data augmentation test...")
    
    # 증강 없는 데이터셋
    ds_no_aug = BeadImageKeypointDataset(
        img_dir=img_dir,
        pkl_dir=pkl_dir,
        annotation_csv=ann_csv,
        size=512,
        enable_augmentation=False,
        training=False,
    )
    
    # 증강 있는 데이터셋 (적절한 강도로 증강 적용)
    ds_with_aug = BeadImageKeypointDataset(
        img_dir=img_dir,
        pkl_dir=pkl_dir,
        annotation_csv=ann_csv,
        size=512,
        enable_augmentation=True,
        augmentation_strength=1.5,  # 적절한 강도로 조정
        training=True,
    )
    
    print(f"✅ Dataset loaded successfully (samples: {len(ds_no_aug)})")
    
    # 첫 번째 샘플로 테스트
    if len(ds_no_aug) == 0:
        print("❌ Dataset is empty.")
        return
    
    # 원본 데이터
    print("📥 Loading original data...")
    sample_orig = ds_no_aug[0]
    img_orig = sample_orig["image"].permute(1, 2, 0).numpy()  # CHW -> HWC
    kpts_orig = sample_orig["coord_px"].numpy()
    mask_orig = sample_orig["polyline_mask"].numpy()
    
    # 증강된 데이터들 (여러 번 샘플링)
    print("🎲 Generating augmented data...")
    augmented_samples = []
    
    # 각 샘플링마다 다른 랜덤 시드를 사용하여 다른 증강 결과 얻기
    import torch
    augmentation_info = []
    
    for i in range(4):  # 4개의 증강된 샘플
        # 다른 증강을 위해 시드 변경
        torch.manual_seed(42 + i * 100)
        np.random.seed(42 + i * 100)
        
        sample_aug = ds_with_aug[0]  # 같은 원본 이미지로 다른 증강 적용
        img_aug = sample_aug["image"].permute(1, 2, 0).numpy()
        kpts_aug = sample_aug["coord_px"].numpy()
        mask_aug = sample_aug["polyline_mask"].numpy()
        augmented_samples.append((img_aug, kpts_aug, mask_aug))
        
        # 좌표 차이 계산으로 증강 확인
        coord_diff = np.linalg.norm(kpts_aug[0] - kpts_orig[0])
        augmentation_info.append(f"diff: {coord_diff:.1f}px")
        
        print(f"  Generated sample {i+1}: keypoint {kpts_aug[0]} (diff: {coord_diff:.1f}px from original)")
    
    # 시각화
    print("🎨 Creating visualization...")
    visualize_augmentation(img_orig, kpts_orig, mask_orig, augmented_samples, augmentation_info)
    
    # 증강 통계 출력
    from src.data.augmentation import print_augmentation_summary
    print_augmentation_summary()
    
    print("✅ Test completed! Check 'augmentation_test_result.png' file.")


def visualize_augmentation(img_orig, kpts_orig, mask_orig, augmented_samples, aug_info=None):
    """증강 결과를 시각화"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Data Augmentation Test Results', fontsize=16)
    
    # 원본 이미지
    ax = axes[0, 0]
    ax.imshow(img_orig)
    ax.scatter(kpts_orig[:, 0], kpts_orig[:, 1], c='red', s=100, marker='o')
    ax.set_title('Original Image')
    ax.axis('off')
    
    # 원본 마스크
    ax = axes[1, 0]
    ax.imshow(mask_orig, cmap='gray')
    ax.set_title('Original Polyline Mask')
    ax.axis('off')
    
    # 증강된 샘플들
    for i, (img_aug, kpts_aug, mask_aug) in enumerate(augmented_samples[:4]):
        if i < 2:
            row, col = 0, i + 1
        else:
            row, col = 1, i - 1
        
        # 증강된 이미지
        ax = axes[row, col]
        ax.imshow(img_aug)
        ax.scatter(kpts_aug[:, 0], kpts_aug[:, 1], c='red', s=100, marker='o')
        
        # 제목에 차이 정보 추가
        title = f'Augmented Sample {i+1}'
        if aug_info and i < len(aug_info):
            title += f'\n({aug_info[i]})'
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_test_result.png', dpi=150, bbox_inches='tight')
    plt.close()


def check_coordinate_transformation():
    """좌표 변환이 올바른지 확인"""
    from src.data.augmentation import (
        sample_augmentation_params, 
        create_transform_matrix, 
        transform_points
    )
    
    print("🔧 Testing coordinate transformation accuracy...")
    
    # 테스트 좌표
    test_points = np.array([[100, 100], [200, 200]], dtype=np.float32)
    H, W = 512, 512
    
    # 여러 변환 테스트
    for i in range(3):
        # 증강 파라미터 샘플링
        params = sample_augmentation_params(training=True, strength=1.0)
        
        if not params.get("apply_aug", False):
            print(f"  Test {i+1}: augmentation skipped")
            continue
        
        # 변환 매트릭스 생성
        M = create_transform_matrix(H, W, params)
        
        # 좌표 변환
        transformed = transform_points(test_points, M)
        
        print(f"  Test {i+1}:")
        print(f"    Rotation: {params.get('rotation', 0):.1f}°")
        print(f"    Scale: {params.get('scale', 1.0):.2f}")
        print(f"    Translation: ({params.get('translate_x', 0):.1f}, {params.get('translate_y', 0):.1f})")
        print(f"    Original: {test_points[0]} -> Transformed: {transformed[0]}")
    
    print("✅ Coordinate transformation test completed")


if __name__ == "__main__":
    try:
        # 좌표 변환 테스트
        check_coordinate_transformation()
        print()
        
        # 전체 증강 테스트
        test_augmentation()
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
