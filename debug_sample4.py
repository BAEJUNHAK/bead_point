#!/usr/bin/env python3
"""
Sample 4의 이상한 증강 원인을 디버그하는 스크립트
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, '.')

from src.data.dataset_img import BeadImageKeypointDataset
from src.data.augmentation import sample_augmentation_params, apply_geometric_augmentation, apply_photometric_augmentation


def debug_sample4():
    """Sample 4와 동일한 조건으로 증강을 재현하고 분석"""
    
    # 테스트할 데이터 경로
    img_dir = "data/train/img"
    pkl_dir = "data/train/pkl" 
    ann_csv = "data/train/annotations.csv"
    
    if not os.path.exists(img_dir):
        print(f"❌ 이미지 디렉토리가 없습니다: {img_dir}")
        return
    
    print("🔍 Sample 4 디버깅 시작...")
    
    # 원본 데이터셋
    ds_orig = BeadImageKeypointDataset(
        img_dir=img_dir,
        pkl_dir=pkl_dir,
        annotation_csv=ann_csv,
        size=512,
        enable_augmentation=False,
        training=False,
    )
    
    # 원본 데이터 로드
    sample_orig = ds_orig[0]
    img_orig = sample_orig["image"].permute(1, 2, 0).numpy()
    kpts_orig = sample_orig["coord_px"].numpy()
    
    print(f"원본 이미지 형태: {img_orig.shape}")
    print(f"원본 키포인트: {kpts_orig}")
    print(f"이미지 값 범위: {img_orig.min():.3f} ~ {img_orig.max():.3f}")
    
    # Sample 4와 동일한 시드 설정
    np.random.seed(42 + 3 * 100)  # Sample 4의 시드
    
    # 증강 파라미터 샘플링
    aug_params = sample_augmentation_params(training=True, strength=2.0)
    
    print(f"\n📊 Sample 4 증강 파라미터:")
    for key, value in aug_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # 폴리라인 좌표 (실제로는 데이터셋에서 가져와야 하지만, 테스트용으로 생성)
    # 실제 구현에서는 poly_r 좌표를 사용
    H, W = img_orig.shape[:2]
    dummy_polyline = np.array([[100, 100], [200, 150], [300, 200], [400, 250]], dtype=np.float32)
    
    if aug_params.get("apply_aug", False):
        print(f"\n🔧 변환 매트릭스 생성 중...")
        
        # 단계별로 증강 적용
        try:
            # 1. 기하학적 증강만
            aug_image_geo, aug_kpts_geo, aug_poly_geo = apply_geometric_augmentation(
                img_orig, kpts_orig, dummy_polyline, aug_params
            )
            
            print(f"기하학적 증강 후:")
            print(f"  이미지 형태: {aug_image_geo.shape}")
            print(f"  이미지 값 범위: {aug_image_geo.min():.3f} ~ {aug_image_geo.max():.3f}")
            print(f"  키포인트: {aug_kpts_geo}")
            
            # 2. 광학적 증강 추가
            aug_image_final = apply_photometric_augmentation(aug_image_geo, aug_params)
            
            print(f"광학적 증강 후:")
            print(f"  이미지 값 범위: {aug_image_final.min():.3f} ~ {aug_image_final.max():.3f}")
            
            # 시각화
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 원본
            axes[0].imshow(img_orig)
            axes[0].scatter(kpts_orig[:, 0], kpts_orig[:, 1], c='red', s=100)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            # 기하학적 증강만
            axes[1].imshow(aug_image_geo)
            axes[1].scatter(aug_kpts_geo[:, 0], aug_kpts_geo[:, 1], c='red', s=100)
            axes[1].set_title('After Geometric Aug')
            axes[1].axis('off')
            
            # 최종 결과
            axes[2].imshow(aug_image_final)
            axes[2].scatter(aug_kpts_geo[:, 0], aug_kpts_geo[:, 1], c='red', s=100)
            axes[2].set_title('Final Result')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig('debug_sample4_result.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\n✅ 디버그 완료! 'debug_sample4_result.png' 확인")
            
        except Exception as e:
            print(f"❌ 증강 중 오류: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ 증강이 적용되지 않음")


def test_extreme_parameters():
    """극단적인 파라미터로 테스트"""
    
    print("\n🔬 극단적인 파라미터 테스트...")
    
    # 테스트 이미지 생성 (간단한 패턴)
    test_img = np.ones((512, 512, 3), dtype=np.uint8) * 128
    cv2.circle(test_img, (256, 256), 100, (255, 0, 0), -1)  # 빨간 원
    cv2.rectangle(test_img, (200, 200), (312, 312), (0, 255, 0), 3)  # 초록 사각형
    
    test_kpts = np.array([[200, 200], [312, 312]], dtype=np.float32)
    test_poly = np.array([[150, 150], [250, 150], [350, 250], [250, 350]], dtype=np.float32)
    
    # 극단적인 파라미터들 테스트
    extreme_params_list = [
        {"apply_aug": True, "rotation": 45, "scale": 1.5, "translate_x": 100, "translate_y": 100},
        {"apply_aug": True, "rotation": -30, "scale": 0.5, "translate_x": -80, "translate_y": 80},
        {"apply_aug": True, "rotation": 60, "scale": 2.0, "translate_x": 150, "translate_y": -150},
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    # 원본
    axes[0].imshow(test_img)
    axes[0].scatter(test_kpts[:, 0], test_kpts[:, 1], c='red', s=100)
    axes[0].set_title('Original Test Image')
    axes[0].axis('off')
    
    for i, params in enumerate(extreme_params_list, 1):
        try:
            aug_img, aug_kpts, aug_poly = apply_geometric_augmentation(
                test_img, test_kpts, test_poly, params
            )
            
            axes[i].imshow(aug_img)
            axes[i].scatter(aug_kpts[:, 0], aug_kpts[:, 1], c='red', s=100)
            axes[i].set_title(f'Extreme Test {i}\nRot:{params["rotation"]}° Scale:{params["scale"]}')
            axes[i].axis('off')
            
            print(f"  극단 테스트 {i}: 성공")
            
        except Exception as e:
            print(f"  극단 테스트 {i}: 실패 - {e}")
            axes[i].text(0.5, 0.5, f'Error:\n{str(e)[:30]}...', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'Extreme Test {i} - FAILED')
    
    plt.tight_layout()
    plt.savefig('debug_extreme_parameters.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 극단 파라미터 테스트 완료! 'debug_extreme_parameters.png' 확인")


if __name__ == "__main__":
    try:
        debug_sample4()
        test_extreme_parameters()
        
    except Exception as e:
        print(f"❌ 디버그 실패: {e}")
        import traceback
        traceback.print_exc()
