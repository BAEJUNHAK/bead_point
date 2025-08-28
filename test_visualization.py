#!/usr/bin/env python3
"""
시각화 기능 테스트 스크립트
"""
import os
import torch
import numpy as np
import cv2
from pathlib import Path

# 프로젝트 모듈 임포트
from inference_complete import BeadPointInference
from src.utils.visualization import CNNVisualizationTool, HeatmapVisualizationTool
from src.utils.training_visualizer import VisualizationManager


def test_cnn_visualization():
    """CNN Activation 시각화 테스트"""
    print("🔍 CNN Activation 시각화 테스트 시작...")
    
    try:
        # 추론 객체 생성 (모델 로드)
        inferencer = BeadPointInference(device="auto")
        
        # CNN 시각화 도구 생성
        cnn_viz = CNNVisualizationTool(inferencer.model.net, device=inferencer.device)
        
        # 테스트 이미지 찾기
        test_dirs = [
            "data/Left/191412/img",
            "data/Right/191412/img", 
            "test_left"
        ]
        
        test_image = None
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                img_files = [f for f in os.listdir(test_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if img_files:
                    test_image = os.path.join(test_dir, img_files[0])
                    break
        
        if test_image is None:
            print("❌ 테스트 이미지를 찾을 수 없습니다.")
            return False
        
        print(f"📷 테스트 이미지: {test_image}")
        
        # 이미지 전처리
        img_tensor, _, _, _ = inferencer.preprocess_image(test_image)
        
        # Feature map 시각화
        save_dir = "runs/test_visualizations/cnn_activations"
        cnn_viz.visualize_feature_maps(
            input_tensor=img_tensor,
            max_channels=8,
            save_dir=save_dir,
            save_prefix="test_cnn"
        )
        
        # Activation 통계 시각화
        cnn_viz.visualize_activation_statistics(
            input_tensor=img_tensor,
            save_dir=save_dir,
            save_prefix="test_stats"
        )
        
        print(f"✅ CNN 시각화 완료! 결과: {save_dir}")
        return True
        
    except Exception as e:
        print(f"❌ CNN 시각화 테스트 실패: {e}")
        return False


def test_heatmap_visualization():
    """히트맵 시각화 테스트"""
    print("🔥 히트맵 시각화 테스트 시작...")
    
    try:
        # 추론 객체 생성
        inferencer = BeadPointInference(device="auto")
        
        # 테스트 이미지 찾기
        test_dirs = [
            "data/Left/191412/img",
            "data/Right/191412/img",
            "test_left"
        ]
        
        test_image = None
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                img_files = [f for f in os.listdir(test_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if img_files:
                    test_image = os.path.join(test_dir, img_files[0])
                    break
        
        if test_image is None:
            print("❌ 테스트 이미지를 찾을 수 없습니다.")
            return False
        
        print(f"📷 테스트 이미지: {test_image}")
        
        # 추론 실행
        result = inferencer.predict(test_image)
        
        # 원본 이미지 로드
        image_bgr = cv2.imread(test_image)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 히트맵 시각화
        save_path = "runs/test_visualizations/heatmap_test.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        HeatmapVisualizationTool.visualize_heatmap_prediction(
            image=image_rgb,
            pred_heatmaps=result['confidence_maps'],
            pred_coords=result['keypoints'],
            save_path=save_path,
            title="Test Heatmap Visualization"
        )
        
        print(f"✅ 히트맵 시각화 완료! 결과: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ 히트맵 시각화 테스트 실패: {e}")
        return False


def test_visualization_manager():
    """통합 시각화 매니저 테스트"""
    print("🎨 통합 시각화 매니저 테스트 시작...")
    
    try:
        # 추론 객체 생성
        inferencer = BeadPointInference(device="auto")
        
        # 시각화 매니저 생성
        viz_manager = VisualizationManager(save_dir="runs/test_visualizations/manager_test")
        
        # 테스트 이미지 찾기
        test_dirs = [
            "data/Left/191412/img",
            "data/Right/191412/img",
            "test_left"
        ]
        
        test_image = None
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                img_files = [f for f in os.listdir(test_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if img_files:
                    test_image = os.path.join(test_dir, img_files[0])
                    break
        
        if test_image is None:
            print("❌ 테스트 이미지를 찾을 수 없습니다.")
            return False
        
        print(f"📷 테스트 이미지: {test_image}")
        
        # 추론 실행
        result = inferencer.predict(test_image)
        
        # 통합 시각화 실행
        saved_files = viz_manager.visualize_inference_results(
            model=inferencer.model,
            image_path=test_image,
            result=result,
            save_prefix="manager_test"
        )
        
        print(f"✅ 통합 시각화 완료! 생성된 파일들:")
        for key, path in saved_files.items():
            print(f"  - {key}: {path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 통합 시각화 테스트 실패: {e}")
        return False


def test_inference_with_visualization():
    """추론 + 시각화 통합 테스트"""
    print("🚀 추론 + 시각화 통합 테스트 시작...")
    
    try:
        # 테스트 이미지가 있는 폴더 찾기
        test_dirs = [
            "data/Left/191412/img",
            "data/Right/191412/img",
            "test_left"
        ]
        
        test_dir = None
        for dir_path in test_dirs:
            if os.path.exists(dir_path):
                img_files = [f for f in os.listdir(dir_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if img_files:
                    test_dir = dir_path
                    break
        
        if test_dir is None:
            print("❌ 테스트 이미지 폴더를 찾을 수 없습니다.")
            return False
        
        print(f"📂 테스트 폴더: {test_dir}")
        
        # 추론 객체 생성
        inferencer = BeadPointInference(device="auto")
        
        # 시각화 포함 배치 처리 (최대 3개 이미지)
        results = inferencer.predict_folder(
            input_dir=test_dir,
            output_dir="runs/test_visualizations/inference_with_viz",
            save_overlay=True,
            save_csv=True,
            max_images=3,
            enable_visualization=True  # 고급 시각화 활성화
        )
        
        print(f"✅ 추론 + 시각화 완료! 처리된 이미지 수: {len(results)}")
        return True
        
    except Exception as e:
        print(f"❌ 추론 + 시각화 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("🧪 시각화 기능 테스트 시작")
    print("=" * 50)
    
    # 결과 저장 디렉토리 생성
    os.makedirs("runs/test_visualizations", exist_ok=True)
    
    test_results = {}
    
    # 1. CNN Activation 시각화 테스트
    test_results['cnn'] = test_cnn_visualization()
    print()
    
    # 2. 히트맵 시각화 테스트  
    test_results['heatmap'] = test_heatmap_visualization()
    print()
    
    # 3. 통합 시각화 매니저 테스트
    test_results['manager'] = test_visualization_manager()
    print()
    
    # 4. 추론 + 시각화 통합 테스트
    test_results['integrated'] = test_inference_with_visualization()
    print()
    
    # 결과 요약
    print("=" * 50)
    print("📊 테스트 결과 요약:")
    success_count = 0
    for test_name, success in test_results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"  {test_name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n🎯 전체 결과: {success_count}/{len(test_results)} 테스트 통과")
    
    if success_count == len(test_results):
        print("🎉 모든 시각화 기능이 정상 작동합니다!")
        print("📁 결과 확인: runs/test_visualizations/ 폴더")
    else:
        print("⚠️ 일부 테스트가 실패했습니다. 로그를 확인하세요.")


if __name__ == "__main__":
    main()
