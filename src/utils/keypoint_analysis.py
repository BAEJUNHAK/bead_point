# src/utils/keypoint_analysis.py
import torch
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader


def check_individual_performance(model, dataloader: DataLoader, device: str = "auto") -> Dict[str, float]:
    """
    P1과 P2의 개별 성능을 분석하여 불균형 학습 여부를 확인
    
    Args:
        model: 학습된 키포인트 모델
        dataloader: 검증용 데이터로더
        device: 추론 디바이스
        
    Returns:
        개별 성능 통계 딕셔너리
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)
    
    p1_errors = []
    p2_errors = []
    p1_coords = []
    p2_coords = []
    gt_p1_coords = []
    gt_p2_coords = []
    
    with torch.no_grad():
        for batch in dataloader:
            img = batch["image"].to(device)
            gt_coords = batch["coord_px"].to(device)  # [B,2,2]
            
            # 모델 예측
            logit = model(img)
            prob = torch.sigmoid(logit)
            
            # 좌표 추출 (argmax 방식)
            from ..losses.heatmap_utils import heatmaps_to_coords_argmax
            pred_coords = heatmaps_to_coords_argmax(prob)  # [B,2,2]
            
            # P1, P2 개별 오차 계산
            p1_error = torch.norm(pred_coords[:, 0] - gt_coords[:, 0], dim=-1)  # [B]
            p2_error = torch.norm(pred_coords[:, 1] - gt_coords[:, 1], dim=-1)  # [B]
            
            # 결과 수집
            p1_errors.extend(p1_error.cpu().numpy())
            p2_errors.extend(p2_error.cpu().numpy())
            p1_coords.extend(pred_coords[:, 0].cpu().numpy())
            p2_coords.extend(pred_coords[:, 1].cpu().numpy())
            gt_p1_coords.extend(gt_coords[:, 0].cpu().numpy())
            gt_p2_coords.extend(gt_coords[:, 1].cpu().numpy())
    
    # 통계 계산
    p1_mae = np.mean(p1_errors)
    p2_mae = np.mean(p2_errors)
    p1_std = np.std(p1_errors)
    p2_std = np.std(p2_errors)
    error_ratio = p1_mae / (p2_mae + 1e-8)
    
    # 거리 분석 (두 점 사이의 거리)
    pred_distances = [np.linalg.norm(p1 - p2) for p1, p2 in zip(p1_coords, p2_coords)]
    gt_distances = [np.linalg.norm(p1 - p2) for p1, p2 in zip(gt_p1_coords, gt_p2_coords)]
    distance_error = np.mean([abs(pred - gt) for pred, gt in zip(pred_distances, gt_distances)])
    
    results = {
        "p1_mae": p1_mae,
        "p2_mae": p2_mae,
        "p1_std": p1_std,
        "p2_std": p2_std,
        "error_ratio": error_ratio,
        "total_mae": (p1_mae + p2_mae) / 2,
        "distance_error": distance_error,
        "avg_pred_distance": np.mean(pred_distances),
        "avg_gt_distance": np.mean(gt_distances),
        "sample_count": len(p1_errors)
    }
    
    return results


def print_performance_report(results: Dict[str, float]) -> None:
    """성능 분석 결과를 보기 좋게 출력"""
    print("=" * 60)
    print("🔍 키포인트 개별 성능 분석 리포트")
    print("=" * 60)
    print(f"📊 샘플 수: {results['sample_count']}")
    print()
    
    print("📍 개별 포인트 성능:")
    print(f"  P1 MAE: {results['p1_mae']:.2f} ± {results['p1_std']:.2f} px")
    print(f"  P2 MAE: {results['p2_mae']:.2f} ± {results['p2_std']:.2f} px")
    print(f"  전체 MAE: {results['total_mae']:.2f} px")
    print()
    
    print("⚖️ 균형성 분석:")
    print(f"  P1/P2 오차 비율: {results['error_ratio']:.2f}")
    if results['error_ratio'] > 2.0:
        print("  ⚠️ 경고: P1 성능이 P2보다 크게 나쁩니다!")
    elif results['error_ratio'] < 0.5:
        print("  ⚠️ 경고: P2 성능이 P1보다 크게 나쁩니다!")
    else:
        print("  ✅ 양호: 두 포인트의 성능이 균형적입니다.")
    print()
    
    print("📏 거리 분석:")
    print(f"  예측 평균 거리: {results['avg_pred_distance']:.2f} px")
    print(f"  정답 평균 거리: {results['avg_gt_distance']:.2f} px")
    print(f"  거리 오차: {results['distance_error']:.2f} px")
    print()
    
    # 권장사항
    print("💡 권장사항:")
    if results['error_ratio'] > 1.5:
        print("  - P1 학습 강화: loss_p1에 더 높은 가중치 적용")
        print("  - P1 데이터 품질 확인")
    elif results['error_ratio'] < 0.67:
        print("  - P2 학습 강화: loss_p2에 더 높은 가중치 적용") 
        print("  - P2 데이터 품질 확인")
    else:
        print("  - 현재 균형이 양호합니다. 전체적인 성능 향상에 집중하세요.")
    
    if results['distance_error'] > 20:
        print("  - 거리 제약 Loss 추가를 고려해보세요.")
    
    print("=" * 60)


def analyze_training_logs(log_file_path: str) -> None:
    """
    학습 로그에서 loss_ratio 변화를 분석
    
    Args:
        log_file_path: Lightning CSV 로거 파일 경로
    """
    try:
        import pandas as pd
        df = pd.read_csv(log_file_path)
        
        if 'train_loss_ratio' in df.columns:
            ratios = df['train_loss_ratio'].dropna()
            
            print("📈 학습 중 Loss Ratio 변화:")
            print(f"  초기 ratio: {ratios.iloc[0]:.2f}")
            print(f"  최종 ratio: {ratios.iloc[-1]:.2f}")
            print(f"  평균 ratio: {ratios.mean():.2f}")
            print(f"  표준편차: {ratios.std():.2f}")
            
            if ratios.std() > 1.0:
                print("  ⚠️ 경고: ratio 변동이 큽니다. 불안정한 학습 가능성")
            
            # 간단한 시각화 (옵션)
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(ratios.values)
                plt.title('Training Loss Ratio (P1/P2) over Time')
                plt.xlabel('Step')
                plt.ylabel('Loss Ratio')
                plt.axhline(y=1.0, color='r', linestyle='--', label='Perfect Balance')
                plt.legend()
                plt.grid(True)
                plt.savefig('loss_ratio_trend.png')
                print(f"  📊 그래프 저장: loss_ratio_trend.png")
            except ImportError:
                print("  (matplotlib 설치시 그래프 생성 가능)")
                
    except Exception as e:
        print(f"로그 분석 실패: {e}")


if __name__ == "__main__":
    # 사용 예시
    print("🚀 키포인트 분석 도구 로드 완료!")
    print("사용법:")
    print("  from src.utils.keypoint_analysis import check_individual_performance, print_performance_report")
    print("  results = check_individual_performance(model, val_dataloader)")
    print("  print_performance_report(results)")
