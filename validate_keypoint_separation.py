#!/usr/bin/env python3
"""
키포인트 분리 학습 검증 스크립트
개선된 Loss 함수의 효과를 확인합니다.
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.data.dataset_img import BeadImageKeypointDataset
from src.models.lit_model import LitKeypoint2
from src.utils.keypoint_analysis import check_individual_performance, print_performance_report


def main():
    parser = argparse.ArgumentParser(description="키포인트 분리 학습 검증")
    parser.add_argument("--checkpoint", required=True, help="검증할 체크포인트 파일")
    parser.add_argument("--data_img", default="data/train/img", help="이미지 디렉토리")
    parser.add_argument("--data_pkl", default="data/train/pkl", help="PKL 디렉토리")
    parser.add_argument("--annotations", default="data/train/annotations.csv", help="정답 CSV")
    parser.add_argument("--batch_size", type=int, default=8, help="배치 크기")
    parser.add_argument("--max_samples", type=int, default=50, help="검증할 최대 샘플 수")
    parser.add_argument("--device", default="auto", help="디바이스")
    
    args = parser.parse_args()
    
    print("🔍 키포인트 분리 학습 검증 시작")
    print(f"📄 체크포인트: {args.checkpoint}")
    
    # 디바이스 설정
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"🔧 디바이스: {device}")
    
    # 모델 로드
    try:
        model = LitKeypoint2.load_from_checkpoint(args.checkpoint, map_location=device)
        model.eval()
        print("✅ 모델 로드 성공")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 데이터셋 준비
    try:
        dataset = BeadImageKeypointDataset(
            img_dir=args.data_img,
            pkl_dir=args.data_pkl,
            size=256,  # 모델 입력 크기에 맞춤
            annotation_csv=args.annotations,
            auto_blue_min=70,
            auto_diff_min=30,
        )
        
        # 샘플 수 제한
        if len(dataset) > args.max_samples:
            indices = torch.randperm(len(dataset))[:args.max_samples]
            dataset = torch.utils.data.Subset(dataset, indices)
            print(f"📊 샘플 수 제한: {args.max_samples}개")
        
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=2
        )
        print(f"✅ 데이터셋 로드 성공: {len(dataset)}개 샘플")
        
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return
    
    # 개별 성능 분석
    print("\n🚀 개별 성능 분석 시작...")
    try:
        results = check_individual_performance(model, dataloader, device)
        print_performance_report(results)
        
        # 추가 분석
        print("\n🔬 추가 분석:")
        
        # 불균형 정도 판정
        ratio = results['error_ratio']
        if ratio > 2.0:
            severity = "심각"
            color = "🔴"
        elif ratio > 1.5 or ratio < 0.67:
            severity = "주의"
            color = "🟡"
        else:
            severity = "양호"
            color = "🟢"
        
        print(f"  {color} 균형성 평가: {severity} (ratio: {ratio:.2f})")
        
        # 성능 등급
        total_mae = results['total_mae']
        if total_mae < 5:
            grade = "A (우수)"
        elif total_mae < 10:
            grade = "B (양호)"
        elif total_mae < 20:
            grade = "C (보통)"
        else:
            grade = "D (개선필요)"
        
        print(f"  📊 전체 성능: {grade} (MAE: {total_mae:.2f}px)")
        
        # 개선 제안
        print("\n💡 구체적 개선 제안:")
        if ratio > 1.5:
            print("  1. training_step에서 loss_p1에 1.5배 가중치 적용")
            print("     loss = 1.5 * loss_p1 + loss_p2")
            print("  2. P1 타겟 데이터의 정확성 재검토")
        elif ratio < 0.67:
            print("  1. training_step에서 loss_p2에 1.5배 가중치 적용")
            print("     loss = loss_p1 + 1.5 * loss_p2")
            print("  2. P2 타겟 데이터의 정확성 재검토")
        
        if total_mae > 15:
            print("  3. 학습률 감소 고려 (현재 학습률의 0.5배)")
            print("  4. 더 많은 epoch 학습")
        
        if results['distance_error'] > 20:
            print("  5. 거리 제약 Loss 추가:")
            print("     distance_loss = F.mse_loss(pred_distance, gt_distance)")
            print("     total_loss = loss_p1 + loss_p2 + 0.1 * distance_loss")
        
    except Exception as e:
        print(f"❌ 성능 분석 실패: {e}")
        return
    
    print("\n✅ 검증 완료!")
    print("📋 요약:")
    print(f"  - P1 MAE: {results['p1_mae']:.2f}px")
    print(f"  - P2 MAE: {results['p2_mae']:.2f}px") 
    print(f"  - 균형성: {severity}")
    print(f"  - 전체 등급: {grade}")


if __name__ == "__main__":
    main()
