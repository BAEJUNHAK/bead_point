#!/usr/bin/env python3
"""
완전한 비드 포인트 검출 추론 코드
모든 파라미터를 코드베이스에서 직접 추출하여 구성
"""
import os
import glob
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path

# 프로젝트 모듈 임포트
from src.models.lit_model import LitKeypoint2
from src.models.img_backbone import DWUNetTiny
from src.losses.heatmap_utils import heatmaps_to_coords_argmax
from src.utils.viz import save_overlay


class BeadPointInference:
    """비드 포인트 검출 추론 클래스"""
    
    def __init__(self, checkpoint_path: str = None, device: str = "auto"):
        """
        Args:
            checkpoint_path: Lightning 체크포인트 파일 경로 (.ckpt)
            device: 추론 디바이스 ("auto", "cpu", "cuda")
        """
        self.device = self._setup_device(device)
        
        # 새로운 손실 시스템 파라미터
        self.model_params = {
            "img_size": 256,  # hparams.yaml에서 확인
            "sigma": 3.0,     # 새로운 권장값: 3px (기존 5.0에서 변경)
            "sigma_polyline": 3.0,
            "polyline_weight": 0.3,
            "pck_thresh_px": 5.0,
            "lr": 0.0001, 
            "base": 64,  # lit_model.py에서 확인된 base=64
            "in_ch": 3,
            "out_ch": 2,
            # 새로운 손실 시스템 파라미터
            "lambda_heatmap": 1.0,     # 히트맵 MSE 가중치
            "lambda_coord": 0.2,       # 좌표 Huber 가중치
            "lambda_separation": 0.0,  # 분리 힌지 가중치 (기본: 비활성화)
            "huber_delta": 2.0,        # Huber 전환점 (2px)
            "min_separation": 5.0,     # 최소 분리 거리 (5px)
            "temperature": 0.02        # soft-argmax 온도 (0.02)
        }
        
        # 최적 체크포인트 파일 자동 선택
        if checkpoint_path is None:
            checkpoint_path = self._find_best_checkpoint()
        
        self.model = self._load_model(checkpoint_path)
        print(f"✅ 모델 로드 완료: {checkpoint_path}")
        print(f"📊 모델 파라미터: {self.model_params}")
        
    def _setup_device(self, device: str) -> torch.device:
        """디바이스 설정"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        print(f"🔧 추론 디바이스: {device}")
        return device
    
    def _find_best_checkpoint(self) -> str:
        """최적 체크포인트 자동 찾기 (가장 낮은 validation MAE)"""
        ckpt_dir = "runs/ckpts"
        if not os.path.exists(ckpt_dir):
            raise FileNotFoundError(f"체크포인트 디렉토리를 찾을 수 없습니다: {ckpt_dir}")
        
        ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {ckpt_dir}")
        
        # val_mae_px 값으로 정렬하여 최적 모델 선택
        best_ckpt = None
        best_mae = float('inf')
        
        for ckpt_file in ckpt_files:
            filename = os.path.basename(ckpt_file)
            # kp2-epoch=014-val_mae_px=10.64.ckpt 형식에서 MAE 추출
            try:
                mae_part = filename.split('val_mae_px=')[1].split('.ckpt')[0]
                mae_value = float(mae_part)
                if mae_value < best_mae:
                    best_mae = mae_value
                    best_ckpt = ckpt_file
            except (IndexError, ValueError):
                continue
        
        if best_ckpt is None:
            # 파싱 실패시 첫 번째 파일 사용
            best_ckpt = ckpt_files[0]
            print(f"⚠️ MAE 파싱 실패, 첫 번째 체크포인트 사용")
        
        print(f"🎯 최적 체크포인트 선택: {os.path.basename(best_ckpt)} (MAE: {best_mae:.2f})")
        return best_ckpt
    
    def _load_model(self, checkpoint_path: str) -> LitKeypoint2:
        """Lightning 체크포인트에서 모델 로드"""
        try:
            # Lightning 체크포인트에서 로드
            model = LitKeypoint2.load_from_checkpoint(
                checkpoint_path,
                map_location=self.device,
                img_size=self.model_params["img_size"],
                sigma=self.model_params["sigma"],
                sigma_polyline=self.model_params["sigma_polyline"],
                polyline_weight=self.model_params["polyline_weight"],
                pck_thresh_px=self.model_params["pck_thresh_px"],
                lr=self.model_params["lr"],
                lambda_heatmap=self.model_params["lambda_heatmap"],
                lambda_coord=self.model_params["lambda_coord"],
                lambda_separation=self.model_params["lambda_separation"],
                huber_delta=self.model_params["huber_delta"],
                min_separation=self.model_params["min_separation"],
                temperature=self.model_params["temperature"]
            )
        except Exception as e:
            print(f"⚠️ Lightning 로드 실패, 수동 로드 시도: {e}")
            # 수동 로드 방식
            model = self._manual_load_model(checkpoint_path)
        
        model.eval()
        model.to(self.device)
        return model
    
    def _manual_load_model(self, checkpoint_path: str) -> LitKeypoint2:
        """수동 모델 로드 (Lightning 로드 실패시)"""
        # 모델 구조 생성
        model = LitKeypoint2(
            img_size=self.model_params["img_size"],
            sigma=self.model_params["sigma"],
            sigma_polyline=self.model_params["sigma_polyline"], 
            polyline_weight=self.model_params["polyline_weight"],
            pck_thresh_px=self.model_params["pck_thresh_px"],
            lr=self.model_params["lr"],
            lambda_heatmap=self.model_params["lambda_heatmap"],
            lambda_coord=self.model_params["lambda_coord"],
            lambda_separation=self.model_params["lambda_separation"],
            huber_delta=self.model_params["huber_delta"],
            min_separation=self.model_params["min_separation"],
            temperature=self.model_params["temperature"]
        )
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # state_dict 추출
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # state_dict 키 정리 (Lightning 접두어 제거)
        clean_state_dict = {}
        for key, value in state_dict.items():
            # "net." 접두어 제거 및 기타 Lightning 키 처리
            if key.startswith("net."):
                clean_key = key.replace("net.", "net.")  # 그대로 유지
            else:
                clean_key = key
            clean_state_dict[clean_key] = value
        
        model.load_state_dict(clean_state_dict, strict=False)
        return model
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """이미지 전처리"""
        # 이미지 로드
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = img_rgb.shape[:2]
        
        # 리사이즈 (모델 입력 크기에 맞춤)
        size = self.model_params["img_size"]
        img_resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
        
        # 정규화 및 텐서 변환
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # 배치 차원 추가 [1, 3, H, W]
        
        # 스케일 팩터 저장 (후처리용)
        scale_x = W0 / size
        scale_y = H0 / size
        
        return img_tensor.to(self.device), (scale_x, scale_y), (H0, W0)
    
    def predict(self, image_path: str, use_constraint: bool = True) -> dict:
        """
        단일 이미지에 대한 키포인트 예측
        
        Args:
            image_path: 입력 이미지 경로
            use_constraint: 폴리라인 제약조건 사용 여부
            
        Returns:
            예측 결과 딕셔너리
        """
        with torch.no_grad():
            # 전처리
            img_tensor, (scale_x, scale_y), (orig_h, orig_w) = self.preprocess_image(image_path)
            
            # 모델 추론
            logits = self.model(img_tensor)  # [1, 2, H, W]
            
            if use_constraint:
                # 폴리라인 제약조건이 있다면 적용
                # 실제 데이터에서는 PKL 파일이나 폴리라인 마스크가 필요
                # 여기서는 제약 없이 진행
                print("⚠️ 폴리라인 제약조건을 위해서는 PKL 파일이 필요합니다.")
            
            # 확률로 변환
            probs = torch.sigmoid(logits)
            
            # 좌표 추출 (argmax 방식)
            coords_pred = heatmaps_to_coords_argmax(probs)  # [1, 2, 2]
            coords_pred = coords_pred[0].cpu().numpy()  # [2, 2]
            
            # 원본 이미지 크기로 스케일 복원
            coords_pred[:, 0] *= scale_x  # x 좌표
            coords_pred[:, 1] *= scale_y  # y 좌표
            
            # 결과 정리
            result = {
                "image_path": image_path,
                "keypoints": coords_pred,  # [[x1, y1], [x2, y2]]
                "point1": coords_pred[0],  # [x1, y1] 
                "point2": coords_pred[1],  # [x2, y2]
                "original_size": (orig_w, orig_h),
                "confidence_maps": probs[0].cpu().numpy(),  # [2, H, W]
                "scale_factors": (scale_x, scale_y)
            }
            
            return result
    
    def predict_folder(self, input_dir: str, output_dir: str, 
                      save_overlay: bool = True, save_csv: bool = True, max_images: int = 20) -> list:
        """
        폴더 내 이미지에 대한 배치 예측 (제한된 개수)
        
        Args:
            input_dir: 입력 이미지 폴더
            output_dir: 결과 저장 폴더
            save_overlay: 오버레이 이미지 저장 여부
            save_csv: CSV 결과 저장 여부
            max_images: 처리할 최대 이미지 개수 (기본값: 20)
            
        Returns:
            예측 결과 리스트
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 지원하는 이미지 확장자
        img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_files = []
        for ext in img_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
        
        if not image_files:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {input_dir}")
        
        image_files = sorted(image_files)
        print(f"📂 총 {len(image_files)}개 이미지 파일 발견")
        
        # 최대 개수로 제한
        if len(image_files) > max_images:
            image_files = image_files[:max_images]
            print(f"⚡ 처리 개수를 {max_images}개로 제한")
        
        print(f"🔄 실제 처리할 이미지: {len(image_files)}개")
        
        all_results = []
        
        for i, img_path in enumerate(image_files):
            print(f"🔄 처리중 ({i+1}/{len(image_files)}): {os.path.basename(img_path)}")
            
            try:
                # 예측 수행
                result = self.predict(img_path)
                all_results.append(result)
                
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # CSV 저장
                if save_csv:
                    csv_path = os.path.join(output_dir, f"{base_name}_pred.csv")
                    np.savetxt(csv_path, result["keypoints"], delimiter=",", fmt="%.2f")
                
                # 오버레이 이미지 저장
                if save_overlay:
                    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
                    self._save_overlay_image(img_path, result, overlay_path)
                
            except Exception as e:
                print(f"❌ 오류 발생 {os.path.basename(img_path)}: {e}")
                continue
        
        print(f"✅ 배치 처리 완료: {len(all_results)}/{len(image_files)} 성공")
        return all_results
    
    def _save_overlay_image(self, original_path: str, result: dict, save_path: str):
        """결과를 오버레이한 이미지 저장"""
        # 원본 이미지 로드
        img = cv2.imread(original_path)
        if img is None:
            return
        
        # 키포인트 그리기
        point1 = result["point1"].astype(int)
        point2 = result["point2"].astype(int)
        
        # 점 1 (빨간색 X)
        cv2.drawMarker(img, tuple(point1), (0, 0, 255), 
                      cv2.MARKER_TILTED_CROSS, 20, 3)
        
        # 점 2 (파란색 원)
        cv2.circle(img, tuple(point2), 8, (255, 0, 0), 3)
        
        # 연결선 (노란색)
        cv2.line(img, tuple(point1), tuple(point2), (0, 255, 255), 2)
        
        # 좌표 텍스트 추가
        cv2.putText(img, f"P1: ({point1[0]}, {point1[1]})", 
                   (point1[0] + 10, point1[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, f"P2: ({point2[0]}, {point2[1]})", 
                   (point2[0] + 10, point2[1] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.imwrite(save_path, img)


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="비드 포인트 검출 추론")
    parser.add_argument("--input", required=True, 
                       help="입력 이미지 파일 또는 폴더 경로")
    parser.add_argument("--output", default="runs/inference_results", 
                       help="결과 저장 폴더")
    parser.add_argument("--checkpoint", default=None, 
                       help="체크포인트 파일 경로 (미지정시 자동 선택)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                       help="추론 디바이스")
    parser.add_argument("--save_overlay", action="store_true", default=True,
                       help="오버레이 이미지 저장")
    parser.add_argument("--save_csv", action="store_true", default=True,
                       help="CSV 결과 저장")
    parser.add_argument("--max_images", type=int, default=20,
                       help="폴더 배치 처리시 최대 이미지 개수 (기본값: 20)")
    
    args = parser.parse_args()
    
    print("🚀 비드 포인트 검출 추론 시작")
    print(f"📄 입력: {args.input}")
    print(f"📁 출력: {args.output}")
    
    # 추론 객체 생성
    try:
        inferencer = BeadPointInference(
            checkpoint_path=args.checkpoint,
            device=args.device
        )
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 입력이 파일인지 폴더인지 확인
    if os.path.isfile(args.input):
        # 단일 파일 처리
        print("📄 단일 이미지 처리 모드")
        try:
            result = inferencer.predict(args.input)
            
            # 결과 출력
            print("\n📊 예측 결과:")
            print(f"  키포인트 1: ({result['point1'][0]:.2f}, {result['point1'][1]:.2f})")
            print(f"  키포인트 2: ({result['point2'][0]:.2f}, {result['point2'][1]:.2f})")
            
            # 결과 저장
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            
            if args.save_csv:
                csv_path = os.path.join(args.output, f"{base_name}_pred.csv")
                np.savetxt(csv_path, result["keypoints"], delimiter=",", fmt="%.2f")
                print(f"💾 CSV 저장: {csv_path}")
            
            if args.save_overlay:
                overlay_path = os.path.join(args.output, f"{base_name}_overlay.png")
                inferencer._save_overlay_image(args.input, result, overlay_path)
                print(f"🖼️ 오버레이 저장: {overlay_path}")
                
        except Exception as e:
            print(f"❌ 예측 실패: {e}")
            
    elif os.path.isdir(args.input):
        # 폴더 처리
        print("📂 폴더 배치 처리 모드")
        try:
            results = inferencer.predict_folder(
                args.input, args.output, 
                save_overlay=args.save_overlay,
                save_csv=args.save_csv,
                max_images=args.max_images
            )
            
            # 전체 통계
            if results:
                print("\n📊 배치 처리 통계:")
                all_distances = []
                for result in results:
                    p1, p2 = result["point1"], result["point2"]
                    distance = np.linalg.norm(p2 - p1)
                    all_distances.append(distance)
                
                print(f"  처리 성공: {len(results)}개 이미지")
                print(f"  평균 키포인트 거리: {np.mean(all_distances):.2f} 픽셀")
                print(f"  거리 범위: {np.min(all_distances):.2f} ~ {np.max(all_distances):.2f} 픽셀")
                
        except Exception as e:
            print(f"❌ 배치 처리 실패: {e}")
    else:
        print(f"❌ 잘못된 입력 경로: {args.input}")
    
    print("🏁 추론 완료!")


if __name__ == "__main__":
    main()
