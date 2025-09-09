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
from src.losses.heatmap_utils import softargmax_2d
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
            "img_size": 256,  # train_img.py에서 기본값 확인 (size=512)
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
            "lambda_separation": 0.05,  # 분리 힌지 가중치 (기본: 비활성화)
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
        ckpt_dir = "runs/ckpts1"
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
    
    def preprocess_image(self, image_path: str, pkl_path: str = None) -> torch.Tensor:
        """
        이미지 전처리 - 학습 데이터셋과 동일한 전처리 적용
        
        Args:
            image_path: 입력 이미지 경로
            pkl_path: PKL 파일 경로 (제약조건 적용시 필요)
        """
        from src.data.dataset_img import detect_blue_roi, blue_mask_roi, load_points_from_pkl_raw, best_flip_from_polyline, map_with_range
        from torchvision.transforms.functional import to_tensor
        
        # 이미지 로드
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = img_rgb.shape[:2]
        
        # Blue ROI 검출 (학습과 동일)
        roi = detect_blue_roi(img_bgr, blue_min=70, diff_min=30)  # 학습 시 사용한 파라미터
        mask_roi = blue_mask_roi(img_bgr, roi, blue_min=70, diff_min=30)
        
        # PKL 폴리라인을 이용한 데이터 범위 및 flip 결정 (학습과 동일)
        flip_x, flip_y, data_range = False, False, (0, W0, 0, H0)  # 기본값
        if pkl_path and os.path.exists(pkl_path):
            try:
                poly = load_points_from_pkl_raw(pkl_path)
                flip_x, flip_y, data_range = best_flip_from_polyline(poly, roi, mask_roi)
            except Exception as e:
                print(f"⚠️ PKL 처리 실패, 기본값 사용: {e}")
        
        # 리사이즈 (학습과 동일)
        size = self.model_params["img_size"]
        img_resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
        
        # 정규화 및 텐서 변환 (학습과 동일: to_tensor 사용)
        img_tensor = to_tensor(img_resized)  # 자동으로 0-255 → 0-1 변환
        img_tensor = img_tensor.unsqueeze(0)  # 배치 차원 추가 [1, 3, H, W]
        
        # 스케일 팩터 계산 (원본 → 리사이즈)
        scale_x = W0 / size
        scale_y = H0 / size
        
        return img_tensor.to(self.device), (scale_x, scale_y), (H0, W0), {
            "roi": roi,
            "flip": (flip_x, flip_y), 
            "data_range": data_range,
            "orig_size": (H0, W0)
        }
    
    def _create_polyline_mask_for_inference(self, pkl_path: str, preprocess_info: dict) -> torch.Tensor:
        """
        추론용 폴리라인 마스크 생성 (학습 데이터셋과 동일한 방식)
        
        Args:
            pkl_path: PKL 파일 경로
            preprocess_info: 전처리 정보 (ROI, flip, data_range)
            
        Returns:
            [1, H, W] 폴리라인 마스크 텐서 (실패시 None)
        """
        try:
            from src.data.dataset_img import load_points_from_pkl_raw, map_with_range
            
            # PKL에서 폴리라인 로드
            poly = load_points_from_pkl_raw(pkl_path)
            
            # 전처리 정보 추출
            roi = preprocess_info["roi"]
            flip_x, flip_y = preprocess_info["flip"]
            data_range = preprocess_info["data_range"]
            
            # 폴리라인을 픽셀 좌표로 변환 (전처리와 동일)
            poly_px = map_with_range(poly, roi, data_range, flip_x, flip_y)
            
            # 리사이즈 스케일 적용
            size = self.model_params["img_size"]
            orig_h, orig_w = preprocess_info.get("orig_size", (roi[3]-roi[1], roi[2]-roi[0]))
            sx, sy = size / float(orig_w), size / float(orig_h)
            poly_r = poly_px.copy()
            poly_r[:,0] *= sx
            poly_r[:,1] *= sy
            
            # 폴리라인 마스크 생성 (학습 데이터셋과 동일)
            mask = self._create_polyline_mask(poly_r, size, size)
            
            # 텐서로 변환하고 배치 차원 추가
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)  # [1, H, W]
            return mask_tensor.to(self.device)
            
        except Exception as e:
            print(f"❌ 폴리라인 마스크 생성 실패: {e}")
            return None
    
    def _create_polyline_mask(self, polyline_points: np.ndarray, H: int, W: int, thickness: int = 8) -> np.ndarray:
        """
        폴리라인을 따라 마스크 생성 - 키포인트 제약조건 (학습 데이터셋과 동일)
        
        Args:
            polyline_points: [N, 2] 폴리라인 포인트들
            H, W: 마스크 크기
            thickness: 라인 두께 (기본값을 8로 증가)
            
        Returns:
            [H, W] 마스크 (0 또는 1)
        """
        mask = np.zeros((H, W), dtype=np.uint8)
        
        if len(polyline_points) < 2:
            print("⚠️ 폴리라인 포인트가 너무 적음")
            return mask.astype(np.float32)
        
        # 폴리라인을 이미지에 그리기
        pts = polyline_points.astype(np.int32)
        pts = np.clip(pts, 0, [W-1, H-1])  # 이미지 경계 내로 제한
        
        # 연결선 그리기
        for i in range(len(pts)-1):
            cv2.line(mask, tuple(pts[i]), tuple(pts[i+1]), 255, thickness)
        
        # 각 폴리라인 점 주변도 마킹 (점 끝부분 보강)
        for pt in pts:
            cv2.circle(mask, tuple(pt), thickness//2 + 2, 255, -1)
        
        # 마스크 확장 (morphology)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        result_mask = (mask > 0).astype(np.float32)
        
        # 마스크 유효성 검사
        mask_pixels = np.sum(result_mask)
        total_pixels = H * W
        print(f"🔍 생성된 마스크: {mask_pixels:.0f}/{total_pixels} 픽셀 ({mask_pixels/total_pixels*100:.1f}%)")
        
        return result_mask
    
    def predict(self, image_path: str, pkl_path: str = None, use_constraint: bool = True, force_constraint: bool = False) -> dict:
        """
        단일 이미지에 대한 키포인트 예측
        
        Args:
            image_path: 입력 이미지 경로
            pkl_path: PKL 파일 경로 (제약조건 적용시 필요)
            use_constraint: 폴리라인 제약조건 사용 여부
            force_constraint: 제약조건 강제 적용 (마스크가 작아도 적용)
            
        Returns:
            예측 결과 딕셔너리
        """
        # PKL 경로 자동 추정
        if pkl_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            pkl_candidates = [
                os.path.join(os.path.dirname(image_path), "..", "pkl", f"{base_name}.pkl"),  # 상위 폴더의 pkl 디렉토리
                os.path.join(os.path.dirname(image_path), "pkl", f"{base_name}.pkl"),      # 같은 폴더의 pkl 디렉토리
                os.path.join(os.path.dirname(image_path), f"{base_name}.pkl"),            # 같은 폴더
            ]
            for candidate in pkl_candidates:
                if os.path.exists(candidate):
                    pkl_path = candidate
                    break
        
        with torch.no_grad():
            # 전처리 (학습과 동일)
            img_tensor, (scale_x, scale_y), (orig_h, orig_w), preprocess_info = self.preprocess_image(image_path, pkl_path)
            
            # 모델 추론
            logits = self.model(img_tensor)  # [1, 2, H, W]
            
            # 폴리라인 제약조건 적용 (학습과 동일)
            polyline_mask = None
            if use_constraint and pkl_path and os.path.exists(pkl_path):
                print(f"✅ PKL 파일 사용: {pkl_path}")
                print(f"📊 전처리 정보: ROI={preprocess_info['roi']}, Flip={preprocess_info['flip']}")
                
                # 폴리라인 마스크 생성 (학습 데이터셋과 동일한 방식)
                polyline_mask = self._create_polyline_mask_for_inference(pkl_path, preprocess_info)
                if polyline_mask is not None:
                    # 마스크 유효성 검사
                    mask_sum = polyline_mask.sum().item()
                    total_pixels = polyline_mask.numel()
                    mask_ratio = mask_sum / total_pixels
                    print(f"📊 마스크 통계: {mask_sum:.0f}/{total_pixels} 픽셀 ({mask_ratio*100:.1f}%)")
                    
                    if mask_ratio > 0.01 or force_constraint:  # 마스크가 충분하거나 강제 적용
                        # 폴리라인 영역만 활성화 (학습/평가와 동일)
                        masked_logits = logits * polyline_mask.unsqueeze(1)  # [1,1,H,W] -> [1,2,H,W]
                        probs = torch.sigmoid(masked_logits)
                        constraint_type = "강제 적용" if force_constraint and mask_ratio <= 0.01 else "정상 적용"
                        print(f"🎯 폴리라인 제약조건 {constraint_type}")
                    else:
                        print("⚠️ 마스크가 너무 작음 - 제약조건 미적용")
                        probs = torch.sigmoid(logits)
                        polyline_mask = None
                else:
                    probs = torch.sigmoid(logits)
                    print("⚠️ 폴리라인 마스크 생성 실패 - 제약조건 미적용")
            else:
                print("⚠️ PKL 파일 없음 - 제약조건 미적용")
                probs = torch.sigmoid(logits)
            
            # 좌표 추출 (softargmax 방식 - 학습과 동일)
            coords_pred = softargmax_2d(probs, temperature=self.model_params["temperature"])  # [1, 2, 2]
            coords_pred = coords_pred[0].cpu().numpy()  # [2, 2]
            
            # 원본 이미지 크기로 스케일 복원
            coords_pred[:, 0] *= scale_x  # x 좌표
            coords_pred[:, 1] *= scale_y  # y 좌표
            
            # 결과 정리
            result = {
                "image_path": image_path,
                "pkl_path": pkl_path,
                "keypoints": coords_pred,  # [[x1, y1], [x2, y2]]
                "point1": coords_pred[0],  # [x1, y1] 
                "point2": coords_pred[1],  # [x2, y2]
                "original_size": (orig_w, orig_h),
                "confidence_maps": probs[0].cpu().numpy(),  # [2, H, W]
                "scale_factors": (scale_x, scale_y),
                "preprocess_info": preprocess_info,
                # 폴리라인 마스크 추가 (시각화용)
                "polyline_mask": polyline_mask[0].cpu().numpy() if polyline_mask is not None else None
            }
            
            return result
    
    def predict_folder(self, input_dir: str, output_dir: str, 
                      save_overlay: bool = True, save_csv: bool = True, max_images: int = 20,
                      enable_visualization: bool = False) -> list:
        """
        폴더 내 이미지에 대한 배치 예측 (제한된 개수)
        
        Args:
            input_dir: 입력 이미지 폴더
            output_dir: 결과 저장 폴더
            save_overlay: 오버레이 이미지 저장 여부
            save_csv: CSV 결과 저장 여부
            max_images: 처리할 최대 이미지 개수 (기본값: 20)
            enable_visualization: 고급 시각화 기능 활성화 여부
            
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
            image_files = image_files[:max_images]  # 뒤에서부터 max_images개 선택
            print(f"⚡ 처리 개수를 {max_images}개로 제한 (하위 부분)")
        
        print(f"🔄 실제 처리할 이미지: {len(image_files)}개")
        
        all_results = []
        
        # 고급 시각화 도구 초기화 (요청시)
        viz_manager = None
        if enable_visualization:
            try:
                from src.utils.training_visualizer import VisualizationManager
                viz_manager = VisualizationManager(save_dir=os.path.join(output_dir, "visualizations"))
                print("✅ 고급 시각화 모드 활성화")
            except Exception as e:
                print(f"⚠️ 시각화 도구 초기화 실패: {e}")
        
        for i, img_path in enumerate(image_files):
            print(f"🔄 처리중 ({i+1}/{len(image_files)}): {os.path.basename(img_path)}")
            
            try:
                # 예측 수행 (PKL 파일 자동 검색 포함)
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
                
                # 고급 시각화 (요청시)
                if viz_manager is not None:
                    try:
                        viz_files = viz_manager.visualize_inference_results(
                            model=self.model,
                            image_path=img_path,
                            result=result,
                            save_prefix=base_name
                        )
                        if viz_files:
                            print(f"🎨 고급 시각화 저장: {len(viz_files)}개 파일")
                    except Exception as e:
                        print(f"⚠️ 고급 시각화 실패 {base_name}: {e}")
                
            except Exception as e:
                print(f"❌ 오류 발생 {os.path.basename(img_path)}: {e}")
                continue
        
        print(f"✅ 배치 처리 완료: {len(all_results)}/{len(image_files)} 성공")
        
        # 배치 요약 시각화 (고급 시각화 모드일 때)
        if viz_manager is not None and all_results:
            try:
                summary_path = viz_manager.create_batch_summary(all_results)
                print(f"📊 배치 요약 시각화 생성: {summary_path}")
            except Exception as e:
                print(f"⚠️ 배치 요약 생성 실패: {e}")
        
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
    parser.add_argument("--enable_visualization", action="store_true",
                       help="고급 시각화 기능 활성화 (CNN activation, 히트맵 분석 등)")
    
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
                max_images=args.max_images,
                enable_visualization=args.enable_visualization
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
