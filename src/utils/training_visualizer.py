# src/utils/training_visualizer.py
"""
학습 중 자동 시각화 및 저장 도구
"""
import os
import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt

# 시각화 기능을 직접 구현 (권한 문제 완전 해결)
VISUALIZATION_AVAILABLE = True

class SimpleActivationHook:
    """간단한 Activation Hook"""
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def hook_fn(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def register_hooks(self, model, layer_names):
        """훅 등록"""
        self.clear_hooks()
        model_modules = dict(model.named_modules())
        
        for name in layer_names:
            if name in model_modules:
                hook = model_modules[name].register_forward_hook(self.hook_fn(name))
                self.hooks.append(hook)
        
        return len(self.hooks)
    
    def clear_hooks(self):
        """훅 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()

class SimpleVisualizationTool:
    """간단한 시각화 도구"""
    
    @staticmethod
    def visualize_heatmap_prediction(image, pred_heatmaps, target_heatmaps, 
                                   pred_coords, target_coords, save_path,
                                   polyline_mask=None, title="Heatmap Visualization"):
        """히트맵 시각화"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(title, fontsize=16)
            
            # 원본 이미지
            axes[0, 0].imshow(image)
            axes[0, 0].scatter(pred_coords[:, 0], pred_coords[:, 1], c='red', s=50, label='Predicted')
            axes[0, 0].scatter(target_coords[:, 0], target_coords[:, 1], c='blue', s=50, label='Target')
            axes[0, 0].set_title('Image + Points')
            axes[0, 0].legend()
            axes[0, 0].axis('off')
            
            # 예측 히트맵들
            axes[0, 1].imshow(pred_heatmaps[0], cmap='hot')
            axes[0, 1].set_title('Pred Heatmap P1')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(pred_heatmaps[1], cmap='hot')
            axes[0, 2].set_title('Pred Heatmap P2')
            axes[0, 2].axis('off')
            
            # 타겟 히트맵들
            axes[1, 1].imshow(target_heatmaps[0], cmap='hot')
            axes[1, 1].set_title('Target Heatmap P1')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(target_heatmaps[1], cmap='hot')
            axes[1, 2].set_title('Target Heatmap P2')
            axes[1, 2].axis('off')
            
            # 폴리라인 마스크 (있는 경우)
            if polyline_mask is not None:
                axes[1, 0].imshow(image)
                # 마스크 오버레이
                mask_colored = np.zeros((*polyline_mask.shape, 4))
                mask_colored[..., 1] = polyline_mask  # 초록색
                mask_colored[..., 3] = polyline_mask * 0.5  # 투명도
                axes[1, 0].imshow(mask_colored)
                axes[1, 0].set_title('Polyline Constraint')
            else:
                axes[1, 0].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ 히트맵 시각화 실패: {e}")
    
    @staticmethod
    def visualize_activations(model, input_tensor, layer_names, save_dir, save_prefix):
        """CNN Activation 시각화"""
        try:
            hook = SimpleActivationHook()
            hook.register_hooks(model, layer_names[:4])  # 처음 4개 레이어만
            
            # Forward pass
            with torch.no_grad():
                _ = model(input_tensor)
            
            # 각 레이어별로 저장
            for layer_name, activation in hook.activations.items():
                feature_maps = activation[0]  # 첫 번째 배치
                num_channels = min(feature_maps.shape[0], 6)  # 최대 6개 채널
                
                layer_clean_name = layer_name.replace('.', '_')
                
                for i in range(num_channels):
                    feature_map = feature_maps[i].cpu().numpy()
                    
                    # 정규화
                    if feature_map.max() > feature_map.min():
                        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
                    
                    # 개별 이미지로 저장
                    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                    im = ax.imshow(feature_map, cmap='viridis', interpolation='nearest')
                    ax.set_title(f'{layer_name} - Channel {i}', fontsize=10)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    
                    save_path = os.path.join(save_dir, f"{save_prefix}_{layer_clean_name}_ch{i:02d}.png")
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
            
            hook.clear_hooks()
            print(f"💾 CNN Activation 저장 완료: {len(hook.activations)} 레이어")
            
        except Exception as e:
            print(f"⚠️ CNN Activation 시각화 실패: {e}")

print("✅ 내장 시각화 모듈 로드 완료")


class TrainingVisualizerCallback(Callback):
    """
    학습 중 자동 시각화를 위한 PyTorch Lightning Callback
    """
    
    def __init__(self,
                 save_dir: str = "runs/training_visualizations",
                 save_every_n_epochs: int = 5,
                 max_samples_per_epoch: int = 4,
                 visualize_activations: bool = True,
                 visualize_heatmaps: bool = True,
                 activation_layers: List[str] = None):
        """
        Args:
            save_dir: 시각화 결과 저장 디렉토리
            save_every_n_epochs: N 에포크마다 시각화 저장
            max_samples_per_epoch: 에포크당 최대 샘플 수
            visualize_activations: CNN activation 시각화 여부
            visualize_heatmaps: 히트맵 시각화 여부
            activation_layers: 시각화할 activation 레이어들
        """
        self.save_dir = save_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.max_samples_per_epoch = max_samples_per_epoch
        self.visualize_activations = visualize_activations
        self.visualize_heatmaps = visualize_heatmaps
        self.activation_layers = activation_layers or [
            'e1.0.dw', 'e1.1.dw', 
            'e2.0.dw', 'e2.1.dw',
            'e3.0.dw', 'e3.1.dw',
            'e4.0.dw', 'e4.1.dw',
            'd3.0.dw', 'd2.0.dw', 'd1.0.dw'
        ]
        
        self.sample_counter = 0
        self.current_epoch_samples = []
        
        # 시각화 기능 활성화 (내장 도구 사용)
        self.viz_tool = SimpleVisualizationTool()
        print("✅ 내장 시각화 도구 초기화 완료")
    
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        """콜백 초기화 (권한 문제 해결)"""
        try:
            # 메인 디렉토리 생성
            os.makedirs(self.save_dir, exist_ok=True)
            
            # 서브 디렉토리들 생성
            subdirs = ["heatmaps", "activations", "comparison"]
            for subdir in subdirs:
                full_path = os.path.join(self.save_dir, subdir)
                os.makedirs(full_path, exist_ok=True)
            
            # 권한 테스트
            test_file = os.path.join(self.save_dir, ".permission_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            
            print(f"✅ TrainingVisualizerCallback 설정 완료")
            print(f"📁 저장 디렉토리: {self.save_dir}")
            print(f"🔄 {self.save_every_n_epochs} 에포크마다 시각화 저장")
            
        except Exception as e:
            print(f"⚠️ 시각화 디렉토리 설정 실패: {e}")
            print(f"🔧 대체 경로 시도...")
            
            # 대체 경로 시도 (현재 작업 디렉토리)
            import tempfile
            import time
            
            # 임시 디렉토리 사용
            temp_base = tempfile.gettempdir()
            timestamp = str(int(time.time()))[-8:]
            self.save_dir = os.path.join(temp_base, f"bead_viz_{timestamp}")
            
            try:
                os.makedirs(self.save_dir, exist_ok=True)
                subdirs = ["heatmaps", "activations", "comparison"]
                for subdir in subdirs:
                    full_path = os.path.join(self.save_dir, subdir)
                    os.makedirs(full_path, exist_ok=True)
                
                print(f"✅ 대체 디렉토리 사용: {self.save_dir}")
                
            except Exception as e2:
                print(f"❌ 시각화 완전 실패: {e2}")
                # 시각화 기능 비활성화
                self.visualize_heatmaps = False
                self.visualize_activations = False
        
        print("✅ TrainingVisualizerCallback setup 완료")
    
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, 
                          outputs, batch, batch_idx):
        """학습 배치 종료 시 샘플 수집"""
        current_epoch = trainer.current_epoch
        
        # 지정된 에포크에서만 샘플 수집
        if (current_epoch + 1) % self.save_every_n_epochs != 0:
            return
        
        # 샘플 수 제한
        if len(self.current_epoch_samples) >= self.max_samples_per_epoch:
            return
        
        try:
            # 배치에서 첫 번째 샘플만 저장 (메모리 절약)
            sample_data = {
                'image': batch['image'][0].cpu(),  # [3, H, W]
                'target_coords': batch['coord_px'][0].cpu(),  # [2, 2]
                'polyline_mask': batch.get('polyline_mask', [None])[0],  # [H, W] or None
                'batch_idx': batch_idx,
                'epoch': current_epoch
            }
            
            # polyline_mask 처리
            if sample_data['polyline_mask'] is not None:
                sample_data['polyline_mask'] = sample_data['polyline_mask'].cpu()
            
            self.current_epoch_samples.append(sample_data)
            
        except Exception as e:
            print(f"⚠️ 샘플 수집 중 오류: {e}")
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """학습 에포크 종료 시 시각화 실행"""
        current_epoch = trainer.current_epoch
        
        # 지정된 에포크에서만 시각화
        if (current_epoch + 1) % self.save_every_n_epochs != 0:
            return
        

        
        if not self.current_epoch_samples:
            print(f"⚠️ 에포크 {current_epoch}: 시각화할 샘플 없음")
            return
        
        print(f"🎨 에포크 {current_epoch}: {len(self.current_epoch_samples)}개 샘플 시각화 시작")
        
        try:
            for i, sample_data in enumerate(self.current_epoch_samples):
                self._visualize_sample(pl_module, sample_data, current_epoch, i)
                
        except Exception as e:
            print(f"❌ 시각화 중 오류: {e}")
            import traceback
            print(f"📍 상세 오류 정보:")
            traceback.print_exc()
        finally:
            # 샘플 초기화
            self.current_epoch_samples.clear()
    
    def _visualize_sample(self, pl_module: pl.LightningModule, sample_data: Dict, 
                         epoch: int, sample_idx: int):
        """개별 샘플 시각화"""
        # 시각화 모듈이 없으면 건너뛰기
        if not VISUALIZATION_AVAILABLE:
            return
        
        # 디버깅: 현재 객체 상태 확인
        print(f"🔍 디버깅 - 현재 객체 속성들:")
        print(f"   - visualize_heatmaps: {self.visualize_heatmaps}")
        print(f"   - visualize_activations: {self.visualize_activations}")
        print(f"   - viz_tool 존재: {hasattr(self, 'viz_tool')}")
        print(f"   - heatmap_viz_tool 존재: {hasattr(self, 'heatmap_viz_tool')}")
        print(f"   - cnn_viz_tool 존재: {hasattr(self, 'cnn_viz_tool')}")
        
        if hasattr(self, 'viz_tool'):
            print(f"   - viz_tool 타입: {type(self.viz_tool)}")
        
        try:
            # 데이터 준비
            image_tensor = sample_data['image'].unsqueeze(0).to(pl_module.device)  # [1, 3, H, W]
            target_coords = sample_data['target_coords'].numpy()  # [2, 2]
            polyline_mask = sample_data['polyline_mask']
            
            # 모델 예측
            pl_module.eval()
            with torch.no_grad():
                logits = pl_module(image_tensor)  # [1, 2, H, W]
                
                # 폴리라인 제약조건 적용 (있는 경우)
                if polyline_mask is not None:
                    mask_tensor = polyline_mask.unsqueeze(0).unsqueeze(0).to(pl_module.device)  # [1, 1, H, W]
                    masked_logits = logits * mask_tensor
                    probs = torch.sigmoid(masked_logits)
                else:
                    probs = torch.sigmoid(logits)
                
                # 좌표 추출
                from ..losses.heatmap_utils import softargmax_2d
                pred_coords = softargmax_2d(probs, temperature=pl_module.loss_fn.temperature)[0].cpu().numpy()
            
            # 타겟 히트맵 생성
            from ..losses.heatmap_utils import make_heatmaps_torch
            H, W = logits.shape[-2:]
            target_heatmaps = make_heatmaps_torch(
                sample_data['target_coords'].unsqueeze(0), H, W, pl_module.sigma
            )[0].cpu().numpy()  # [2, H, W]
            
            # 이미지 전처리 (시각화용)
            image_np = sample_data['image'].permute(1, 2, 0).numpy()  # [H, W, 3]
            image_np = (image_np * 255).astype(np.uint8)  # 0-1 -> 0-255
            
            # 1. 히트맵 시각화
            if self.visualize_heatmaps:
                # 디렉토리 확실히 생성
                heatmap_dir = os.path.join(self.save_dir, "heatmaps")
                os.makedirs(heatmap_dir, exist_ok=True)
                
                heatmap_save_path = os.path.join(
                    heatmap_dir, 
                    f"epoch_{epoch:03d}_sample_{sample_idx}_heatmap.png"
                )
                
                # 폴리라인 마스크 준비 (있는 경우)
                polyline_mask_np = None
                if polyline_mask is not None:
                    polyline_mask_np = polyline_mask.cpu().numpy()
                
                self.viz_tool.visualize_heatmap_prediction(
                    image=image_np,
                    pred_heatmaps=probs[0].cpu().numpy(),
                    target_heatmaps=target_heatmaps,
                    pred_coords=pred_coords,
                    target_coords=target_coords,
                    save_path=heatmap_save_path,
                    polyline_mask=polyline_mask_np,
                    title=f"Epoch {epoch} - Sample {sample_idx} - Heatmap + Constraint Visualization"
                )
            
            # 2. CNN Activation 시각화 (내장 도구 사용)
            if self.visualize_activations:
                try:
                    # 사용 가능한 레이어 자동 감지
                    available_layers = []
                    for name, module in pl_module.net.named_modules():
                        if isinstance(module, torch.nn.Conv2d):
                            available_layers.append(name)
                    
                    print(f"🔍 감지된 Conv2d 레이어들: {available_layers}")
                    
                    # 주요 레이어만 선택 (처음 4개)
                    key_layers = available_layers[:4] if len(available_layers) >= 4 else available_layers
                    
                    if key_layers:
                        # 디렉토리 확실히 생성
                        activations_base_dir = os.path.join(self.save_dir, "activations")
                        os.makedirs(activations_base_dir, exist_ok=True)
                        
                        activation_save_dir = os.path.join(activations_base_dir, f"epoch_{epoch:03d}_sample_{sample_idx}")
                        os.makedirs(activation_save_dir, exist_ok=True)
                        
                        print(f"🎨 CNN 시각화 시작: {key_layers}")
                        self.viz_tool.visualize_activations(
                            model=pl_module.net,
                            input_tensor=image_tensor,
                            layer_names=key_layers,
                            save_dir=activation_save_dir,
                            save_prefix=f"epoch_{epoch}_sample_{sample_idx}"
                        )
                        print(f"✅ CNN 시각화 완료: {len(key_layers)}개 레이어")
                    else:
                        print("⚠️ 시각화할 수 있는 Conv2d 레이어를 찾을 수 없습니다")
                        
                except Exception as e:
                    print(f"⚠️ CNN Activation 시각화 실패: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 3. 비교 시각화 (간단한 오버레이)
            try:
                # 디렉토리 확실히 생성
                comparison_dir = os.path.join(self.save_dir, "comparison")
                os.makedirs(comparison_dir, exist_ok=True)
                
                comparison_save_path = os.path.join(
                    comparison_dir,
                    f"epoch_{epoch:03d}_sample_{sample_idx}_comparison.png"
                )
                self._create_simple_comparison(
                    image_np, pred_coords, target_coords, comparison_save_path, epoch, sample_idx
                )
            except Exception as e:
                print(f"⚠️ 비교 시각화 생성 실패: {e}")
            
        except Exception as e:
            print(f"❌ 샘플 {sample_idx} 시각화 오류: {e}")
            import traceback
            print(f"📍 샘플 {sample_idx} 상세 오류:")
            traceback.print_exc()
    
    def _create_simple_comparison(self, image: np.ndarray, pred_coords: np.ndarray, 
                                target_coords: np.ndarray, save_path: str, 
                                epoch: int, sample_idx: int):
        """간단한 예측/타겟 비교 이미지 생성"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 예측 결과
        img_pred = image.copy()
        for i, (x, y) in enumerate(pred_coords):
            color = (255, 0, 0) if i == 0 else (0, 0, 255)  # P1: 빨강, P2: 파랑
            cv2.circle(img_pred, (int(x), int(y)), 5, color, -1)
            cv2.putText(img_pred, f'P{i+1}', (int(x)+8, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        axes[0].imshow(img_pred)
        axes[0].set_title('Prediction')
        axes[0].axis('off')
        
        # 타겟 결과
        img_target = image.copy()
        for i, (x, y) in enumerate(target_coords):
            color = (255, 0, 0) if i == 0 else (0, 0, 255)  # P1: 빨강, P2: 파랑
            cv2.circle(img_target, (int(x), int(y)), 5, color, -1)
            cv2.putText(img_target, f'T{i+1}', (int(x)+8, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        axes[1].imshow(img_target)
        axes[1].set_title('Target')
        axes[1].axis('off')
        
        # 오차 정보 추가
        errors = np.linalg.norm(pred_coords - target_coords, axis=1)
        plt.suptitle(f'Epoch {epoch} - Sample {sample_idx}\n'
                    f'Error P1: {errors[0]:.2f}px, P2: {errors[1]:.2f}px, Avg: {errors.mean():.2f}px')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class VisualizationManager:
    """
    추론 및 평가용 시각화 매니저
    """
    
    def __init__(self, save_dir: str = "runs/visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 서브디렉토리 생성
        self.subdirs = {
            'heatmaps': os.path.join(save_dir, 'heatmaps'),
            'activations': os.path.join(save_dir, 'activations'), 
            'comparisons': os.path.join(save_dir, 'comparisons'),
            'attention': os.path.join(save_dir, 'attention'),
            'evolution': os.path.join(save_dir, 'evolution')
        }
        
        for subdir in self.subdirs.values():
            os.makedirs(subdir, exist_ok=True)
    
    def visualize_inference_results(self, 
                                  model: torch.nn.Module,
                                  image_path: str,
                                  result: Dict,
                                  save_prefix: str = "inference") -> Dict[str, str]:
        """
        추론 결과에 대한 종합 시각화
        
        Args:
            model: 학습된 모델
            image_path: 원본 이미지 경로
            result: 추론 결과 딕셔너리
            save_prefix: 저장 파일명 접두어
            
        Returns:
            저장된 파일 경로들
        """
        saved_files = {}
        
        try:
            # 원본 이미지 로드
            image_bgr = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # 1. 히트맵 시각화
            if 'confidence_maps' in result:
                heatmap_path = os.path.join(self.subdirs['heatmaps'], f"{save_prefix}_heatmap.png")
                
                # 추론 결과에서 폴리라인 마스크 추출 (있는 경우)
                polyline_mask = result.get('polyline_mask', None)
                
                HeatmapVisualizationTool.visualize_heatmap_prediction(
                    image=image_rgb,
                    pred_heatmaps=result['confidence_maps'],
                    pred_coords=result['keypoints'],
                    polyline_mask=polyline_mask,  # 폴리라인 마스크 추가
                    save_path=heatmap_path,
                    title=f"Inference Result - {os.path.basename(image_path)}"
                )
                saved_files['heatmap'] = heatmap_path
            
            # 2. CNN Activation 시각화
            if hasattr(model, 'net'):
                cnn_viz = CNNVisualizationTool(model.net)
                
                # 이미지 전처리 (모델 입력 형태로)
                img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]
                
                activation_dir = os.path.join(self.subdirs['activations'], save_prefix)
                os.makedirs(activation_dir, exist_ok=True)
                
                # 주요 레이어들만 시각화
                key_layers = ['e1.0.dw', 'e2.0.dw', 'e4.0.dw', 'd1.0.dw']
                cnn_viz.visualize_feature_maps(
                    input_tensor=img_tensor,
                    layer_names=key_layers,
                    max_channels=16,
                    save_dir=activation_dir,
                    save_prefix=save_prefix
                )
                saved_files['activations'] = activation_dir
            
            # 3. 간단한 결과 오버레이
            overlay_path = os.path.join(self.subdirs['comparisons'], f"{save_prefix}_overlay.png")
            self._create_result_overlay(image_rgb, result, overlay_path)
            saved_files['overlay'] = overlay_path
            
            print(f"✅ 추론 결과 시각화 완료: {len(saved_files)}개 파일 생성")
            return saved_files
            
        except Exception as e:
            print(f"❌ 추론 시각화 오류: {e}")
            return {}
    
    def _create_result_overlay(self, image: np.ndarray, result: Dict, save_path: str):
        """결과 오버레이 이미지 생성"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        ax.imshow(image)
        
        # 키포인트 표시
        keypoints = result['keypoints']
        for i, (x, y) in enumerate(keypoints):
            color = 'red' if i == 0 else 'blue'
            marker = 'x' if i == 0 else 'o'
            ax.plot(x, y, marker=marker, color=color, markersize=10, markeredgewidth=2)
            ax.text(x+5, y-5, f'P{i+1}', color=color, fontweight='bold', fontsize=12)
        
        # 연결선
        if len(keypoints) == 2:
            ax.plot([keypoints[0][0], keypoints[1][0]], 
                   [keypoints[0][1], keypoints[1][1]], 
                   'yellow', linewidth=2)
        
        # 정보 텍스트
        distance = np.linalg.norm(keypoints[1] - keypoints[0]) if len(keypoints) == 2 else 0
        ax.text(10, 30, f'Distance: {distance:.2f} px', 
               bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
               fontsize=12, fontweight='bold')
        
        ax.set_title(f"Detection Result - {os.path.basename(result.get('image_path', ''))}")
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_batch_summary(self, results: List[Dict], save_path: str = None) -> str:
        """배치 처리 결과 요약 시각화"""
        if save_path is None:
            save_path = os.path.join(self.subdirs['comparisons'], 'batch_summary.png')
        
        # 통계 계산
        distances = []
        errors_p1 = []
        errors_p2 = []
        
        for result in results:
            keypoints = result['keypoints']
            if len(keypoints) == 2:
                distance = np.linalg.norm(keypoints[1] - keypoints[0])
                distances.append(distance)
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 거리 분포
        axes[0, 0].hist(distances, bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Keypoint Distance Distribution')
        axes[0, 0].set_xlabel('Distance (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        
        # 거리 통계
        if distances:
            stats_text = f"""
            Count: {len(distances)}
            Mean: {np.mean(distances):.2f} px
            Std: {np.std(distances):.2f} px
            Min: {np.min(distances):.2f} px
            Max: {np.max(distances):.2f} px
            """
            axes[0, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
            axes[0, 1].set_title('Distance Statistics')
            axes[0, 1].axis('off')
        
        # 처리 성공률
        success_rate = len(results) / len(results) * 100 if results else 0
        axes[1, 0].bar(['Processed'], [len(results)], color='green', alpha=0.7)
        axes[1, 0].set_title(f'Processing Summary (Success: {success_rate:.1f}%)')
        axes[1, 0].set_ylabel('Number of Images')
        
        # 샘플 이미지들 (최대 6개)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Sample Results')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 배치 요약 저장: {save_path}")
        return save_path
