# src/utils/visualization.py
"""
CNN Activation 시각화와 히트맵 시각화 도구
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from typing import Dict, List, Tuple, Optional, Union
try:
    import seaborn as sns
except ImportError:
    sns = None


class ActivationHook:
    """CNN 레이어의 activation을 캡처하는 훅"""
    
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def hook_fn(self, name: str):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def register_hooks(self, model: torch.nn.Module, layer_names: List[str] = None):
        """지정된 레이어에 훅 등록"""
        if layer_names is None:
            # 기본적으로 모든 Conv2d 레이어에 훅 등록
            layer_names = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    layer_names.append(name)
        
        registered_hooks = []
        model_modules = dict(model.named_modules())
        
        for name in layer_names:
            try:
                if name in model_modules:
                    module = model_modules[name]
                    hook = module.register_forward_hook(self.hook_fn(name))
                    self.hooks.append(hook)
                    registered_hooks.append(name)
                else:
                    print(f"⚠️ 레이어 '{name}'을 찾을 수 없음")
            except Exception as e:
                print(f"⚠️ 레이어 '{name}' 훅 등록 실패: {e}")
        
        print(f"✅ {len(registered_hooks)}개 레이어에 activation 훅 등록: {registered_hooks}")
        return registered_hooks
    
    def clear_hooks(self):
        """모든 훅 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """캡처된 activation 반환"""
        return self.activations


class CNNVisualizationTool:
    """CNN Activation 시각화 도구"""
    
    def __init__(self, model: torch.nn.Module, device: str = "auto"):
        self.model = model
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
        self.model.to(self.device)
        self.hook = ActivationHook()
        
    def visualize_feature_maps(self, 
                              input_tensor: torch.Tensor, 
                              layer_names: List[str] = None,
                              max_channels: int = 16,
                              save_dir: str = "runs/visualizations",
                              save_prefix: str = "feature_maps") -> Dict[str, np.ndarray]:
        """
        특정 레이어의 feature map 시각화
        
        Args:
            input_tensor: [1, 3, H, W] 입력 텐서
            layer_names: 시각화할 레이어 이름들 (None이면 자동 선택)
            max_channels: 시각화할 최대 채널 수
            save_dir: 저장 디렉토리
            save_prefix: 파일명 접두어
        
        Returns:
            시각화된 feature map들
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 훅 등록
        registered_layers = self.hook.register_hooks(self.model, layer_names)
        
        try:
            self.model.eval()
            with torch.no_grad():
                # 순전파 실행하여 activation 캡처
                _ = self.model(input_tensor.to(self.device))
            
            activations = self.hook.get_activations()
            visualization_results = {}
            
            for layer_name, activation in activations.items():
                print(f"🔍 레이어 '{layer_name}' 시각화 중... Shape: {activation.shape}")
                
                # [B, C, H, W] -> [C, H, W] (첫 번째 배치만)
                feature_maps = activation[0]
                
                # 채널 수 제한
                num_channels = min(feature_maps.shape[0], max_channels)
                
                # 간단한 방식: 각 채널을 개별 이미지로 저장
                layer_clean_name = layer_name.replace('.', '_')
                
                for i in range(num_channels):
                    try:
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
                        
                        # 저장
                        save_path = os.path.join(save_dir, f"{save_prefix}_{layer_clean_name}_ch{i:02d}.png")
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        if i == 0:  # 첫 번째 채널만 결과에 포함
                            visualization_results[layer_name] = feature_maps.cpu().numpy()
                            print(f"💾 저장 완료: {layer_clean_name} ({num_channels} 채널)")
                        
                    except Exception as e:
                        print(f"⚠️ {layer_name} 채널 {i} 시각화 실패: {e}")
                        continue
            
            return visualization_results
            
        finally:
            self.hook.clear_hooks()
    
    def visualize_activation_statistics(self, 
                                      input_tensor: torch.Tensor,
                                      layer_names: List[str] = None,
                                      save_dir: str = "runs/visualizations",
                                      save_prefix: str = "activation_stats") -> Dict[str, Dict]:
        """
        레이어별 activation 통계 시각화
        
        Args:
            input_tensor: [1, 3, H, W] 입력 텐서
            layer_names: 분석할 레이어 이름들
            save_dir: 저장 디렉토리  
            save_prefix: 파일명 접두어
        
        Returns:
            레이어별 통계 정보
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 훅 등록
        registered_layers = self.hook.register_hooks(self.model, layer_names)
        
        try:
            self.model.eval()
            with torch.no_grad():
                # 순전파 실행
                _ = self.model(input_tensor.to(self.device))
            
            activations = self.hook.get_activations()
            stats_results = {}
            
            # 통계 수집
            layer_stats = []
            for layer_name, activation in activations.items():
                act_np = activation.cpu().numpy()
                
                stats = {
                    'layer': layer_name,
                    'shape': act_np.shape,
                    'mean': np.mean(act_np),
                    'std': np.std(act_np),
                    'min': np.min(act_np),
                    'max': np.max(act_np),
                    'sparsity': np.mean(act_np == 0) * 100,  # 0인 비율 (%)
                    'saturation': np.mean(act_np > 0.9) * 100 if np.max(act_np) > 0 else 0  # 포화 비율 (%)
                }
                layer_stats.append(stats)
                stats_results[layer_name] = stats
                
                print(f"📊 {layer_name}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, "
                      f"Sparsity={stats['sparsity']:.1f}%, Saturation={stats['saturation']:.1f}%")
            
            # 시각화
            if layer_stats:
                # 1. Mean/Std 그래프
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                layers = [s['layer'] for s in layer_stats]
                means = [s['mean'] for s in layer_stats]
                stds = [s['std'] for s in layer_stats]
                sparsities = [s['sparsity'] for s in layer_stats]
                saturations = [s['saturation'] for s in layer_stats]
                
                # Mean
                axes[0,0].bar(range(len(layers)), means)
                axes[0,0].set_title('Activation Means by Layer')
                axes[0,0].set_ylabel('Mean Activation')
                axes[0,0].set_xticks(range(len(layers)))
                axes[0,0].set_xticklabels([l.split('.')[-1] for l in layers], rotation=45)
                
                # Std
                axes[0,1].bar(range(len(layers)), stds)
                axes[0,1].set_title('Activation Standard Deviations by Layer')
                axes[0,1].set_ylabel('Std Activation')
                axes[0,1].set_xticks(range(len(layers)))
                axes[0,1].set_xticklabels([l.split('.')[-1] for l in layers], rotation=45)
                
                # Sparsity
                axes[1,0].bar(range(len(layers)), sparsities)
                axes[1,0].set_title('Activation Sparsity by Layer (%)')
                axes[1,0].set_ylabel('Sparsity (%)')
                axes[1,0].set_xticks(range(len(layers)))
                axes[1,0].set_xticklabels([l.split('.')[-1] for l in layers], rotation=45)
                
                # Saturation
                axes[1,1].bar(range(len(layers)), saturations)
                axes[1,1].set_title('Activation Saturation by Layer (%)')
                axes[1,1].set_ylabel('Saturation (%)')
                axes[1,1].set_xticks(range(len(layers)))
                axes[1,1].set_xticklabels([l.split('.')[-1] for l in layers], rotation=45)
                
                plt.tight_layout()
                
                # 저장
                save_path = os.path.join(save_dir, f"{save_prefix}_statistics.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"💾 통계 그래프 저장: {save_path}")
            
            return stats_results
            
        finally:
            self.hook.clear_hooks()


class HeatmapVisualizationTool:
    """히트맵 시각화 도구"""
    
    @staticmethod
    def visualize_heatmap_prediction(image: np.ndarray,
                                   pred_heatmaps: np.ndarray,
                                   target_heatmaps: np.ndarray = None,
                                   pred_coords: np.ndarray = None,
                                   target_coords: np.ndarray = None,
                                   polyline_mask: np.ndarray = None,
                                   save_path: str = None,
                                   title: str = "Heatmap Visualization") -> np.ndarray:
        """
        예측 히트맵과 타겟 히트맵 비교 시각화
        
        Args:
            image: [H, W, 3] 원본 이미지 (RGB)
            pred_heatmaps: [2, H, W] 예측 히트맵
            target_heatmaps: [2, H, W] 타겟 히트맵 (옵션)
            pred_coords: [2, 2] 예측 좌표 (옵션)
            target_coords: [2, 2] 타겟 좌표 (옵션)
            polyline_mask: [H, W] 폴리라인 제약조건 마스크 (옵션)
            save_path: 저장 경로 (옵션)
            title: 그래프 제목
        
        Returns:
            시각화 결과 이미지
        """
        # 서브플롯 개수 계산 (폴리라인 마스크 고려)
        has_polyline = polyline_mask is not None
        n_cols = 4 if (target_heatmaps is not None and has_polyline) else 3 if (target_heatmaps is not None or has_polyline) else 2
        n_rows = 2  # Point 1, Point 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 4))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # 원본 이미지 크기로 히트맵 리사이즈
        img_h, img_w = image.shape[:2]
        pred_heatmaps_resized = np.array([
            cv2.resize(pred_heatmaps[i], (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            for i in range(2)
        ])
        
        if target_heatmaps is not None:
            target_heatmaps_resized = np.array([
                cv2.resize(target_heatmaps[i], (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                for i in range(2)
            ])
        
        # 폴리라인 마스크 리사이즈 (있는 경우)
        if has_polyline:
            polyline_mask_resized = cv2.resize(polyline_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        for point_idx in range(2):
            row = point_idx
            
            col_idx = 0
            
            # 1. 원본 이미지 + 예측 히트맵 오버레이
            ax = axes[row, col_idx]
            ax.imshow(image)
            
            # 폴리라인 마스크 먼저 표시 (있는 경우)
            if has_polyline:
                # 폴리라인 영역을 초록색으로 반투명 표시
                mask_colored = np.zeros((*polyline_mask_resized.shape, 4))
                mask_colored[..., 1] = polyline_mask_resized  # 초록색 채널
                mask_colored[..., 3] = polyline_mask_resized * 0.3  # 투명도
                ax.imshow(mask_colored)
            
            # 히트맵 오버레이 (반투명)
            hm = pred_heatmaps_resized[point_idx]
            hm_normalized = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
            
            # 컬러맵 적용 (jet 사용)
            cmap = plt.cm.jet
            hm_colored = cmap(hm_normalized)
            hm_colored[..., 3] = hm_normalized * 0.6  # 투명도 설정
            
            ax.imshow(hm_colored)
            
            # 예측 좌표 표시
            if pred_coords is not None:
                x, y = pred_coords[point_idx]
                ax.plot(x, y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
                ax.text(x+5, y-5, f'P{point_idx+1}', color='white', fontweight='bold')
            
            # 타겟 좌표 표시
            if target_coords is not None:
                x, y = target_coords[point_idx]
                ax.plot(x, y, 'bx', markersize=10, markeredgewidth=3)
            
            title = f'Point {point_idx+1} - Prediction Overlay'
            if has_polyline:
                title += ' + Polyline Mask'
            ax.set_title(title)
            ax.axis('off')
            col_idx += 1
            
            # 2. 예측 히트맵만
            ax = axes[row, col_idx]
            im = ax.imshow(hm_normalized, cmap='jet', interpolation='bilinear')
            ax.set_title(f'Point {point_idx+1} - Prediction Heatmap')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            col_idx += 1
            
            # 3. 폴리라인 마스크만 (있는 경우)
            if has_polyline:
                ax = axes[row, col_idx]
                # 폴리라인 마스크를 흑백으로 표시
                ax.imshow(polyline_mask_resized, cmap='gray', interpolation='nearest')
                ax.set_title(f'Point {point_idx+1} - Polyline Constraint')
                ax.axis('off')
                col_idx += 1
            
            # 4. 타겟 히트맵 (있는 경우)
            if target_heatmaps is not None:
                ax = axes[row, col_idx]
                target_hm = target_heatmaps_resized[point_idx]
                target_hm_norm = (target_hm - target_hm.min()) / (target_hm.max() - target_hm.min() + 1e-8)
                im = ax.imshow(target_hm_norm, cmap='jet', interpolation='bilinear')
                ax.set_title(f'Point {point_idx+1} - Target Heatmap')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                col_idx += 1
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 히트맵 시각화 저장: {save_path}")
        
        plt.close()
        
        # 간단한 더미 배열 반환 (학습 중 시각화에서는 필요 없음)
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    @staticmethod
    def create_heatmap_evolution_gif(heatmap_sequence: List[np.ndarray],
                                   image: np.ndarray,
                                   save_path: str,
                                   fps: int = 5,
                                   point_idx: int = 0):
        """
        학습 중 히트맵 변화를 GIF로 생성 (옵션 기능)
        
        Args:
            heatmap_sequence: 히트맵 시퀀스 리스트
            image: 원본 이미지
            save_path: GIF 저장 경로
            fps: 프레임 레이트
            point_idx: 시각화할 포인트 인덱스 (0 또는 1)
        """
        try:
            from PIL import Image
            import imageio
        except ImportError:
            print("❌ PIL 또는 imageio 패키지가 필요합니다: pip install Pillow imageio")
            return
        
        print("⚠️ GIF 생성 기능은 현재 비활성화되어 있습니다 (matplotlib 호환성 이슈)")
        return
    
    @staticmethod
    def visualize_attention_maps(attention_weights: np.ndarray,
                               image: np.ndarray,
                               save_path: str = None,
                               title: str = "Attention Maps") -> np.ndarray:
        """
        어텐션 맵 시각화 (Grad-CAM 스타일)
        
        Args:
            attention_weights: [H, W] 어텐션 가중치
            image: [H, W, 3] 원본 이미지
            save_path: 저장 경로
            title: 제목
        
        Returns:
            시각화 결과
        """
        img_h, img_w = image.shape[:2]
        
        # 어텐션 맵을 이미지 크기로 리사이즈
        attention_resized = cv2.resize(attention_weights, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        attention_norm = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
        
        # 시각화
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 원본 이미지
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 2. 어텐션 맵
        im = axes[1].imshow(attention_norm, cmap='jet', interpolation='bilinear')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 3. 오버레이
        axes[2].imshow(image)
        
        # 어텐션 오버레이
        cmap = plt.cm.jet
        attention_colored = cmap(attention_norm)
        attention_colored[..., 3] = attention_norm * 0.6
        axes[2].imshow(attention_colored)
        
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 어텐션 맵 저장: {save_path}")
        
        plt.close()
        
        # 간단한 더미 배열 반환 (학습 중 시각화에서는 필요 없음)
        return np.zeros((100, 100, 3), dtype=np.uint8)
