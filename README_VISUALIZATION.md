# CNN Activation 시각화 및 히트맵 시각화 가이드

이 문서는 새로 구현된 CNN Activation 시각화와 히트맵 시각화 기능의 사용법을 설명합니다.

## 🎨 구현된 시각화 기능

### 1. CNN Activation 시각화
- **Feature Map 시각화**: 각 CNN 레이어의 feature map들을 채널별로 시각화
- **Activation 통계**: 레이어별 activation의 평균, 표준편차, sparsity, saturation 분석
- **다중 레이어 지원**: encoder, decoder의 주요 레이어들 자동 선택

### 2. 히트맵 시각화 
- **예측 vs 타겟 비교**: 모델이 예측한 히트맵과 실제 타겟 히트맵 비교
- **오버레이 시각화**: 원본 이미지 위에 히트맵 반투명 오버레이
- **키포인트 표시**: 예측된 좌표와 실제 좌표 동시 표시
- **컬러맵 적용**: jet 컬러맵으로 히트맵 강도 시각화

### 3. 학습 중 자동 시각화
- **에포크별 자동 저장**: 지정된 에포크마다 자동으로 시각화 저장
- **샘플 제한**: 메모리 효율성을 위한 샘플 수 제한
- **구조화된 저장**: 히트맵, activation, 비교 이미지를 별도 폴더에 정리

## 📁 현재 히트맵 구현 분석

코드베이스의 현재 히트맵 구현은 다음과 같습니다:

### 히트맵 생성 방식
1. **키포인트 히트맵** (`make_heatmaps_torch`):
   - 가우시안 분포 기반 히트맵 생성
   - 표준편차 σ=3.0px로 조정 가능
   - 출력: `[B, 2, H, W]` (2개 키포인트)

2. **폴리라인 히트맵** (`make_polyline_heatmaps_from_mask`):
   - 거리 변환(distance transform) 기반
   - 폴리라인을 따라 가우시안 확산
   - 키포인트 제약조건으로 활용

3. **결합 히트맵** (`make_combined_heatmaps_from_mask`):
   - 키포인트 + 폴리라인 히트맵 가중 결합
   - `polyline_weight=0.3`으로 가중치 조절

### 좌표 추출 방식
- **softargmax_2d**: 온도 파라미터(temperature=0.02)를 사용한 soft-argmax
- **argmax**: 간단한 최댓값 위치 추출 (백업용)

## 🚀 사용 방법

### 1. 학습 중 자동 시각화

학습 시 자동으로 시각화가 활성화됩니다:

```bash
python src/train_img.py \
    --train_img data/train/img \
    --train_pkl data/train/pkl \
    --ann_csv data/train/annotations.csv \
    --max_epochs 50 \
    --batch_size 8
```

**시각화 설정:**
- **저장 주기**: 10 에포크마다 자동 저장
- **샘플 수**: 에포크당 최대 2개 샘플
- **저장 위치**: `runs/ckpts/training_visualizations/`

**생성되는 파일들:**
```
runs/ckpts/training_visualizations/
├── heatmaps/              # 히트맵 비교 이미지
│   ├── epoch_010_sample_0_heatmap.png
│   └── epoch_010_sample_1_heatmap.png
├── activations/           # CNN activation 시각화
│   ├── epoch_010_sample_0/
│   │   ├── epoch_10_sample_0_net_e1_0_dw.png
│   │   └── epoch_10_sample_0_net_e2_0_dw.png
│   └── epoch_010_sample_1/
└── comparison/            # 간단한 예측/타겟 비교
    ├── epoch_010_sample_0_comparison.png
    └── epoch_010_sample_1_comparison.png
```

### 2. 추론 시 고급 시각화

추론 시 `--enable_visualization` 플래그로 활성화:

```bash
# 단일 이미지
python inference_complete.py \
    --input data/Left/191412/img/191412_000.png \
    --output runs/inference_viz \
    --enable_visualization

# 폴더 배치 처리 (고급 시각화 포함)
python inference_complete.py \
    --input data/Left/191412/img \
    --output runs/inference_viz \
    --max_images 5 \
    --enable_visualization
```

**생성되는 고급 시각화:**
```
runs/inference_viz/
├── visualizations/
│   ├── heatmaps/          # 히트맵 분석
│   ├── activations/       # CNN feature map
│   ├── comparisons/       # 결과 오버레이
│   └── batch_summary.png  # 배치 처리 요약
├── 191412_000_overlay.png # 기본 오버레이
└── 191412_000_pred.csv    # 좌표 결과
```

### 3. 수동 시각화 테스트

시각화 기능을 개별적으로 테스트:

```bash
python test_visualization.py
```

**테스트 항목:**
- CNN Activation 시각화
- 히트맵 시각화  
- 통합 시각화 매니저
- 추론 + 시각화 통합

## 🛠️ 고급 사용법

### CNN Activation 커스터마이징

```python
from src.utils.visualization import CNNVisualizationTool

# 모델과 시각화 도구 초기화
cnn_viz = CNNVisualizationTool(model.net, device="cuda")

# 특정 레이어들만 시각화
target_layers = ['net.e1.0.dw', 'net.e4.0.dw', 'net.d1.0.dw']
cnn_viz.visualize_feature_maps(
    input_tensor=image_tensor,
    layer_names=target_layers,
    max_channels=16,  # 최대 채널 수
    save_dir="custom_viz",
    save_prefix="custom"
)

# Activation 통계 분석
stats = cnn_viz.visualize_activation_statistics(
    input_tensor=image_tensor,
    layer_names=target_layers
)
```

### 히트맵 시각화 커스터마이징

```python
from src.utils.visualization import HeatmapVisualizationTool

# 커스텀 히트맵 시각화
HeatmapVisualizationTool.visualize_heatmap_prediction(
    image=original_image,
    pred_heatmaps=prediction_heatmaps,
    target_heatmaps=ground_truth_heatmaps,  # 옵션
    pred_coords=predicted_coordinates,
    target_coords=ground_truth_coordinates, # 옵션
    save_path="custom_heatmap.png",
    title="Custom Heatmap Analysis"
)
```

### 학습 시각화 설정 변경

`src/train_img.py`의 콜백 설정 수정:

```python
# 시각화 콜백 커스터마이징
viz_cb = TrainingVisualizerCallback(
    save_dir="custom_training_viz",
    save_every_n_epochs=5,      # 5 에포크마다
    max_samples_per_epoch=4,    # 에포크당 4개 샘플
    visualize_activations=True,
    visualize_heatmaps=True,
    activation_layers=[         # 커스텀 레이어 선택
        'net.e1.0.dw', 'net.e2.0.dw', 
        'net.e3.0.dw', 'net.d1.0.dw'
    ]
)
```

## 📊 시각화 결과 해석

### CNN Activation Maps
- **초기 레이어**: 에지, 텍스처 등 저수준 특징
- **중간 레이어**: 복합적인 패턴과 형태
- **후반 레이어**: 태스크별 고수준 특징 (키포인트 위치 등)
- **Sparsity**: 높을수록 sparse한 representation
- **Saturation**: 높을수록 activation이 포화 상태

### 히트맵 분석
- **밝은 영역**: 높은 confidence 영역
- **색상 강도**: jet 컬러맵 (파랑→빨강, 낮음→높음)
- **예측 vs 타겟**: 모델의 학습 진행 상황 파악
- **공간적 분포**: 키포인트 주변의 uncertainty

## 🔧 의존성 패키지

새로 추가된 시각화 관련 패키지:

```txt
matplotlib      # 기본 플롯팅
seaborn        # 통계 시각화
scipy          # 거리 변환 등
Pillow         # 이미지 처리
imageio        # GIF 생성 (옵션)
```

설치:
```bash
pip install matplotlib seaborn scipy Pillow imageio
```

## 💡 팁과 주의사항

### 성능 최적화
- **배치 크기**: 시각화는 첫 번째 샘플만 사용하여 메모리 절약
- **채널 제한**: `max_channels` 파라미터로 시각화 채널 수 제한
- **저장 주기**: `save_every_n_epochs`로 저장 빈도 조절

### 메모리 관리
- 시각화는 `torch.no_grad()` 컨텍스트에서 실행
- 대용량 이미지는 자동으로 리사이즈
- 훅(hook)은 사용 후 자동으로 제거

### 문제 해결
- **ImportError**: `pip install -r requirements.txt`로 패키지 설치
- **CUDA 메모리 부족**: `max_channels` 및 `max_samples_per_epoch` 감소
- **저장 공간**: 정기적으로 오래된 시각화 파일 정리

## 🔄 향후 확장 가능성

현재 구현을 기반으로 다음과 같은 확장이 가능합니다:

1. **Grad-CAM**: 그래디언트 기반 attention 시각화
2. **Layer-wise Learning Rate**: 레이어별 학습 진행 상황 시각화  
3. **실시간 스트리밍**: 학습 중 실시간 시각화 대시보드
4. **3D 시각화**: 키포인트 연결성과 3차원 표현
5. **비교 분석**: 다른 모델/설정 간의 시각적 비교

이러한 확장은 현재 구축된 시각화 프레임워크를 기반으로 쉽게 추가할 수 있습니다.
