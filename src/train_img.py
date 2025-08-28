# src/train_img.py
import os
import argparse
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from src.data.dataset_img import BeadImageKeypointDataset
from src.models import LitKeypoint2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_img", required=True, help="이미지 디렉토리 (train 전체)")
    ap.add_argument("--train_pkl", required=True, help="동일 베이스명의 PKL 디렉토리")
    ap.add_argument("--ann_csv", default=None, help="정답 CSV (image,point1,point2)")

    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--sigma", type=float, default=3.0, help="키포인트 가우시안 표준편차(px) - 새로운 권장값")
    ap.add_argument("--sigma_polyline", type=float, default=3.0, help="폴리라인 가우시안 표준편차(px)")
    ap.add_argument("--polyline_weight", type=float, default=0.3, help="폴리라인 히트맵 가중치 (0~1)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-4)
    
    # 새로운 손실 시스템 파라미터들
    ap.add_argument("--lambda_heatmap", type=float, default=1.0, help="히트맵 MSE 손실 가중치")
    ap.add_argument("--lambda_coord", type=float, default=0.2, help="좌표 Huber 손실 가중치")
    ap.add_argument("--lambda_separation", type=float, default=0.0, help="분리 힌지 손실 가중치 (0=비활성화)")
    ap.add_argument("--huber_delta", type=float, default=2.0, help="Huber 손실 전환점 (픽셀)")
    ap.add_argument("--min_separation", type=float, default=5.0, help="최소 분리 거리 (픽셀)")
    ap.add_argument("--temperature", type=float, default=0.02, help="soft-argmax 온도 파라미터")

    ap.add_argument("--val_ratio", type=float, default=0.2, help="train/val 분할 비율")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--devices", default="auto")
    ap.add_argument("--precision", default="16-mixed")

    ap.add_argument("--pck_thresh_px", type=float, default=5.0, help="PCK 임계값(px)")
    ap.add_argument("--save_dir", default="runs/ckpts", 
                    help="베이스 저장 디렉토리 (자동으로 version_XX_timestamp 폴더 생성)")
    ap.add_argument("--disable_versioning", action="store_true",
                    help="버전 관리 비활성화 (권한 문제 해결용)")


    # 로그/프로그레스 관련
    ap.add_argument("--progress", choices=["rich", "tqdm", "none"], default="none",
                    help="진행 표시 방식 (코랩에서는 none 권장)")
    ap.add_argument("--persist_console", action="store_true", default=True,
                    help="에폭마다 콘솔에 고정 요약 로그를 한 줄씩 남김")
    ap.add_argument("--log_every_n_steps", type=int, default=50,
                    help="step 로그 갱신 빈도(작을수록 잦게 출력, 코랩에서는 큰 값 권장)")
    return ap.parse_args()


def build_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    ds = BeadImageKeypointDataset(
        img_dir=args.train_img,
        pkl_dir=args.train_pkl,
        size=args.size,
        annotation_csv=args.ann_csv,
        auto_blue_min=70,
        auto_diff_min=30,
    )
    val_len = max(1, int(len(ds) * args.val_ratio))
    train_len = len(ds) - val_len
    gen = torch.Generator().manual_seed(args.seed)
    tr_ds, va_ds = random_split(ds, [train_len, val_len], generator=gen)

    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, persistent_workers=True, 
                    drop_last=True)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    return tr, va


class PersistConsole(pl.Callback):
    """매 에폭 끝에 '사라지지 않는' 요약 로그를 한 줄로 출력."""
    def _fmt(self, v):
        try:
            if v is None:
                return "NA"
            return f"{float(v):.3f}"
        except Exception:
            return str(v)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        m = trainer.callback_metrics  # train/val의 on_epoch=True 지표들이 들어옴
        epoch = trainer.current_epoch
        best = getattr(getattr(trainer, "checkpoint_callback", None), "best_model_score", None)

        line = (
            f"[E{epoch:03d}] "
            f"loss={self._fmt(m.get('train_loss'))} "
            f"hm={self._fmt(m.get('train_heatmap_loss'))} "
            f"coord={self._fmt(m.get('train_coord_loss'))} "
            f"sep={self._fmt(m.get('train_separation_loss'))} "
            f"mae={self._fmt(m.get('train_mae_px'))} "
            f"mae_p1={self._fmt(m.get('train_mae_p1'))} "
            f"mae_p2={self._fmt(m.get('train_mae_p2'))} "
            f"m_ratio={self._fmt(m.get('train_mae_ratio'))} "
            f"dist={self._fmt(m.get('train_separation'))} | "
            f"val_loss={self._fmt(m.get('val_loss'))} "
            f"v_hm={self._fmt(m.get('val_heatmap_loss'))} "
            f"v_coord={self._fmt(m.get('val_coord_loss'))} "
            f"v_mae={self._fmt(m.get('val_mae_px'))} "
            f"v_mae_p1={self._fmt(m.get('val_mae_p1'))} "
            f"v_mae_p2={self._fmt(m.get('val_mae_p2'))} "
            f"v_dist={self._fmt(m.get('val_separation'))}"
        )
        if best is not None:
            line += f" | best_val_mae={self._fmt(best)}"

        # rank_zero_info는 멀티 GPU에서도 중복 없이 한 번만 출력
        pl.utilities.rank_zero_info(line)


def make_callbacks(args) -> List[pl.Callback]:
    cbs: List[pl.Callback] = []

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=args.save_dir,
        filename="kp2-{epoch:03d}-{val_mae_px:.2f}",
        save_top_k=3,
        monitor="val_mae_px",
        mode="min"
    )
    cbs.append(ckpt_cb)

    # 진행 표시 방식 선택
    if args.progress == "rich":
        try:
            from pytorch_lightning.callbacks import RichProgressBar
            cbs.append(RichProgressBar())
        except Exception:
            pass
    elif args.progress == "tqdm":
        try:
            from pytorch_lightning.callbacks import TQDMProgressBar
            # refresh_rate를 크게 설정해서 코랩에서 렉 방지
            cbs.append(TQDMProgressBar(refresh_rate=20))
        except Exception:
            pass
    else:
        # "none" → 진행 표시 비활성
        pass

    # 에폭별 고정 로그
    if args.persist_console:
        cbs.append(PersistConsole())
    
    # 무조건 시각화 콜백 추가 (내장 도구로 완전 해결)
    from src.utils.training_visualizer import TrainingVisualizerCallback
    TRAINING_VISUALIZER_AVAILABLE = True
    print("✅ 내장 시각화 콜백 로드 성공")
        
    # 시각화 콜백 추가 (내장 도구로 안전하게)
    viz_dir = os.path.join(args.save_dir, "training_visualizations")
    
    viz_cb = TrainingVisualizerCallback(
        save_dir=viz_dir,
        save_every_n_epochs=10,  # 10 에포크마다
        max_samples_per_epoch=2,  # 에포크당 2개 샘플
        visualize_activations=True,
        visualize_heatmaps=True
    )
    cbs.append(viz_cb)
    print(f"✅ 학습 중 시각화 콜백 추가됨 (10 에포크마다, 내장 도구 사용)")

    return cbs


def get_versioned_save_dir(base_dir: str) -> str:
    """매우 간단한 타임스탬프 폴더 생성"""
    # 현재 시간을 초 단위로 변환하여 고유한 폴더명 생성
    import time
    timestamp = str(int(time.time() * 1000))[-10:]  # 밀리초 포함으로 더 고유하게
    return base_dir + "_" + timestamp

def save_training_config(args, save_dir: str):
    """학습 설정을 간단한 텍스트 파일로 저장 (안전한 버전)"""
    try:
        import time
        config_text = f"""Training Configuration
=====================
Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
Save Directory: {save_dir}

Data Config:
- train_img: {args.train_img}
- train_pkl: {args.train_pkl}
- ann_csv: {args.ann_csv}
- size: {args.size}
- batch_size: {args.batch_size}
- max_epochs: {args.max_epochs}
- lr: {args.lr}

Model Config:
- sigma: {args.sigma}
- sigma_polyline: {args.sigma_polyline}
- polyline_weight: {args.polyline_weight}

Loss Config:
- lambda_heatmap: {args.lambda_heatmap}
- lambda_coord: {args.lambda_coord}
- lambda_separation: {args.lambda_separation}
- temperature: {args.temperature}
"""
        
        config_path = os.path.join(save_dir, "training_config.txt")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_text)
        
        print(f"💾 학습 설정 저장: {config_path}")
    except Exception as e:
        print(f"⚠️ 설정 파일 저장 실패 (무시하고 계속): {e}")

def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    
    # 무조건 타임스탬프 폴더 생성
    if args.disable_versioning:
        print("🔧 버전 관리 비활성화됨")
        final_save_dir = args.save_dir
    else:
        print("📁 타임스탬프 폴더 생성...")
        final_save_dir = get_versioned_save_dir(args.save_dir)
        print(f"📁 새로운 학습 폴더: {final_save_dir}")
        args.save_dir = final_save_dir
    
    # 디렉토리 생성
    os.makedirs(final_save_dir, exist_ok=True)
    
    # 설정 파일 저장 (실패해도 무시)
    try:
        save_training_config(args, final_save_dir)
    except:
        pass  # 설정 파일 저장 실패는 무시

    tr, va = build_dataloaders(args)
    
    # log_every_n_steps를 훈련 배치 수에 맞게 자동 조정
    train_batches = len(tr)
    effective_log_steps = min(args.log_every_n_steps, max(1, train_batches // 4))
    
    model = LitKeypoint2(
        img_size=args.size,
        sigma=args.sigma,
        sigma_polyline=args.sigma_polyline,
        polyline_weight=args.polyline_weight,
        lr=args.lr,
        pck_thresh_px=args.pck_thresh_px,
        # 새로운 손실 시스템 파라미터들
        lambda_heatmap=args.lambda_heatmap,
        lambda_coord=args.lambda_coord,
        lambda_separation=args.lambda_separation,
        huber_delta=args.huber_delta,
        min_separation=args.min_separation,
        temperature=args.temperature
    )

    # 로거 디렉토리 미리 생성
    log_dir = os.path.join(args.save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = pl.loggers.CSVLogger(save_dir=args.save_dir, name="logs")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=make_callbacks(args),
        log_every_n_steps=effective_log_steps,  # 자동 조정된 로그 빈도
        enable_progress_bar=(args.progress != "none"),
        enable_model_summary=True,
    )
    
    print(f"훈련 배치 수: {train_batches}, 로그 스텝 간격: {effective_log_steps}")
    trainer.fit(model, tr, va)


if __name__ == "__main__":
    main()
