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
    ap.add_argument("--sigma", type=float, default=6.0, help="키포인트 가우시안 표준편차(px)")
    ap.add_argument("--sigma_polyline", type=float, default=3.0, help="폴리라인 가우시안 표준편차(px)")
    ap.add_argument("--polyline_weight", type=float, default=0.3, help="폴리라인 히트맵 가중치 (0~1)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-4)

    ap.add_argument("--val_ratio", type=float, default=0.2, help="train/val 분할 비율")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--devices", default="auto")
    ap.add_argument("--precision", default="16-mixed")

    ap.add_argument("--pck_thresh_px", type=float, default=5.0, help="PCK 임계값(px)")
    ap.add_argument("--save_dir", default="runs/ckpts")

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
            f"p1={self._fmt(m.get('train_loss_p1'))} "
            f"p2={self._fmt(m.get('train_loss_p2'))} "
            f"l_ratio={self._fmt(m.get('train_loss_ratio'))} "
            f"mae={self._fmt(m.get('train_mae_px'))} "
            f"mae_p1={self._fmt(m.get('train_mae_p1'))} "
            f"mae_p2={self._fmt(m.get('train_mae_p2'))} "
            f"m_ratio={self._fmt(m.get('train_mae_ratio'))} | "
            f"val_loss={self._fmt(m.get('val_loss'))} "
            f"v_mae={self._fmt(m.get('val_mae_px'))} "
            f"v_mae_p1={self._fmt(m.get('val_mae_p1'))} "
            f"v_mae_p2={self._fmt(m.get('val_mae_p2'))} "
            f"v_m_ratio={self._fmt(m.get('val_mae_ratio'))}"
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

    return cbs


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.save_dir, exist_ok=True)

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
        pck_thresh_px=args.pck_thresh_px
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
