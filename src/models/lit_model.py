# src/models/lit_model.py
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .img_backbone import DWUNetTiny  # 기존 경량 UNet (약 0.26MB)
from ..losses.heatmap_utils import (
    make_heatmaps_torch, make_combined_heatmaps_from_mask, 
    heatmaps_to_coords_argmax, softargmax_2d,
    NewKeypointLoss
)


class LitKeypoint2(pl.LightningModule):
    def __init__(self, img_size: int = 512, sigma: float = 3.0, lr: float = 5e-4,
                 pck_thresh_px: float = 5.0, sigma_polyline: float = 3.0, polyline_weight: float = 0.3,
                 # 새로운 손실 시스템 파라미터
                 lambda_heatmap: float = 1.0, lambda_coord: float = 0.2, lambda_separation: float = 0.0,
                 huber_delta: float = 2.0, min_separation: float = 5.0, temperature: float = 0.02):
        super().__init__()
        self.save_hyperparameters()
        self.net = DWUNetTiny(in_ch=3, base=64, out_ch=2) # 더 큰 모델 (64 base)
        # 정규화를 위한 드롭아웃 추가
        self.dropout = torch.nn.Dropout2d(p=0.10)
        self.img_size = img_size
        self.sigma = sigma  # 히트맵 가우시안 표준편차 (권장: 3px)
        self.sigma_polyline = sigma_polyline  # 폴리라인용 가우시안 표준편차
        self.polyline_weight = polyline_weight  # 폴리라인 히트맵 가중치
        self.lr = lr
        self.pck_thresh_px = pck_thresh_px
        
        # 새로운 손실 시스템
        self.loss_fn = NewKeypointLoss(
            lambda_heatmap=lambda_heatmap,
            lambda_coord=lambda_coord,
            lambda_separation=lambda_separation,
            huber_delta=huber_delta,
            min_separation=min_separation,
            temperature=temperature
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,2,H,W] 로짓 반환
        logit = self.net(x)
        # 훈련 시에만 드롭아웃 적용
        if self.training:
            logit = self.dropout(logit)
        return logit

    @torch.no_grad()
    def _metrics(self, pred_logit: torch.Tensor, gt_xy: torch.Tensor, polyline_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        새로운 손실 시스템을 위한 메트릭 계산
        
        Args:
            pred_logit: [B,2,H,W] - 모델 출력 로짓
            gt_xy: [B,2,2] - GT 좌표 (픽셀)
            polyline_mask: [B,H,W] - 폴리라인 제약조건 마스크 (옵션)
        """
        H, W = pred_logit.shape[-2:]
        
        # 마스크가 제공되면 제약조건 적용
        if polyline_mask is not None:
            # 폴리라인 영역만 활성화
            masked_logit = pred_logit * polyline_mask.unsqueeze(1)  # [B,1,H,W] -> [B,2,H,W]
            prob = torch.sigmoid(masked_logit)
        else:
            # 마스크가 없으면 기존 방식
            prob = torch.sigmoid(pred_logit)
        
        # soft-argmax로 좌표 추출 (새로운 방식)
        pred_xy = softargmax_2d(prob, temperature=self.loss_fn.temperature)  # [B,2,2]

        # MAE(px) - P1, P2 개별 및 전체 계산
        mae_per_point = torch.linalg.norm(pred_xy - gt_xy, dim=-1)  # [B,2] - 각 포인트별 유클리드 거리
        mae_p1 = mae_per_point[:, 0].mean()  # P1 전용 MAE
        mae_p2 = mae_per_point[:, 1].mean()  # P2 전용 MAE
        mae_px = mae_per_point.mean()        # 전체 평균 (기존 호환성)

        # PCK@thr(px) - P1, P2 개별 및 전체 계산
        dist = torch.linalg.norm(pred_xy - gt_xy, dim=-1)  # [B,2]
        pck_p1 = (dist[:, 0] <= self.pck_thresh_px).float().mean()  # P1 전용 PCK
        pck_p2 = (dist[:, 1] <= self.pck_thresh_px).float().mean()  # P2 전용 PCK
        pck = (dist <= self.pck_thresh_px).float().mean()           # 전체 평균
        
        # 두 점 간 거리 (분리도 평가)
        p1, p2 = pred_xy[:, 0], pred_xy[:, 1]  # [B, 2] each
        separation_distance = torch.norm(p2 - p1, dim=-1).mean()  # 평균 분리 거리

        return {
            "mae_px": mae_px,      # 전체 평균 (기존 호환성)
            "mae_p1": mae_p1,      # P1 개별 MAE
            "mae_p2": mae_p2,      # P2 개별 MAE
            "mae_ratio": mae_p1 / (mae_p2 + 1e-8),  # MAE 불균형 지표
            "pck_px": pck,         # 전체 평균 (기존 호환성)
            "pck_p1": pck_p1,      # P1 개별 PCK
            "pck_p2": pck_p2,      # P2 개별 PCK
            "separation": separation_distance,  # 두 점 간 평균 거리
        }

    def training_step(self, batch, batch_idx):
        img = batch["image"]  # [B,3,H,W]
        gt = batch["coord_px"].float()  # [B,2,2]
        polyline_mask = batch["polyline_mask"]  # [B,H,W] - 폴리라인 제약조건

        H, W = img.shape[-2:]

        # 1. 모델 순전파
        logit = self(img)  # [B,2,H,W]
        
        # 2. 폴리라인 영역만 활성화 (제약조건 적용)
        masked_logit = logit * polyline_mask.unsqueeze(1)  # [B,1,H,W] -> [B,2,H,W]
        prob = torch.sigmoid(masked_logit)
        
        # 3. 단순한 가우시안 히트맵 생성 (키포인트만)
        target = make_heatmaps_torch(gt, H, W, self.sigma)  # [B,2,H,W]
        
        # 4. 타겟 히트맵도 폴리라인 영역에만 제한
        mask_expanded = polyline_mask.unsqueeze(1).expand_as(target)  # [B,2,H,W]
        target_constrained = target * mask_expanded  # 타겟도 마스크 적용
        
        # 5. 새로운 손실 시스템 적용
        loss_dict = self.loss_fn(prob, target_constrained, gt)
        
        loss = loss_dict['total']
        
        # 6. 메트릭 계산
        m = self._metrics(logit, gt, polyline_mask)
        
        # 7. 로깅 (새로운 손실 구성 요소 포함)
        self.log_dict({
            "train_loss": loss,
            "train_heatmap_loss": loss_dict['heatmap'],
            "train_coord_loss": loss_dict['coord'],
            "train_separation_loss": loss_dict['separation'],
            "train_mae_px": m["mae_px"],
            "train_mae_p1": m["mae_p1"],
            "train_mae_p2": m["mae_p2"],
            "train_mae_ratio": m["mae_ratio"],
            "train_pck_px": m["pck_px"],
            "train_pck_p1": m["pck_p1"],
            "train_pck_p2": m["pck_p2"],
            "train_separation": m["separation"],
        }, on_step=True, on_epoch=True, batch_size=img.size(0))
        
        # Progress bar에 핵심 메트릭 표시
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("hm_loss", loss_dict['heatmap'], prog_bar=True, on_step=True, on_epoch=True)
        self.log("coord_loss", loss_dict['coord'], prog_bar=True, on_step=True, on_epoch=True)
        self.log("mae", m["mae_px"], prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        img = batch["image"]
        gt = batch["coord_px"].float()
        polyline_mask = batch["polyline_mask"]  # [B,H,W] - 폴리라인 제약조건

        H, W = img.shape[-2:]
        
        # 1. 모델 순전파
        logit = self(img)  # [B,2,H,W]
        
        # 2. 폴리라인 영역만 활성화 (제약조건 적용)
        masked_logit = logit * polyline_mask.unsqueeze(1)  # [B,1,H,W] -> [B,2,H,W]
        prob = torch.sigmoid(masked_logit)
        
        # 3. 단순한 가우시안 히트맵 생성 (키포인트만)
        target = make_heatmaps_torch(gt, H, W, self.sigma)  # [B,2,H,W]
        
        # 4. 타겟 히트맵도 폴리라인 영역에만 제한
        mask_expanded = polyline_mask.unsqueeze(1).expand_as(target)  # [B,2,H,W]
        target_constrained = target * mask_expanded  # 타겟도 마스크 적용
        
        # 5. 새로운 손실 시스템 적용
        loss_dict = self.loss_fn(prob, target_constrained, gt)
        
        loss = loss_dict['total']
        
        # 6. 메트릭 계산
        m = self._metrics(logit, gt, polyline_mask)
        
        # 7. 로깅 (새로운 손실 구성 요소 포함)
        self.log_dict({
            "val_loss": loss,
            "val_heatmap_loss": loss_dict['heatmap'],
            "val_coord_loss": loss_dict['coord'],
            "val_separation_loss": loss_dict['separation'],
            "val_mae_px": m["mae_px"],
            "val_mae_p1": m["mae_p1"],
            "val_mae_p2": m["mae_p2"],
            "val_mae_ratio": m["mae_ratio"],
            "val_pck_px": m["pck_px"],
            "val_pck_p1": m["pck_p1"],
            "val_pck_p2": m["pck_p2"],
            "val_separation": m["separation"],
        }, on_step=False, on_epoch=True, batch_size=img.size(0))
        
        # Progress bar에 핵심 메트릭 표시
        self.log("v_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("v_hm_loss", loss_dict['heatmap'], prog_bar=True, on_step=False, on_epoch=True)
        self.log("v_coord_loss", loss_dict['coord'], prog_bar=True, on_step=False, on_epoch=True)
        self.log("v_mae", m["mae_px"], prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mae_px",
                "frequency": 1,
            },
        }
