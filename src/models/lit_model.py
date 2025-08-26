# src/models/lit_model.py
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .img_backbone import DWUNetTiny  # 기존 경량 UNet (약 0.26MB)
from ..losses.heatmap_utils import make_heatmaps_torch, make_combined_heatmaps_from_mask, heatmaps_to_coords_argmax, softargmax_2d


class LitKeypoint2(pl.LightningModule):
    def __init__(self, img_size: int = 512, sigma: float = 5.0, lr: float = 3e-4,
                 pck_thresh_px: float = 5.0, sigma_polyline: float = 3.0, polyline_weight: float = 0.3):
        super().__init__()
        self.save_hyperparameters()
        self.net = DWUNetTiny(in_ch=3, base=64, out_ch=2) # 더 큰 모델 (64 base)
        # 정규화를 위한 드롭아웃 추가
        self.dropout = torch.nn.Dropout2d(p=0.10)
        self.img_size = img_size
        self.sigma = sigma
        self.sigma_polyline = sigma_polyline  # 폴리라인용 가우시안 표준편차
        self.polyline_weight = polyline_weight  # 폴리라인 히트맵 가중치
        self.lr = lr
        self.pck_thresh_px = pck_thresh_px

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
        pred_logit: [B,2,H,W], gt_xy: [B,2,2] (픽셀)
        polyline_mask: [B,H,W] 폴리라인 제약조건 마스크 (옵션)
        """
        H, W = pred_logit.shape[-2:]
        
        # 마스크가 제공되면 제약조건 적용
        if polyline_mask is not None:
            # 폴리라인 영역만 활성화 (훈련과 동일한 방식)
            masked_logit = pred_logit * polyline_mask.unsqueeze(1)  # [B,1,H,W] -> [B,2,H,W]
            prob = torch.sigmoid(masked_logit)
        else:
            # 마스크가 없으면 기존 방식 (하위 호환성)
            prob = torch.sigmoid(pred_logit)
            
        pred_xy = heatmaps_to_coords_argmax(prob)  # [B,2,2] - 마스크된 영역에서 argmax

        # MAE(px) over 2 points - 유클리드 거리 기반
        mae_per_point = torch.linalg.norm(pred_xy - gt_xy, dim=-1)  # [B,2] - 각 포인트별 유클리드 거리
        mae_px = mae_per_point.mean()  # 모든 배치와 포인트의 평균

        # PCK@thr(px): 두 점 평균
        dist = torch.linalg.norm(pred_xy - gt_xy, dim=-1)  # [B,2]
        pck = (dist <= self.pck_thresh_px).float().mean()

        return {"mae_px": mae_px, "pck_px": pck}

    def training_step(self, batch, batch_idx):
        img = batch["image"]  # [B,3,H,W]
        gt = batch["coord_px"].float()  # [B,2,2]
        polyline_mask = batch["polyline_mask"]  # [B,H,W] - 폴리라인 제약조건


        H, W = img.shape[-2:]

        logit = self(img)
        
        # 폴리라인 영역만 활성화 (제약조건 적용)
        masked_logit = logit * polyline_mask.unsqueeze(1)  # [B,1,H,W] broadcast to [B,2,H,W]
        prob = torch.sigmoid(masked_logit)
        
        # 키포인트 + 폴리라인 결합 히트맵 생성
        target = make_combined_heatmaps_from_mask(
            gt, polyline_mask, H, W, 
            self.sigma, self.sigma_polyline, self.polyline_weight
        )
        
        # 타겟 히트맵도 폴리라인 영역에만 제한 (핵심 개선!)
        mask_expanded = polyline_mask.unsqueeze(1).expand_as(target)  # [B,2,H,W]
        target_constrained = target * mask_expanded  # 타겟도 마스크 적용
        
        # 마스크 영역에서만 Loss 계산
        mask_bool = mask_expanded.bool()  # bool 타입으로 변환
        prob_masked = prob[mask_bool]  # [N] - 마스크 True인 픽셀만
        target_masked = target_constrained[mask_bool]  # [N] - 제약된 타겟
        
        loss = F.mse_loss(prob_masked, target_masked)
        m = self._metrics(logit, gt, polyline_mask)  # 마스크 전달하여 제약조건 적용
        self.log_dict({
            "train_loss": loss,
            "train_mae_px": m["mae_px"],
            "train_pck_px": m["pck_px"],
        }, prog_bar=True, on_step=True, on_epoch=True, batch_size=img.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        img = batch["image"]
        gt = batch["coord_px"].float()
        polyline_mask = batch["polyline_mask"]  # [B,H,W] - 폴리라인 제약조건


        H, W = img.shape[-2:]
        
        logit = self(img)
        
        # 폴리라인 영역만 활성화 (제약조건 적용)
        masked_logit = logit * polyline_mask.unsqueeze(1)
        prob = torch.sigmoid(masked_logit)
        
        # 키포인트 + 폴리라인 결합 히트맵 생성
        target = make_combined_heatmaps_from_mask(
            gt, polyline_mask, H, W, 
            self.sigma, self.sigma_polyline, self.polyline_weight
        )
        
        # 타겟 히트맵도 폴리라인 영역에만 제한 (핵심 개선!)
        mask_expanded = polyline_mask.unsqueeze(1).expand_as(target)  # [B,2,H,W]
        target_constrained = target * mask_expanded  # 타겟도 마스크 적용
        
        # 마스크 영역에서만 Loss 계산
        mask_bool = mask_expanded.bool()  # bool 타입으로 변환
        prob_masked = prob[mask_bool]  # [N] - 마스크 True인 픽셀만
        target_masked = target_constrained[mask_bool]  # [N] - 제약된 타겟
        
        loss = F.mse_loss(prob_masked, target_masked)
        m = self._metrics(logit, gt, polyline_mask)  # 마스크 전달하여 제약조건 적용
        self.log_dict({
            "val_loss": loss,
            "val_mae_px": m["mae_px"],
            "val_pck_px": m["pck_px"],
        }, prog_bar=True, on_step=False, on_epoch=True, batch_size=img.size(0))
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
