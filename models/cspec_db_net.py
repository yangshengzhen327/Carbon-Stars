# -*- coding: utf-8 -*-
"""CSpec-DB-Net model definition."""

import os

import torch
import torch.nn as nn

from ..config import (
    INDEX_FEATURE_MODE,
    MODEL_CARBON_BANDS,
    WAVE_END,
    WAVE_START,
    build_index_feature_names,
    require_index_feature_mode,
)

class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k, s=1, groups=1, act_layer=nn.GELU):
        super().__init__()
        p = (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=k, stride=s, padding=p, groups=groups, bias=False),
            nn.BatchNorm1d(cout),
            act_layer(),
        )

    def forward(self, x):
        return self.net(x)


class LearnablePositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_tokens, dropout=0.10):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        n_tok = x.size(1)
        if n_tok > self.pos.size(1):
            raise ValueError(f"token length {n_tok} exceeds max positional length {self.pos.size(1)}")
        return self.drop(x + self.pos[:, :n_tok, :])


class SequenceTransformer1D(nn.Module):
    def __init__(self, token_dim=192, max_tokens=33, num_layers=3, nhead=6, dropout=0.10, ff_mult=4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos = LearnablePositionalEncoding1D(token_dim, max_tokens=max_tokens, dropout=dropout)
        self.token_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=token_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, tokens):
        bsz = tokens.size(0)
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = self.pos(x)
        x = self.token_dropout(x)
        x = self.encoder(x)
        return self.norm(x)


class WDCNNSequenceBackbone(nn.Module):
    def __init__(self, in_ch=1, channels=(64, 32, 64, 64, 128), conv_dropout=0.05):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.out_channels = c5
        self.block1 = nn.Sequential(
            ConvBNAct(in_ch, c1, k=64, s=8, act_layer=nn.SiLU),
            nn.MaxPool1d(2),
        )
        self.block2 = nn.Sequential(
            ConvBNAct(c1, c2, k=5, s=1, act_layer=nn.SiLU),
            nn.MaxPool1d(2),
        )
        self.block3 = nn.Sequential(
            ConvBNAct(c2, c3, k=3, s=1, act_layer=nn.SiLU),
            nn.MaxPool1d(2),
            nn.Dropout1d(conv_dropout),
        )
        self.block4 = nn.Sequential(
            ConvBNAct(c3, c4, k=3, s=1, act_layer=nn.SiLU),
            nn.MaxPool1d(2),
            nn.Dropout1d(conv_dropout),
        )
        self.block5 = nn.Sequential(
            ConvBNAct(c4, c5, k=3, s=1, act_layer=nn.SiLU),
            nn.Dropout1d(conv_dropout),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class WDCNNTransformerWaveEncoder(nn.Module):
    def __init__(
        self,
        signal_len=4096,
        embed_dim=192,
        transformer_layers=3,
        num_heads=6,
        dropout=0.10,
        conv_dropout=0.05,
        transformer_ff_mult=4,
        post_hidden=(256, 192),
    ):
        super().__init__()
        self.backbone = WDCNNSequenceBackbone(in_ch=1, conv_dropout=conv_dropout)
        self.token_proj = nn.Conv1d(self.backbone.out_channels, embed_dim, kernel_size=1, bias=False)
        self.token_bn = nn.BatchNorm1d(embed_dim)
        self.token_act = nn.GELU()
        self.token_drop = nn.Dropout1d(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, signal_len)
            token_count = self.backbone(dummy).shape[-1]
        self.token_count = int(token_count)

        self.token_norm = nn.LayerNorm(embed_dim)
        self.transformer = SequenceTransformer1D(
            token_dim=embed_dim,
            max_tokens=self.token_count + 1,
            num_layers=transformer_layers,
            nhead=num_heads,
            dropout=dropout,
            ff_mult=transformer_ff_mult,
        )

        hidden1, hidden2 = post_hidden
        self.post_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat_map = self.backbone(x)
        feat_map = self.token_proj(feat_map)
        feat_map = self.token_bn(feat_map)
        feat_map = self.token_act(feat_map)
        feat_map = self.token_drop(feat_map)

        tokens = feat_map.transpose(1, 2)
        tokens = self.token_norm(tokens)
        z = self.transformer(tokens)

        cls_feat = z[:, 0]
        seq = z[:, 1:]
        mean_feat = seq.mean(dim=1)
        max_feat = seq.max(dim=1).values
        wave_feat = self.post_mlp(torch.cat([cls_feat, mean_feat, max_feat], dim=1))
        return wave_feat, seq


class CarbonPriorLayer(nn.Module):
    def __init__(self, wave_start=WAVE_START, wave_end=WAVE_END, n_pix=4096, carbon_bands=None, mode=INDEX_FEATURE_MODE):
        super().__init__()
        require_index_feature_mode(mode)
        self.mode = mode
        wave = torch.linspace(wave_start, wave_end, n_pix)
        if carbon_bands is None:
            carbon_bands = MODEL_CARBON_BANDS
        self.carbon_bands = carbon_bands

        band_windows = [(4620, 4742), (4980, 5170), (5350, 5640), (7065, 7190), (7820, 8000)]
        band_masks = [((wave >= a) & (wave < b)).float() for a, b in band_windows]
        coarse_windows = [(4100, 5200), (5200, 7000), (7000, 8600)]
        coarse_masks = [((wave >= a) & (wave < b)).float() for a, b in coarse_windows]

        self.register_buffer("band_masks", torch.stack(band_masks))
        self.register_buffer("coarse_masks", torch.stack(coarse_masks))
        self.eps = 1e-6

    def masked_mean(self, x, masks):
        num = (x[:, None, :] * masks[None, :, :]).sum(dim=-1)
        den = masks.sum(dim=-1)[None, :] + self.eps
        return num / den

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, 0, :]
        raw_band_means = self.masked_mean(x, self.band_masks)
        raw_coarse = self.masked_mean(x, self.coarse_masks)
        band_means = raw_band_means
        coarse = raw_coarse
        slope = raw_coarse[:, 2:3] - raw_coarse[:, 0:1]
        spread = x.std(dim=-1, keepdim=True)
        band_span = band_means.max(dim=1, keepdim=True).values - band_means.min(dim=1, keepdim=True).values
        band_strength = band_means.mean(dim=1, keepdim=True)
        return torch.cat([band_means, coarse, slope, spread, band_span, band_strength], dim=1)


class GlobalStatLayer(nn.Module):
    def __init__(self, mode=INDEX_FEATURE_MODE):
        super().__init__()
        require_index_feature_mode(mode)
        self.mode = mode

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, 0, :]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        mx = x.max(dim=-1, keepdim=True).values
        mn = x.min(dim=-1, keepdim=True).values
        slope = x[:, -1:] - x[:, :1]
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        return torch.cat([mean, std, mx, mn, slope, abs_mean], dim=1)


class EnhancedBandFeatureLayer(nn.Module):
    def __init__(self, wave_start=3986.0, wave_end=9100.0, n_pix=4096, mode=INDEX_FEATURE_MODE):
        super().__init__()
        require_index_feature_mode(mode)
        self.mode = mode

        wave = torch.linspace(wave_start, wave_end, n_pix)
        self.band_defs = [
            ("c2_4737", (4620, 4742), (4580, 4620), (4742, 4812)),
            ("c2_5165", (4980, 5170), (4930, 4980), (5170, 5235)),
            ("c2_5635", (5350, 5640), (5300, 5350), (5640, 5700)),
            ("cn_7065", (7065, 7190), (7025, 7065), (7190, 7230)),
            ("cn_7820", (7820, 8000), (7790, 7820), (8000, 8040)),
        ]
        self.ctrl_defs = [
            ("ctrl_1", (4500, 4580)),
            ("ctrl_2", (4815, 4890)),
            ("ctrl_3", (5240, 5320)),
            ("ctrl_4", (7240, 7320)),
        ]

        band_masks = []
        left_masks = []
        right_masks = []
        for _, band, left, right in self.band_defs:
            band_masks.append(((wave >= band[0]) & (wave < band[1])).float())
            left_masks.append(((wave >= left[0]) & (wave < left[1])).float())
            right_masks.append(((wave >= right[0]) & (wave < right[1])).float())

        ctrl_masks = [((wave >= ctrl[0]) & (wave < ctrl[1])).float() for _, ctrl in self.ctrl_defs]
        self.register_buffer("band_masks", torch.stack(band_masks))
        self.register_buffer("left_masks", torch.stack(left_masks))
        self.register_buffer("right_masks", torch.stack(right_masks))
        self.register_buffer("ctrl_masks", torch.stack(ctrl_masks))
        self.register_buffer("wave_grid", wave)
        self.register_buffer(
            "band_starts",
            torch.tensor([band[0] for _, band, _, _ in self.band_defs], dtype=torch.float32),
        )
        self.register_buffer(
            "band_ends",
            torch.tensor([band[1] for _, band, _, _ in self.band_defs], dtype=torch.float32),
        )
        self.eps = 1e-6

    def masked_mean(self, x, masks):
        num = (x[:, None, :] * masks[None, :, :]).sum(dim=-1)
        den = masks.sum(dim=-1)[None, :] + self.eps
        return num / den

    def masked_std(self, x, masks):
        mean = self.masked_mean(x, masks)
        diff2 = ((x[:, None, :] - mean[:, :, None]) ** 2) * masks[None, :, :]
        den = masks.sum(dim=-1)[None, :] + self.eps
        return torch.sqrt(diff2.sum(dim=-1) / den + self.eps)

    def masked_min(self, x, masks):
        bsz, seq_len = x.shape
        n_mask = masks.shape[0]
        big = torch.full((bsz, n_mask, seq_len), 1e9, device=x.device, dtype=x.dtype)
        xx = x[:, None, :].expand(bsz, n_mask, seq_len)
        return torch.where(masks[None, :, :].bool(), xx, big).min(dim=-1).values

    def build_band_features(self, x):
        raw_band_mean = self.masked_mean(x, self.band_masks)
        band_min = self.masked_min(x, self.band_masks)
        band_std = self.masked_std(x, self.band_masks)
        left_mean = self.masked_mean(x, self.left_masks)
        right_mean = self.masked_mean(x, self.right_masks)
        cont = torch.clamp(0.5 * (left_mean + right_mean), min=1e-4)

        depth = 1.0 - raw_band_mean / cont
        pseudo_ew_values = []
        for band_index in range(self.band_masks.shape[0]):
            mask = self.band_masks[band_index].bool()
            band_wave = self.wave_grid[mask]
            alpha = (band_wave - self.band_starts[band_index]) / (
                self.band_ends[band_index] - self.band_starts[band_index]
            )
            continuum = left_mean[:, band_index : band_index + 1] + (
                right_mean[:, band_index : band_index + 1] - left_mean[:, band_index : band_index + 1]
            ) * alpha[None, :]
            continuum = torch.clamp(continuum, min=1e-4)
            integrand = 1.0 - x[:, mask] / continuum
            pseudo_ew_values.append(torch.trapezoid(integrand, band_wave, dim=1))
        pseudo_ew = torch.stack(pseudo_ew_values, dim=1)
        continuum_balance = torch.abs(left_mean - right_mean)
        features = torch.cat([raw_band_mean, band_min, band_std, depth, pseudo_ew, continuum_balance], dim=1)
        return features, {"depth": depth, "pseudo_ew": pseudo_ew}

    def build_ratio_features(self, cache):
        depth = torch.clamp(cache["depth"], min=0.0)
        pseudo_ew = torch.clamp(cache["pseudo_ew"], min=0.0)
        eps = self.eps

        d4737 = depth[:, 0:1]
        d5165 = depth[:, 1:2]
        d5635 = depth[:, 2:3]
        d7065 = depth[:, 3:4]
        d7820 = depth[:, 4:5]

        p4737 = pseudo_ew[:, 0:1]
        p5165 = pseudo_ew[:, 1:2]
        p5635 = pseudo_ew[:, 2:3]
        p7065 = pseudo_ew[:, 3:4]
        p7820 = pseudo_ew[:, 4:5]

        c2_depth_mean = (d4737 + d5165 + d5635) / 3.0
        cn_depth_mean = (d7065 + d7820) / 2.0
        c2_pew_sum = p4737 + p5165 + p5635
        cn_pew_sum = p7065 + p7820
        c2_stack = torch.cat([d4737, d5165, d5635], dim=1)
        c2_max = c2_stack.max(dim=1, keepdim=True).values
        c2_min = c2_stack.min(dim=1, keepdim=True).values

        ratios = [
            d5165 / (d5635 + eps),
            d7065 / (d7820 + eps),
            c2_depth_mean / (cn_depth_mean + eps),
            p5165 / (p5635 + eps),
            c2_pew_sum / (cn_pew_sum + eps),
            c2_max / (c2_min + eps),
        ]
        return torch.cat([torch.clamp(r, 0.0, 5.0) for r in ratios], dim=1)

    def build_control_features(self, x):
        ctrl_mean = self.masked_mean(x, self.ctrl_masks)
        ctrl_std = self.masked_std(x, self.ctrl_masks)
        return torch.cat([ctrl_mean, ctrl_std], dim=1)

    def get_out_dim(self):
        return 44

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, 0, :]
        band_features, cache = self.build_band_features(x)
        return torch.cat(
            [band_features, self.build_ratio_features(cache), self.build_control_features(x)],
            dim=1,
        )


class SpectralDerivativeFeatureLayer(nn.Module):
    def __init__(self, wave_start=3986.0, wave_end=9100.0, n_pix=4096):
        super().__init__()
        wave = torch.linspace(wave_start, wave_end, n_pix)
        band_defs = [(4620, 4742), (4980, 5170), (5350, 5640), (7065, 7190), (7820, 8000)]
        masks = [((wave >= a) & (wave < b)).float() for a, b in band_defs]
        self.register_buffer("band_masks", torch.stack(masks))
        self.eps = 1e-6

    def masked_mean(self, x, masks):
        num = (x[:, None, :] * masks[None, :, :]).sum(dim=-1)
        den = masks.sum(dim=-1)[None, :] + self.eps
        return num / den

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, 0, :]
        d1 = torch.diff(x, dim=-1, prepend=x[:, :1])
        d2 = torch.diff(d1, dim=-1, prepend=d1[:, :1])
        d1_mean = self.masked_mean(d1, self.band_masks)
        d1_abs = self.masked_mean(d1.abs(), self.band_masks)
        d2_mean = self.masked_mean(d2, self.band_masks)
        d2_abs = self.masked_mean(d2.abs(), self.band_masks)
        return torch.cat([d1_mean, d1_abs, d2_mean, d2_abs], dim=1)


class StrongIndexBranch(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=192,
        hidden1=256,
        hidden2=192,
        dropout=0.20,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.mlp(x)


def init_weights_trunc_normal(module):
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


class CSpecDBNet(nn.Module):
    def __init__(
        self,
        signal_len=4096,
        wave_start=WAVE_START,
        wave_end=WAVE_END,
        carbon_bands=None,
        embed_dim=192,
        num_heads=6,
        wave_transformer_layers=3,
        dropout=0.10,
        conv_dropout=0.05,
        transformer_ff_mult=4,
        wave_hidden=(256, 192),
        idx_hidden=(256, 192),
        fusion_hidden=(256, 128),
        fusion_dropouts=(0.15, 0.10),
        idx_feature_mode=INDEX_FEATURE_MODE,
    ):
        super().__init__()
        require_index_feature_mode(idx_feature_mode)
        self.idx_feature_mode = idx_feature_mode
        self.wave_branch = WDCNNTransformerWaveEncoder(
            signal_len=signal_len,
            embed_dim=embed_dim,
            transformer_layers=wave_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            conv_dropout=conv_dropout,
            transformer_ff_mult=transformer_ff_mult,
            post_hidden=wave_hidden,
        )

        self.band_feature_layer = EnhancedBandFeatureLayer(
            wave_start=wave_start,
            wave_end=wave_end,
            n_pix=signal_len,
            mode=idx_feature_mode,
        )
        self.carbon_priors = CarbonPriorLayer(
            wave_start=wave_start,
            wave_end=wave_end,
            n_pix=signal_len,
            carbon_bands=carbon_bands,
            mode=idx_feature_mode,
        )
        self.global_stats = GlobalStatLayer(mode=idx_feature_mode)
        self.derivative_stats = SpectralDerivativeFeatureLayer(
            wave_start=wave_start,
            wave_end=wave_end,
            n_pix=signal_len,
        )
        self.index_feature_names = tuple(build_index_feature_names(idx_feature_mode))
        self.index_feature_full_names = self.index_feature_names
        self.index_feature_dropped_names = tuple()
        self.index_feature_handcrafted_dim = len(self.index_feature_names)
        self.index_feature_branch_input_dim = self.index_feature_handcrafted_dim
        self.index_feature_scale = None
        self.index_feature_lift = None

        self.index_branch = StrongIndexBranch(
            in_dim=self.index_feature_handcrafted_dim,
            out_dim=embed_dim,
            hidden1=idx_hidden[0],
            hidden2=idx_hidden[1],
            dropout=max(dropout, 0.20),
        )

        fusion_hidden1, fusion_hidden2 = fusion_hidden
        drop1, drop2 = fusion_dropouts
        self.cross_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(drop1),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(embed_dim * 4, fusion_hidden1),
            nn.LayerNorm(fusion_hidden1),
            nn.GELU(),
            nn.Dropout(drop1),
            nn.Linear(fusion_hidden1, fusion_hidden2),
            nn.LayerNorm(fusion_hidden2),
            nn.GELU(),
            nn.Dropout(drop2),
        )
        self.bin_head = nn.Linear(fusion_hidden2, 2)
        self.apply(init_weights_trunc_normal)

    def build_index_feature_parts(self, x_idx):
        band_feat = self.band_feature_layer(x_idx)
        prior_feat = self.carbon_priors(x_idx)
        stat_feat = self.global_stats(x_idx)
        deriv_feat = self.derivative_stats(x_idx)
        return band_feat, prior_feat, stat_feat, deriv_feat

    def build_index_features(self, x_idx):
        band_feat, prior_feat, stat_feat, deriv_feat = self.build_index_feature_parts(x_idx)
        return torch.cat([band_feat, prior_feat, stat_feat, deriv_feat], dim=1)

    def forward(self, x, x_idx=None):
        if x_idx is None:
            x_idx = x
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x_idx.dim() == 2:
            x_idx = x_idx.unsqueeze(1)

        wave_feat, wave_tokens = self.wave_branch(x)
        idx_input = self.build_index_features(x_idx)
        idx_feat = self.index_branch(idx_input)
        gate = self.cross_gate(torch.cat([wave_feat, idx_feat], dim=1))
        gated_wave = wave_feat * gate
        gated_idx = idx_feat * (1.0 - gate)
        diff_feat = torch.abs(wave_feat - idx_feat)
        prod_feat = wave_feat * idx_feat
        fused_feat = self.fusion_head(torch.cat([gated_wave, gated_idx, diff_feat, prod_feat], dim=1))
        bin_logits = self.bin_head(fused_feat)
        return {
            "bin_logits": bin_logits,
            "feat": fused_feat,
            "feat_wave": wave_feat,
            "feat_idx": idx_feat,
            "fusion_gate": gate,
            "wave_tokens": wave_tokens,
            "index_evidence_bonus": None,
            "index_evidence_band_bonus": None,
            "index_evidence_feature": None,
            "index_calibration_penalty": None,
        }


WDCNN_BASELINE_TRANSFER_BLOCKS = (
    ("backbone.features.0.", "wave_branch.backbone.block1.0."),
    ("backbone.features.2.", "wave_branch.backbone.block2.0."),
    ("backbone.features.4.", "wave_branch.backbone.block3.0."),
    ("backbone.features.6.", "wave_branch.backbone.block4.0."),
    ("backbone.features.8.", "wave_branch.backbone.block5.0."),
)


def load_wdcnn_backbone_checkpoint(model, checkpoint_path):
    if not checkpoint_path:
        return 0, 0
    if not os.path.exists(checkpoint_path):
        print(f"[WARN] init_wdcnn_checkpoint not found: {checkpoint_path}")
        return 0, 0

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_state = model.state_dict()
    transfer_state = {}
    total = 0

    for src_prefix, dst_prefix in WDCNN_BASELINE_TRANSFER_BLOCKS:
        for key, value in state_dict.items():
            if not key.startswith(src_prefix):
                continue
            total += 1
            dst_key = dst_prefix + key[len(src_prefix):]
            if dst_key in model_state and model_state[dst_key].shape == value.shape:
                transfer_state[dst_key] = value

    if transfer_state:
        model.load_state_dict(transfer_state, strict=False)
    print(f"[INFO] wdcnn preinit loaded {len(transfer_state)}/{total} tensors from {checkpoint_path}")
    return len(transfer_state), total


def load_matching_state_dict(model, state_dict, checkpoint_path="", skip_prefixes=()):
    model_state = model.state_dict()
    matched = {}
    total = 0
    for key, value in state_dict.items():
        total += 1
        if skip_prefixes and any(str(key).startswith(prefix) for prefix in skip_prefixes):
            continue
        if key in model_state and model_state[key].shape == value.shape:
            matched[key] = value
    if matched:
        model.load_state_dict(matched, strict=False)
    path_text = checkpoint_path or "<memory>"
    print(f"[INFO] init checkpoint loaded {len(matched)}/{total} tensors from {path_text}")
    return len(matched), total


def set_module_trainable(module, trainable=True):
    for param in module.parameters():
        param.requires_grad = bool(trainable)


def apply_train_only_prefixes(model, train_only_prefixes):
    prefixes = tuple(prefix for prefix in train_only_prefixes if prefix)
    if not prefixes:
        return {"trainable_param_count": 0, "frozen_param_count": 0, "matched_names": []}

    matched_names = []
    trainable_param_count = 0
    frozen_param_count = 0
    for name, param in model.named_parameters():
        is_trainable = name.startswith(prefixes)
        param.requires_grad = bool(is_trainable)
        if is_trainable:
            matched_names.append(name)
            trainable_param_count += param.numel()
        else:
            frozen_param_count += param.numel()

    unique_matched_prefixes = sorted({name.split(".", 1)[0] for name in matched_names})
    if matched_names:
        print(
            "[INFO] train_only_prefixes matched "
            f"{len(matched_names)} tensors across {len(unique_matched_prefixes)} top-level modules: "
            + ", ".join(unique_matched_prefixes)
        )
    else:
        print("[WARN] train_only_prefixes matched 0 tensors")
    return {
        "trainable_param_count": int(trainable_param_count),
        "frozen_param_count": int(frozen_param_count),
        "matched_names": matched_names,
    }


# Backward-compatible name used by the original training script/checkpoints.
WDTransformerCarbonNet = CSpecDBNet

__all__ = [
    "CSpecDBNet",
    "WDTransformerCarbonNet",
    "load_wdcnn_backbone_checkpoint",
    "load_matching_state_dict",
    "set_module_trainable",
    "apply_train_only_prefixes",
]
