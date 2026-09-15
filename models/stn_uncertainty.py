import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
from models.vq_vae import ModernResidualBlock

class VisionMambaBlock(nn.Module):
    def __init__(self, channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.channels = channels
        
        self.mamba_fwd = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)
        x_norm = self.norm(x_flat)
        
        out_fwd = self.mamba_fwd(x_norm)
        
        x_reversed = torch.flip(x_norm, dims=[1])
        out_bwd = torch.flip(self.mamba_bwd(x_reversed), dims=[1])
        
        out = out_fwd + out_bwd + x_flat
        return out.transpose(1, 2).view(B, C, H, W)


class MambaSTNUncertainty(nn.Module):
    def __init__(self, in_channels=3, base_dims=32):
        super().__init__()
        
        # --- ENCODER ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_dims, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, base_dims),
            nn.GELU()
        )
        self.enc2 = nn.Sequential(
            ModernResidualBlock(base_dims),
            nn.Conv2d(base_dims, base_dims * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_dims * 2),
            nn.GELU()
        )
        self.enc3 = nn.Sequential(
            ModernResidualBlock(base_dims * 2),
            nn.Conv2d(base_dims * 2, base_dims * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_dims * 4),
            nn.GELU()
        )

        # --- BOTTLENECK ---
        self.bottleneck = nn.Sequential(
            ModernResidualBlock(base_dims * 4),
            VisionMambaBlock(channels=base_dims * 4),
            ModernResidualBlock(base_dims * 4)
        )

        # --- DECODER (Đã tách Upsample để tránh lỗi Mismatch) ---
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_dims * 4 + base_dims * 2, base_dims * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dims * 2),
            nn.GELU()
        )
        
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_dims * 2 + base_dims, base_dims, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dims),
            nn.GELU()
        )
        
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec0 = nn.Sequential(
            nn.Conv2d(base_dims, base_dims, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dims),
            nn.GELU(),
            ModernResidualBlock(base_dims)
        )

        # --- DUAL HEADS ---
        self.flow_head = nn.Sequential(
            nn.Conv2d(base_dims, 2, kernel_size=3, padding=1),
            nn.Tanh() 
        )
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(base_dims, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, img_X):
        B, C, H, W = img_X.shape

        e1 = self.enc1(img_X)         
        e2 = self.enc2(e1)            
        e3 = self.enc3(e2)            
        b = self.bottleneck(e3)       

        # Phóng to không gian trước khi Concatenate
        b_up = self.upsample2(b)
        d2 = self.dec2(torch.cat([b_up, e2], dim=1))
        
        d2_up = self.upsample1(d2)
        d1 = self.dec1(torch.cat([d2_up, e1], dim=1))
        
        d1_up = self.upsample0(d1)
        feat = self.dec0(d1_up)       

        flow = self.flow_head(feat)                    
        uncertainty_map = self.uncertainty_head(feat)  

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=img_X.device),
            torch.linspace(-1.0, 1.0, W, device=img_X.device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        flow_permuted = flow.permute(0, 2, 3, 1)
        sampling_grid = base_grid + flow_permuted

        img_Y_prime = F.grid_sample(
            img_X, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True
        )

        return img_Y_prime, uncertainty_map, flow