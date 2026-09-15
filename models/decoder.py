# models/decoder.py
import torch
import torch.nn as nn

from models.vq_vae import ModernResidualBlock

class VQFeatureDecoder(nn.Module):
    def __init__(self, latent_dim=64, base_dims=32):
        """
        Progressive spatial decoder for quantized latent tokens (z_q).
        Upsamples the VQ bottleneck features [B, latent_dim, H/8, W/8] 
        back to native resolution [B, base_dims, H, W].
        """
        super().__init__()
        
        # Up-stage 2: H/8 -> H/4
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(latent_dim, base_dims * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dims * 4),
            nn.GELU(),
            ModernResidualBlock(base_dims * 4)
        )
        
        # Up-stage 1: H/4 -> H/2
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_dims * 4, base_dims * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dims * 2),
            nn.GELU(),
            ModernResidualBlock(base_dims * 2)
        )
        
        # Up-stage 0: H/2 -> H
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_dims * 2, base_dims, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dims),
            nn.GELU(),
            ModernResidualBlock(base_dims)
        )

    def forward(self, z_q):
        d1 = self.up1(z_q)
        d2 = self.up2(d1)
        out = self.up3(d2)
        return out


class GatedFusionDecoder(nn.Module):
    def __init__(self, latent_dim=64, base_dims=32, out_channels=3):
        """
        Uncertainty-aware Gated Fusion Network.
        Dynamically merges warped side information (Y') with transmitted VQ tokens (z_q)
        based on the spatial uncertainty map (U).
        """
        super().__init__()
        
        # 1. VQ Token Decoder (Main Information from ego-vehicle)
        self.vq_decoder = VQFeatureDecoder(latent_dim, base_dims)
        
        # 2. Warped Side Information Encoder (Collaborative Information)
        # Projects Y' into the same latent space dimension as the decoded z_q
        self.y_prime_encoder = nn.Sequential(
            nn.Conv2d(3, base_dims, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dims),
            nn.GELU(),
            ModernResidualBlock(base_dims)
        )
        
        # 3. Post-Fusion Feature Refinement
        # Smoothes out the transition boundaries after spatial gating
        self.fusion_refine = nn.Sequential(
            ModernResidualBlock(base_dims),
            nn.Conv2d(base_dims, base_dims, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dims),
            nn.GELU()
        )
        
        # 4. Final RGB Reconstruction
        self.to_rgb = nn.Sequential(
            nn.Conv2d(base_dims, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Normalizes output to [-1.0, 1.0] image space
        )

    def forward(self, z_q, img_Y_prime, uncertainty_map):
        """
        Args:
            z_q: Quantized latent tokens from V2X channel [B, latent_dim, H/8, W/8]
            img_Y_prime: Warped side information [B, 3, H, W]
            uncertainty_map: Reliability map U from Mamba-STN [B, 1, H, W]
        Returns:
            img_Y_hat: Final reconstructed Cooperative Perception frame [B, 3, H, W]
        """
        # Step 1: Decode VQ tokens to native spatial resolution
        feat_vq = self.vq_decoder(z_q)               # [B, 32, H, W]
        
        # Step 2: Extract features from the warped image
        feat_y_prime = self.y_prime_encoder(img_Y_prime) # [B, 32, H, W]
        
        # Step 3: Differentiable Gated Fusion (Soft Spatial Masking)
        # If U -> 1 (Uncertain): Retain feat_vq (transmitted data).
        # If U -> 0 (Certain): Trust feat_y_prime (warped side information).
        feat_fused = (uncertainty_map * feat_vq) + ((1.0 - uncertainty_map) * feat_y_prime)
        
        # Step 4: Refine the fused feature map to eliminate ghosting artifacts
        feat_refined = self.fusion_refine(feat_fused)
        
        # Step 5: Project back to RGB space
        img_Y_hat = self.to_rgb(feat_refined)
        
        return img_Y_hat