import torch
import torch.nn as nn
from models.vq_vae import ModernResidualBlock

class VQFeatureDecoder(nn.Module):
    def __init__(self, latent_dim=64, base_dims=32, downsample_factor=8):
        """
        Decoder giờ đây tự động sinh ra số lớp (2, 3, hoặc 4 lớp) 
        dựa vào downsample_factor (4, 8, hoặc 16) truyền từ file config.
        """
        super().__init__()
        
        stride_counts = {4: 2, 8: 3, 16: 4}
        num_strides = stride_counts[downsample_factor]
        
        # Cấu hình channel giảm dần cho từng bước upsample
        dim_map = {
            2: [base_dims * 2, base_dims],
            3: [base_dims * 4, base_dims * 2, base_dims],
            4: [base_dims * 4, base_dims * 4, base_dims * 2, base_dims]
        }
        out_channels_list = dim_map[num_strides]
        
        layers = []
        curr_in = latent_dim
        
        for out_c in out_channels_list:
            layers.extend([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(curr_in, out_c, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_c),
                nn.GELU(),
                ModernResidualBlock(out_c)
            ])
            curr_in = out_c
            
        self.net = nn.Sequential(*layers)

    def forward(self, z_q):
        return self.net(z_q)


class GatedFusionDecoder(nn.Module):
    def __init__(self, latent_dim=64, base_dims=32, out_channels=3, downsample_factor=8):
        super().__init__()
        
        # Truyền downsample_factor xuống VQFeatureDecoder
        self.vq_decoder = VQFeatureDecoder(latent_dim, base_dims, downsample_factor)
        
        self.y_prime_encoder = nn.Sequential(
            nn.Conv2d(3, base_dims, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dims),
            nn.GELU(),
            ModernResidualBlock(base_dims)
        )
        
        self.fusion_refine = nn.Sequential(
            ModernResidualBlock(base_dims),
            nn.Conv2d(base_dims, base_dims, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dims),
            nn.GELU()
        )
        
        self.to_rgb = nn.Sequential(
            nn.Conv2d(base_dims, out_channels, kernel_size=3, padding=1),
            nn.Tanh() 
        )

    def forward(self, z_q, img_Y_prime, uncertainty_map):
        feat_vq = self.vq_decoder(z_q)               
        feat_y_prime = self.y_prime_encoder(img_Y_prime) 
        
        feat_fused = (uncertainty_map * feat_vq) + ((1.0 - uncertainty_map) * feat_y_prime)
        feat_refined = self.fusion_refine(feat_fused)
        img_Y_hat = self.to_rgb(feat_refined)
        
        return img_Y_hat