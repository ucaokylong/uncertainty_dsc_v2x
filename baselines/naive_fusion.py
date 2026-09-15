import torch
import torch.nn as nn
from models.vq_vae import VQEncoder, EMAVectorQuantizer, ModernResidualBlock
from models.decoder import VQFeatureDecoder
from models.channel import DigitalV2XChannel

class NaiveFusionVQVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, num_embeddings=256, base_dims=32):
        super().__init__()
        self.encoder = VQEncoder(in_channels, latent_dim, downsample_factor=8)
        self.quantizer = EMAVectorQuantizer(num_embeddings, latent_dim)
        self.channel = DigitalV2XChannel(erasure_token_index=-1)
        
        self.side_info_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_dims, 4, 2, 1), nn.GELU(),
            nn.Conv2d(base_dims, base_dims, 4, 2, 1), nn.GELU(),
            nn.Conv2d(base_dims, latent_dim, 4, 2, 1), ModernResidualBlock(latent_dim)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(latent_dim * 2, latent_dim, 3, 1, 1), nn.GroupNorm(8, latent_dim), nn.GELU()
        )
        
        self.decoder = nn.Sequential(
            VQFeatureDecoder(latent_dim, base_dims),
            nn.Conv2d(base_dims, base_dims, 3, 1, 1), nn.GroupNorm(8, base_dims), nn.GELU(),
            nn.Conv2d(base_dims, in_channels, 3, 1, 1), nn.Tanh()
        )

    def forward(self, img_X, img_Y, erasure_rate=0.0, bit_flip_prob=0.0):
        z_e = self.encoder(img_Y)
        z_q, vq_loss, indices = self.quantizer(z_e)
        
        if erasure_rate > 0.0 or bit_flip_prob > 0.0:
            corrupted_indices, erasure_mask = self.channel(indices.long(), erasure_rate, bit_flip_prob)
            z_q_received = self.quantizer.lookup_indices(corrupted_indices, erasure_mask)
            z_q_final = z_e + (z_q_received - z_e).detach()
        else:
            z_q_final = z_q
            
        feat_x = self.side_info_encoder(img_X)
        fused_latent = self.fusion_layer(torch.cat([z_q_final, feat_x], dim=1))
        img_Y_hat = self.decoder(fused_latent)
        
        return img_Y_hat, vq_loss, indices