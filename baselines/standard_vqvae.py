import torch
import torch.nn as nn
from models.vq_vae import VQEncoder, EMAVectorQuantizer
from models.decoder import VQFeatureDecoder
from models.channel import DigitalV2XChannel

class StandardVQVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, num_embeddings=256, base_dims=32):
        super().__init__()
        self.encoder = VQEncoder(in_channels, latent_dim, downsample_factor=8)
        self.quantizer = EMAVectorQuantizer(num_embeddings, latent_dim)
        self.channel = DigitalV2XChannel(erasure_token_index=-1)
        
        self.decoder = nn.Sequential(
            VQFeatureDecoder(latent_dim, base_dims),
            nn.Conv2d(base_dims, base_dims, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dims),
            nn.GELU(),
            nn.Conv2d(base_dims, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, img_Y, erasure_rate=0.0, bit_flip_prob=0.0):
        z_e = self.encoder(img_Y)
        z_q, vq_loss, indices = self.quantizer(z_e)
        
        if erasure_rate > 0.0 or bit_flip_prob > 0.0:
            indices_long = indices.long()
            corrupted_indices, erasure_mask = self.channel(indices_long, erasure_rate, bit_flip_prob)
            z_q_received = self.quantizer.lookup_indices(corrupted_indices, erasure_mask)
            z_q_final = z_e + (z_q_received - z_e).detach()
        else:
            z_q_final = z_q
            
        img_Y_hat = self.decoder(z_q_final)
        return img_Y_hat, vq_loss, indices