import torch
import torch.nn as nn

from models.vq_vae import VQEncoder, EMAVectorQuantizer
from models.stn_uncertainty import MambaSTNUncertainty
from models.decoder import GatedFusionDecoder
from models.channel import DigitalV2XChannel  

class DSCV2XModel(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, num_embeddings=256, base_dims=64, downsample_factor=8):
        super().__init__()
        
        self.encoder = VQEncoder(in_channels=in_channels, latent_dim=latent_dim, downsample_factor=downsample_factor)
        
        # Sửa khởi tạo Codebook an toàn (Uniform) chống Dead Codes
        self.quantizer = EMAVectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dim)
        self.quantizer.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        self.channel = DigitalV2XChannel(erasure_token_index=-1)
        
        self.stn = MambaSTNUncertainty(in_channels=in_channels, base_dims=base_dims)
        self.decoder = GatedFusionDecoder(
            latent_dim=latent_dim, 
            base_dims=base_dims, 
            out_channels=in_channels,
            downsample_factor=downsample_factor
        )

    def forward(self, img_X, img_Y, erasure_rate=0.0, bit_flip_prob=0.0):
        # (Nội dung hàm forward giữ nguyên không thay đổi)
        z_e = self.encoder(img_Y)
        z_q_ideal, vq_loss, indices = self.quantizer(z_e)
        
        if erasure_rate > 0.0 or bit_flip_prob > 0.0:
            indices_long = indices.long()
            corrupted_indices, erasure_mask = self.channel(
                indices_long, erasure_rate=erasure_rate, bit_flip_prob=bit_flip_prob
            )
            z_q_received = self.quantizer.lookup_indices(corrupted_indices, erasure_mask)
            z_q_final = z_e + (z_q_received - z_e).detach()
        else:
            z_q_final = z_q_ideal
        
        img_Y_prime, uncertainty_map, flow = self.stn(img_X)
        img_Y_hat = self.decoder(z_q_final, img_Y_prime, uncertainty_map)
        
        return img_Y_hat, img_Y_prime, uncertainty_map, flow, vq_loss, indices