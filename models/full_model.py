import torch
import torch.nn as nn

from models.vq_vae import VQEncoder, EMAVectorQuantizer
from models.stn_uncertainty import MambaSTNUncertainty
from models.decoder import GatedFusionDecoder
from models.channel import DigitalV2XChannel  

class DSCV2XModel(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, num_embeddings=256, base_dims=32):
        super().__init__()
        
        # --- BÊN PHÁT (TRANSMITTER - Xe 2) ---
        self.encoder = VQEncoder(in_channels=in_channels, latent_dim=latent_dim, downsample_factor=8)
        self.quantizer = EMAVectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dim)
        
        # --- KÊNH TRUYỀN (CHANNEL) ---
        self.channel = DigitalV2XChannel(erasure_token_index=-1)
        
        # --- BÊN NHẬN (RECEIVER - Xe 1) ---
        self.stn = MambaSTNUncertainty(in_channels=in_channels, base_dims=base_dims)
        self.decoder = GatedFusionDecoder(latent_dim=latent_dim, base_dims=base_dims, out_channels=in_channels)

    def forward(self, img_X, img_Y, erasure_rate=0.0, bit_flip_prob=0.0):
        # 1. Xe 2 nén ảnh
        z_e = self.encoder(img_Y)
        z_q_ideal, vq_loss, indices = self.quantizer(z_e)
        
        # 2. Truyền qua kênh vật lý
        if erasure_rate > 0.0 or bit_flip_prob > 0.0:
            indices_long = indices.long()
            corrupted_indices, erasure_mask = self.channel(
                indices_long, 
                erasure_rate=erasure_rate, 
                bit_flip_prob=bit_flip_prob
            )
            # Xe 1 giải nén tín hiệu bị lỗi từ Codebook
            z_q_received = self.quantizer.lookup_indices(corrupted_indices, erasure_mask)
            
            # --- SỬA LỖI ĐỨT GRADIENT BẰNG STE TRÊN TÍN HIỆU LỖI ---
            # Gradient sẽ chảy qua z_q_received, mượn đường z_e để đi ngược về Encoder
            z_q_final = z_e + (z_q_received - z_e).detach()
        else:
            # Nếu truyền hoàn hảo (không nhiễu), dùng luôn z_q_ideal có sẵn STE
            z_q_final = z_q_ideal
        
        # 3. Xe 1 nắn tọa độ ảnh Side Info
        img_Y_prime, uncertainty_map, flow = self.stn(img_X)
        
        # 4. Xe 1 dung hợp khôi phục ảnh từ tín hiệu thực tế (z_q_final)
        img_Y_hat = self.decoder(z_q_final, img_Y_prime, uncertainty_map)
        
        return img_Y_hat, img_Y_prime, uncertainty_map, flow, vq_loss, indices