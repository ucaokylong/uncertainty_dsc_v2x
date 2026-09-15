import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

from models.vq_vae import VQEncoder, EMAVectorQuantizer
from models.decoder import VQFeatureDecoder
from models.channel import DigitalV2XChannel

class RAFTFusionBaseline(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, num_embeddings=256, base_dims=32):
        super().__init__()
        
        self.encoder = VQEncoder(in_channels, latent_dim, downsample_factor=8)
        self.quantizer = EMAVectorQuantizer(num_embeddings, latent_dim)
        self.channel = DigitalV2XChannel(erasure_token_index=-1)
        
        # 1. Bộ giải mã độc lập (Tạo ảnh nháp Y_indep)
        self.indep_decoder = nn.Sequential(
            VQFeatureDecoder(latent_dim, base_dims),
            nn.Conv2d(base_dims, in_channels, 3, 1, 1),
            nn.Tanh()
        )
        
        # 2. Đóng băng (Freeze) SOTA RAFT
        weights = Raft_Small_Weights.DEFAULT
        self.raft = raft_small(weights=weights, progress=False)
        for param in self.raft.parameters():
            param.requires_grad = False
        self.raft.eval() # Luôn ở trạng thái đánh giá
        
        # 3. Mạng CNN trộn 2 ảnh (Late Fusion)
        self.late_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_dims, 3, 1, 1),
            nn.GroupNorm(8, base_dims),
            nn.GELU(),
            nn.Conv2d(base_dims, base_dims, 3, 1, 1),
            nn.GroupNorm(8, base_dims),
            nn.GELU(),
            nn.Conv2d(base_dims, in_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, img_X, img_Y, erasure_rate=0.0, bit_flip_prob=0.0):
        B, C, H, W = img_Y.shape
        
        # Nén và truyền
        z_e = self.encoder(img_Y)
        z_q, vq_loss, indices = self.quantizer(z_e)
        
        if erasure_rate > 0.0 or bit_flip_prob > 0.0:
            corrupted_indices, erasure_mask = self.channel(indices.long(), erasure_rate, bit_flip_prob)
            z_q_received = self.quantizer.lookup_indices(corrupted_indices, erasure_mask)
            z_q_final = z_e + (z_q_received - z_e).detach()
        else:
            z_q_final = z_q
            
        # Giải mã ra ảnh nháp
        img_Y_indep = self.indep_decoder(z_q_final)
        
        # Tính Optical Flow bằng RAFT (Đầu vào cần scale về [-1, 1] chuẩn PyTorch)
        with torch.no_grad():
            flow_predictions = self.raft(img_X, img_Y_indep)
            flow_px = flow_predictions[-1] # Lấy flow tinh chỉnh cuối cùng [B, 2, H, W]
        
        # Chuẩn hóa Flow từ Pixel về hệ tọa độ Grid [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=img_X.device),
            torch.linspace(-1.0, 1.0, W, device=img_X.device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Công thức quy đổi Pixel Displacement -> Normalized Grid Displacement
        flow_norm_x = 2.0 * flow_px[:, 0, :, :] / (W - 1)
        flow_norm_y = 2.0 * flow_px[:, 1, :, :] / (H - 1)
        flow_norm = torch.stack([flow_norm_x, flow_norm_y], dim=-1)
        
        sampling_grid = base_grid + flow_norm
        img_Y_prime = F.grid_sample(img_X, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        # Late Fusion giữa Ảnh nắn và Ảnh nháp
        img_Y_hat = self.late_fusion(torch.cat([img_Y_prime, img_Y_indep], dim=1))
        
        return img_Y_hat, vq_loss, indices