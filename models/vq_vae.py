# models/vq_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernResidualBlock(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        # Thay thế ReLU chay bằng cấu trúc Pre-activation: Norm -> GELU -> Conv
        # GroupNorm ổn định hơn BatchNorm khi train với Batch Size nhỏ
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return x + self.block(x)


class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=256, embedding_dim=64, beta=0.25, decay=0.99, epsilon=1e-5):
        """
        Lượng tử hóa hiện đại sử dụng Exponential Moving Average (EMA).
        - decay: Tốc độ trượt của EMA (thường dùng 0.99)
        - epsilon: Hệ số tránh chia cho 0
        """
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon

        # Bảng từ điển không nhận gradient (được cập nhật thủ công qua EMA)
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.normal_()
        self.embedding.weight.requires_grad = False  # Khóa gradient

        # Các bộ đệm (buffers) để tính EMA, không được tối ưu bởi Optimizer
        self.register_buffer('cluster_size', torch.zeros(self.K))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())

    def forward(self, z_e):
        # [B, D, h, w] -> [B, h, w, D] -> [N, D]
        z_e_perm = z_e.permute(0, 2, 3, 1).contiguous()
        flat_ze = z_e_perm.view(-1, self.D)

        # Tính khoảng cách Euclidean
        distances = (
            torch.sum(flat_ze ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_ze, self.embedding.weight.t())
            + torch.sum(self.embedding.weight ** 2, dim=1)
        )

        indices = torch.argmin(distances, dim=1)
        indices_2d = indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])
        z_q = self.embedding(indices_2d).permute(0, 3, 1, 2).contiguous()

        # Quá trình cập nhật EMA Codebook (Chỉ diễn ra trong lúc Train)
        if self.training:
            # Mã hóa one-hot cho các index được chọn [N, K]
            encodings = F.one_hot(indices, self.K).float()
            
            # Tính số lượng token rơi vào từng cụm
            cluster_size_batch = torch.sum(encodings, dim=0)
            
            # Tính tổng các vector z_e rơi vào từng cụm
            embed_avg_batch = torch.matmul(encodings.t(), flat_ze)

            # Cập nhật trung bình trượt EMA
            self.cluster_size.data.mul_(self.decay).add_(cluster_size_batch, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_avg_batch, alpha=1 - self.decay)

            # Laplace smoothing (Tránh cụm có 0 phần tử làm codebook bị nổ hỏng)
            n = torch.sum(self.cluster_size.data)
            cluster_size_smooth = (
                (self.cluster_size.data + self.epsilon) / 
                (n + self.K * self.epsilon) * n
            )
            
            # Gán lại trọng số từ điển mới
            embed_normalized = self.embed_avg.data / cluster_size_smooth.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

        # Mất mát duy nhất còn lại là Commitment Loss
        loss_commitment = F.mse_loss(z_e, z_q.detach())
        vq_loss = self.beta * loss_commitment

        # STE cầu vượt Gradient
        z_q = z_e + (z_q - z_e).detach()

        return z_q, vq_loss, indices_2d

    def lookup_indices(self, indices, erasure_mask=None):
        valid_indices = indices.clone()
        if erasure_mask is not None:
            valid_indices[erasure_mask] = 0

        z_q = self.embedding(valid_indices).permute(0, 3, 1, 2).contiguous()

        if erasure_mask is not None:
            mask_expanded = erasure_mask.unsqueeze(1).float()
            z_q = z_q * (1.0 - mask_expanded)

        return z_q


class VQEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, downsample_factor=8):
        super().__init__()
        layers = []
        curr_in = in_channels
        
        stride_counts = {4: 2, 8: 3, 16: 4}
        num_strides = stride_counts[downsample_factor]

        # Khối thu nhỏ ảnh ban đầu (Downsampling)
        for i in range(num_strides):
            out_dim = 64 if i == 0 else (128 if i == 1 else latent_dim)
            layers.extend([
                nn.Conv2d(curr_in, out_dim, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, out_dim),
                nn.GELU()
            ])
            curr_in = out_dim

        # Khối trích xuất đặc trưng sâu bằng Modern Residual
        layers.extend([
            nn.Conv2d(curr_in, latent_dim, kernel_size=3, stride=1, padding=1),
            ModernResidualBlock(latent_dim),
            ModernResidualBlock(latent_dim),
            nn.GroupNorm(8, latent_dim),
            nn.GELU()
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)