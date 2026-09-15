import torch
import torch.nn as nn
import torch.nn.functional as F

class DSCV2XLoss(nn.Module):
    def __init__(self, lambda_recon=1.0, lambda_warp=0.5, lambda_unc=0.1, lambda_flow=0.1):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_warp = lambda_warp
        self.lambda_unc = lambda_unc
        self.lambda_flow = lambda_flow

    def forward(self, img_Y_hat, img_Y_prime, img_Y, flow, uncertainty_map, vq_loss):
        # 1. Reconstruction Loss (L1 để giữ cạnh sắc nét)
        loss_recon = F.l1_loss(img_Y_hat, img_Y)
        
        # 2. Uncertainty-weighted Warping Loss
        # Chỉ ép mạng học nắn ảnh tại những vùng nó TỰ TIN (U -> 0)
        # Tại vùng nó bó tay (U -> 1), đạo hàm chỗ này bị triệt tiêu bằng 0
        loss_warp = torch.mean((1.0 - uncertainty_map) * torch.abs(img_Y_prime - img_Y))
        
        # 3. Uncertainty Regularization
        # Phạt U cao để tránh mạng hội tụ về nghiệm lười (luôn dự đoán U=1 ở mọi nơi)
        loss_unc = torch.mean(uncertainty_map)
        
        # 4. Flow Smoothness Loss (Total Variation)
        dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
        dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
        loss_smooth = torch.mean(dx) + torch.mean(dy)
        
        # 5. Tổng hợp (Đã khớp với cấu hình yaml)
        loss_total = (self.lambda_recon * loss_recon + 
                      self.lambda_warp * loss_warp +
                      self.lambda_unc * loss_unc +
                      self.lambda_flow * loss_smooth +
                      vq_loss)
                      
        return loss_total, loss_recon, loss_warp, loss_unc, loss_smooth