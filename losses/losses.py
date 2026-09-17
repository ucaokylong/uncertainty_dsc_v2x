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
        # 1. Reconstruction Loss: Kết hợp MSE (Kéo PSNR) và L1 (Giữ độ sắc nét)
        loss_mse = F.mse_loss(img_Y_hat, img_Y)
        loss_l1 = F.l1_loss(img_Y_hat, img_Y)
        loss_recon = 0.5 * loss_mse + 0.5 * loss_l1
        
        # 2. Uncertainty-weighted Warping Loss
        loss_warp = torch.mean((1.0 - uncertainty_map) * torch.abs(img_Y_prime - img_Y))
        
        # 3. Uncertainty Regularization
        loss_unc = torch.mean(uncertainty_map)
        
        # 4. Flow Smoothness Loss (Total Variation)
        dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
        dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
        loss_smooth = torch.mean(dx) + torch.mean(dy)
        
        # 5. Tổng hợp
        loss_total = (self.lambda_recon * loss_recon + 
                      self.lambda_warp * loss_warp +
                      self.lambda_unc * loss_unc +
                      self.lambda_flow * loss_smooth +
                      vq_loss)
                      
        return loss_total, loss_recon, loss_warp, loss_unc, loss_smooth