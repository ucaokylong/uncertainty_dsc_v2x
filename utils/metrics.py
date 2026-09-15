import torch
import torch.nn.functional as F
from math import exp

def calculate_psnr(img1, img2, max_val=2.0):
    """
    Tính PSNR cho tensor ảnh trong miền giá trị [-1.0, 1.0].
    Khoảng cách tối đa (max_val) giữa -1 và 1 là 2.0.
    """
    with torch.no_grad():
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def calculate_ssim(img1, img2, window_size=11, max_val=2.0):
    """
    Tính Structural Similarity Index (SSIM).
    Đánh giá độ tương đồng cấu trúc, bám sát cảm nhận thị giác con người hơn PSNR.
    """
    with torch.no_grad():
        channel = img1.size(1)
        window = create_window(window_size, channel).to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = (0.01 * max_val) ** 2
        C2 = (0.03 * max_val) ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def calculate_bpp(indices, original_img_shape, num_bits=8):
    """
    Tính Bits Per Pixel (Rate).
    original_img_shape: (B, C, H, W)
    indices: (B, H_latent, W_latent)
    """
    with torch.no_grad():
        B, H_latent, W_latent = indices.shape
        _, _, H, W = original_img_shape
        total_bits = H_latent * W_latent * num_bits
        total_pixels = H * W
        bpp = total_bits / total_pixels
    return bpp