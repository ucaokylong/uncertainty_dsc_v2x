import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def denormalize(tensor):
    """Chuyển tensor từ [-1, 1] về [0, 1] để lưu thành ảnh."""
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)

def save_comparison_grid(img_X, img_Y, img_Y_prime, img_Y_hat, uncertainty_map, epoch, batch_idx, save_dir):
    """
    Lưu 5 ảnh tách biệt vào một thư mục riêng để tiện chèn vào bài báo.
    """
    # Tạo thư mục con cho từng mẫu để không bị lẫn lộn
    sample_dir = os.path.join(save_dir, f"erasure_{epoch:03d}_batch_{batch_idx:04d}")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Lấy sample đầu tiên trong batch và denormalize
    X = denormalize(img_X[0].cpu())
    Y = denormalize(img_Y[0].cpu())
    Y_prime = denormalize(img_Y_prime[0].cpu())
    Y_hat = denormalize(img_Y_hat[0].cpu())
    
    # Xử lý Uncertainty Map thành Heatmap (Bản đồ nhiệt màu)
    U = uncertainty_map[0, 0].cpu().numpy() # Lấy kênh đơn [H, W]
    U = np.clip(U, 0, 1)
    cmap = cm.get_cmap('jet') # Dùng thang màu Jet (Xanh -> Đỏ)
    U_color = cmap(U)[..., :3] # Lấy RGB, bỏ kênh Alpha
    U_heatmap = torch.from_numpy(U_color).permute(2, 0, 1).float()
    
    # Lưu từng file ảnh riêng lẻ
    torchvision.utils.save_image(X, os.path.join(sample_dir, "01_X_SideInfo.png"))
    torchvision.utils.save_image(Y, os.path.join(sample_dir, "02_Y_GroundTruth.png"))
    torchvision.utils.save_image(Y_prime, os.path.join(sample_dir, "03_Y_prime_Warped.png"))
    torchvision.utils.save_image(U_heatmap, os.path.join(sample_dir, "04_U_Uncertainty.png"))
    torchvision.utils.save_image(Y_hat, os.path.join(sample_dir, "05_Y_hat_Reconstructed.png"))