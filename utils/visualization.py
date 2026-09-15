import torch
import torchvision
import matplotlib.pyplot as plt
import os

def denormalize(tensor):
    """Chuyển tensor từ [-1, 1] về [0, 1] để lưu thành ảnh."""
    return (tensor + 1.0) / 2.0

def save_comparison_grid(img_X, img_Y, img_Y_prime, img_Y_hat, uncertainty_map, epoch, batch_idx, save_dir):
    """
    Xuất một lưới ảnh so sánh để đưa vào bài báo học thuật.
    Cột: Xe 1 (Side Info), Xe 2 (Ground Truth), Ảnh nắn (Warped), Khôi phục (Reconstructed), Bản đồ bất định (Uncertainty).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Lấy sample đầu tiên trong batch
    X = denormalize(img_X[0].cpu())
    Y = denormalize(img_Y[0].cpu())
    Y_prime = denormalize(img_Y_prime[0].cpu())
    Y_hat = denormalize(img_Y_hat[0].cpu())
    
    # Uncertainty map (1 kênh) -> lặp thành 3 kênh để ghép lưới cùng ảnh màu
    U = uncertainty_map[0].cpu()
    U_heatmap = U.repeat(3, 1, 1) 
    
    # Ghép 5 ảnh thành 1 hàng ngang
    grid = torchvision.utils.make_grid([X, Y, Y_prime, Y_hat, U_heatmap], nrow=5, padding=2, normalize=False)
    
    # Lưu file
    file_path = os.path.join(save_dir, f"epoch_{epoch:03d}_batch_{batch_idx:04d}.png")
    torchvision.utils.save_image(grid, file_path)