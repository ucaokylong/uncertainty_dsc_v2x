import os
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm

from models.full_model import DSCV2XModel
from losses.losses import DSCV2XLoss

# --- CẤU HÌNH ĐƯỜNG DẪN TRỰC TIẾP ---
CONFIG_PATH = "configs/default_config.yaml"
RATE_CONFIG_PATH = "configs/model_r_mid.yaml"

def main():
    # 1. Đọc file cấu hình
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    with open(RATE_CONFIG_PATH, 'r') as f:
        rate_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Khởi tạo mô hình và chuyển lên GPU
    model = DSCV2XModel(
        latent_dim=rate_config['compression']['downsample_factor'] * 16,
        num_embeddings=rate_config['compression']['codebook_size']
    ).to(device)
    
    # 3. Khởi tạo Loss và Optimizer
    criterion = DSCV2XLoss(
        lambda_recon=config['loss_weights']['lambda_recon'],
        lambda_warp=config['loss_weights']['lambda_warp'],
        lambda_unc=config['loss_weights']['lambda_uncertainty']
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(config['training']['lr']), 
        weight_decay=float(config['training']['weight_decay'])
    )

    # TODO: Khởi tạo Dataloader thực tế (VD: KITTIDataset)
    # train_loader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=True)
    
    epochs = config['training']['epochs']
    erasure_range = config['channel']['train_erasure_range']
    save_dir = config['experiment']['save_dir']
    
    # 4. Vòng lặp huấn luyện
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Thay thế range(100) bằng train_loader khi có dữ liệu thật
        for batch_idx in tqdm(range(100), desc=f"Epoch {epoch+1}/{epochs}"):
            # Dữ liệu giả định trong khoảng [-1.0, 1.0]
            img_X = torch.rand(4, 3, 256, 256).to(device) * 2 - 1 
            img_Y = torch.rand(4, 3, 256, 256).to(device) * 2 - 1
            
            # Lấy ngẫu nhiên tỷ lệ nhiễu cho từng batch để tăng tính chống chịu
            current_erasure = torch.empty(1).uniform_(erasure_range[0], erasure_range[1]).item()
            
            optimizer.zero_grad()
            
            img_Y_hat, img_Y_prime, uncertainty_map, flow, vq_loss, _ = model(
                img_X, img_Y, erasure_rate=current_erasure
            )
            
            loss, l_recon, l_warp, l_unc, l_smooth = criterion(
                img_Y_hat, img_Y_prime, img_Y, flow, uncertainty_map, vq_loss
            )
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Total Loss: {epoch_loss/100:.4f}")
        
        # 5. Lưu trọng số định kỳ
        if (epoch + 1) % 10 == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/dsc_model_ep{epoch+1}.pth")

if __name__ == "__main__":
    main()