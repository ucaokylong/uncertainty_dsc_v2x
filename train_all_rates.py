import os
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm

from models.full_model import DSCV2XModel
from losses.losses import DSCV2XLoss
from data.kitti_loader import build_dataloader

CONFIG_PATH = "configs/default_config.yaml"
RATE_CONFIGS = [
    "configs/model_r_low.yaml",
    "configs/model_r_mid.yaml",
    "configs/model_r_high.yaml"
]

class EarlyStopping:
    """Monitors the loss. If it doesn't improve after 'patience' epochs, training is halted."""
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_single_rate(config, rate_config, train_loader, device):
    rate_name = rate_config["compression"]["name"]
    print(f"\n{'='*60}\nSTART TRAINING: {rate_name.upper()}\n{'='*60}")

    # Initialize model geometry (Lưu ý: Nâng base_dims=64 và nạp downsample_factor)
    model = DSCV2XModel(
        in_channels=3,
        latent_dim=64,
        num_embeddings=rate_config["compression"]["codebook_size"],
        base_dims=64, # Nâng cấp bộ nhớ mạng
        downsample_factor=rate_config["compression"]["downsample_factor"] # Kích hoạt lưới động
    ).to(device)

    # Initialize loss function
    criterion = DSCV2XLoss(
        lambda_recon=config["loss_weights"]["lambda_recon"],
        lambda_warp=config["loss_weights"]["lambda_warp"],
        lambda_unc=config["loss_weights"]["lambda_uncertainty"],
        lambda_flow=0.1,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    epochs = config["training"]["epochs"]
    erasure_range = config["channel"]["train_erasure_range"]
    bit_flip_range = config["channel"].get("train_bit_flip_range", [0.0, 0.02])
    save_dir = os.path.join(config["experiment"]["save_dir"], rate_name)
    os.makedirs(save_dir, exist_ok=True)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(config["training"]["lr"]),
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1, 
    )
    
    early_stopping = EarlyStopping(patience=15, min_delta=1e-3)

    for epoch in range(epochs):
        model.train()
        metrics = {"total": 0.0, "recon": 0.0, "warp": 0.0, "unc": 0.0, "smooth": 0.0, "vq": 0.0}
        pbar = tqdm(train_loader, desc=f"[{rate_name}] Epoch {epoch+1:03d}/{epochs:03d}")

        for img_X, img_Y in pbar:
            img_X = img_X.to(device, non_blocking=True)
            img_Y = img_Y.to(device, non_blocking=True)
            
            # Sinh ngẫu nhiên cả tỷ lệ mất gói và tỷ lệ lật bit
            current_erasure = torch.empty(1).uniform_(erasure_range[0], erasure_range[1]).item()
            current_bit_flip = torch.empty(1).uniform_(bit_flip_range[0], bit_flip_range[1]).item()

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                img_Y_hat, img_Y_prime, uncertainty_map, flow, vq_loss, _ = model(
                    img_X, img_Y, 
                    erasure_rate=current_erasure,
                    bit_flip_prob=current_bit_flip
                )
                loss, l_recon, l_warp, l_unc, l_smooth = criterion(
                    img_Y_hat, img_Y_prime, img_Y, flow, uncertainty_map, vq_loss
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Tích lũy các Loss
            metrics["total"] += loss.item()
            metrics["recon"] += l_recon.item()
            metrics["warp"] += l_warp.item()
            metrics["unc"] += l_unc.item()
            metrics["smooth"] += l_smooth.item()
            metrics["vq"] += vq_loss.item()

            # Hiển thị trên thanh chạy thời gian thực
            pbar.set_postfix({
                "Loss": f"{loss.item():.3f}",
                "Rec": f"{l_recon.item():.3f}",
                "Unc": f"{l_unc.item():.3f}",
                "LR": f"{scheduler.get_last_lr()[0]:.1e}", 
            })

        # Tổng hợp và in log ra file/màn hình vào cuối mỗi Epoch
        num_batches = len(train_loader)
        avg_loss = metrics['total'] / num_batches
        
        print(f"Epoch {epoch+1:03d} | "
              f"Tot: {avg_loss:.4f} | "
              f"Rec: {metrics['recon']/num_batches:.4f} | "
              f"Warp: {metrics['warp']/num_batches:.4f} | "
              f"Unc: {metrics['unc']/num_batches:.4f} | "
              f"Sm: {metrics['smooth']/num_batches:.4f} | "
              f"VQ: {metrics['vq']/num_batches:.4f}")

        early_stopping(avg_loss)
        if early_stopping.early_stop:
            print(f"-> Loss plateaued! Early stopping triggered at Epoch {epoch+1}.")
            break

        # 6. Lưu mô hình tốt nhất (Best Checkpoint)
        if avg_loss < early_stopping.best_loss or epoch == 0:
            best_ckpt_path = os.path.join(save_dir, "model_best.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
            }, best_ckpt_path)
            print(f"  --> Saved new BEST model with Loss: {avg_loss:.4f}")

        # 7. Lưu Checkpoint định kỳ (dự phòng)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs or early_stopping.early_stop:
            ckpt_path = os.path.join(save_dir, f"model_ep{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
            }, ckpt_path)

def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computational device: {device}")

    train_loader = build_dataloader(
        data_root=config["data"]["root"],
        split="train",
        batch_size=config["data"]["batch_size"],
        img_size=tuple(config["data"]["img_size"]),
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    for rate_path in RATE_CONFIGS:
        with open(rate_path, "r") as f:
            rate_config = yaml.safe_load(f)
        train_single_rate(config, rate_config, train_loader, device)
        
    print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    main()