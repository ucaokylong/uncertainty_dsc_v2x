import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from data.kitti_loader import build_dataloader

# Import các Baselines
from baselines.standard_vqvae import StandardVQVAE
from baselines.naive_fusion import NaiveFusionVQVAE
from baselines.raft_fusion import RAFTFusionBaseline

CONFIG_PATH = "configs/default_config.yaml"
BASELINE_CKPT_DIR = "checkpoints/baselines"

class EarlyStopping:
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

def compute_reconstruction_loss(pred, target):
    """Kết hợp L1 (giữ viền sắc nét) và MSE (Tối ưu chỉ số PSNR)"""
    return 0.5 * F.mse_loss(pred, target) + 0.5 * F.l1_loss(pred, target)

def train_baseline(model, model_name, ckpt_name, config, train_loader, device):
    print(f"\n{'='*60}\nSTART TRAINING BASELINE: {model_name.upper()}\n{'='*60}")
    os.makedirs(BASELINE_CKPT_DIR, exist_ok=True)

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(config["training"]["lr"]), 
        weight_decay=float(config["training"]["weight_decay"])
    )
    
    epochs = config["training"]["epochs"]
    erasure_range = config["channel"]["train_erasure_range"]
    bit_flip_range = config["channel"].get("train_bit_flip_range", [0.0, 0.02])

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=float(config["training"]["lr"]), 
        steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.1
    )
    
    early_stopping = EarlyStopping(patience=10, min_delta=1e-3)

    # Khởi tạo Uniform để chống Codebook Collapse cho Baseline
    if hasattr(model, 'quantizer'):
        model.quantizer.embedding.weight.data.uniform_(-1.0 / model.quantizer.K, 1.0 / model.quantizer.K)

    for epoch in range(epochs):
        model.train()
        metrics = {"total": 0.0, "recon": 0.0, "vq": 0.0, "aux": 0.0}
        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1:03d}/{epochs:03d}")

        for img_X, img_Y in pbar:
            img_X = img_X.to(device, non_blocking=True)
            img_Y = img_Y.to(device, non_blocking=True)
            
            # Mô phỏng cả BEC (Mất gói) và BSC (Lật bit) trong quá trình train
            current_erasure = torch.empty(1).uniform_(erasure_range[0], erasure_range[1]).item()
            current_bit_flip = torch.empty(1).uniform_(bit_flip_range[0], bit_flip_range[1]).item()

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Truyền cả 2 tham số lỗi kênh truyền vào Baseline
                if model_name == "Standard_VQVAE":
                    out = model(img_Y, erasure_rate=current_erasure, bit_flip_prob=current_bit_flip)
                else:
                    out = model(img_X, img_Y, erasure_rate=current_erasure, bit_flip_prob=current_bit_flip)
                
                img_Y_hat = out[0]
                vq_loss = out[1]
                
                # Tính Loss chính
                loss_recon = compute_reconstruction_loss(img_Y_hat, img_Y)
                loss_total = loss_recon + vq_loss
                loss_aux = torch.tensor(0.0, device=device)

                # Auxiliary Loss bắt buộc cho nhánh giải mã độc lập của RAFT
                if model_name == "RAFT_Late_Fusion" and len(out) > 3:
                    img_Y_indep = out[3]
                    loss_aux = compute_reconstruction_loss(img_Y_indep, img_Y)
                    loss_total += loss_aux

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            metrics["total"] += loss_total.item()
            metrics["recon"] += loss_recon.item()
            metrics["vq"] += vq_loss.item()
            metrics["aux"] += loss_aux.item()

            pbar.set_postfix({
                "Loss": f"{loss_total.item():.3f}",
                "Rec": f"{loss_recon.item():.3f}",
                "Aux": f"{loss_aux.item():.3f}" if loss_aux.item() > 0 else "-"
            })

        num_batches = len(train_loader)
        avg_loss = metrics['total'] / num_batches
        print(f"Epoch {epoch+1:03d} | Tot: {avg_loss:.4f} | Rec: {metrics['recon']/num_batches:.4f} | VQ: {metrics['vq']/num_batches:.4f}")

        early_stopping(avg_loss)
        if avg_loss < early_stopping.best_loss or epoch == 0:
            best_ckpt_path = os.path.join(BASELINE_CKPT_DIR, ckpt_name)
            torch.save({
                "epoch": epoch + 1, 
                "model_state_dict": model.state_dict(), 
                "loss": avg_loss
            }, best_ckpt_path)
            print(f"  --> Saved BEST {model_name} Checkpoint!")

        if early_stopping.early_stop:
            print(f"-> Loss plateaued. Early stopping at Epoch {epoch+1}.")
            break

def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Baselines on: {device}")

    train_loader = build_dataloader(
        data_root=config["data"]["root"], split="train", 
        batch_size=config["data"]["batch_size"], 
        img_size=tuple(config["data"]["img_size"]), 
        num_workers=config["data"]["num_workers"], pin_memory=True
    )

    # Khởi tạo với base_dims=64 tương ứng mô hình chính
    model_vq = StandardVQVAE(in_channels=3, latent_dim=64, num_embeddings=256, base_dims=64).to(device)
    train_baseline(model_vq, "Standard_VQVAE", "vqvae_best.pth", config, train_loader, device)

    model_naive = NaiveFusionVQVAE(in_channels=3, latent_dim=64, num_embeddings=256, base_dims=64).to(device)
    train_baseline(model_naive, "Naive_Fusion", "naive_best.pth", config, train_loader, device)

    model_raft = RAFTFusionBaseline(in_channels=3, latent_dim=64, num_embeddings=256, base_dims=64).to(device)
    train_baseline(model_raft, "RAFT_Late_Fusion", "raft_best.pth", config, train_loader, device)

    print("\n[SUCCESS] Đã huấn luyện xong toàn bộ các Baseline Neural Networks!")

if __name__ == "__main__":
    main()