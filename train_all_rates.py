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
    print(f"\n{'='*50}\nSTART TRAINING: {rate_name.upper()}\n{'='*50}")

    # Initialize model geometry based on compression rate configuration
    model = DSCV2XModel(
        in_channels=3,
        latent_dim=64,
        num_embeddings=rate_config["compression"]["codebook_size"],
        base_dims=32,
    ).to(device)

    # Initialize loss function with YAML weights
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
    save_dir = os.path.join(config["experiment"]["save_dir"], rate_name)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Initialize Scheduler (OneCycleLR) and AMP Scaler for A100/A40 optimization
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(config["training"]["lr"]),
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1, # Allocate the first 10% of epochs for learning rate warm-up
    )
    scaler = torch.cuda.amp.GradScaler()
    
    # 2. Initialize Early Stopping to prevent overfitting
    early_stopping = EarlyStopping(patience=15, min_delta=1e-3)

    for epoch in range(epochs):
        model.train()
        metrics = {"total": 0.0, "recon": 0.0, "warp": 0.0, "unc": 0.0, "smooth": 0.0}
        pbar = tqdm(train_loader, desc=f"[{rate_name}] Epoch {epoch+1}/{epochs}")

        for img_X, img_Y in pbar:
            # Transfer data to GPU asynchronously
            img_X = img_X.to(device, non_blocking=True)
            img_Y = img_Y.to(device, non_blocking=True)
            
            # Sample a random erasure rate uniformly for the current batch
            current_erasure = torch.empty(1).uniform_(erasure_range[0], erasure_range[1]).item()

            optimizer.zero_grad()

            # 3. Enable Automatic Mixed Precision (AMP) for the forward pass
            with torch.cuda.amp.autocast():
                img_Y_hat, img_Y_prime, uncertainty_map, flow, vq_loss, _ = model(
                    img_X, img_Y, erasure_rate=current_erasure
                )
                loss, l_recon, l_warp, l_unc, l_smooth = criterion(
                    img_Y_hat, img_Y_prime, img_Y, flow, uncertainty_map, vq_loss
                )

            # 4. Execute Backward pass via GradScaler
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping to ensure max_norm is evaluated correctly
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step optimizer and update scaler multipliers
            scaler.step(optimizer)
            scaler.update()
            
            # Step the OneCycleLR scheduler per batch (NOT per epoch)
            scheduler.step()

            metrics["total"] += loss.item()
            metrics["recon"] += l_recon.item()
            metrics["warp"] += l_warp.item()
            metrics["unc"] += l_unc.item()
            metrics["smooth"] += l_smooth.item()

            # Update progress bar statistics in real-time
            pbar.set_postfix({
                "T_Loss": f"{loss.item():.3f}",
                "LR": f"{scheduler.get_last_lr()[0]:.2e}", 
            })

        num_batches = len(train_loader)
        epoch_avg_loss = metrics['total'] / num_batches
        print(f"Epoch {epoch+1:03d} | Total: {epoch_avg_loss:.4f} | Recon: {metrics['recon']/num_batches:.4f} | Unc: {metrics['unc']/num_batches:.4f}")

        # 5. Check Early Stopping condition
        early_stopping(epoch_avg_loss)
        if early_stopping.early_stop:
            print(f"-> Loss plateaued! Early stopping triggered at Epoch {epoch+1}.")
            break

        # Save checkpoint periodically or if training ends
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs or early_stopping.early_stop:
            ckpt_path = os.path.join(save_dir, f"model_ep{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict()
            }, ckpt_path)

def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computational device: {device}")

    # Build the real dataset dataloader
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