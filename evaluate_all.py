import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.full_model import DSCV2XModel
from data.kitti_loader import build_dataloader
from utils.metrics import calculate_psnr, calculate_ssim, calculate_bpp
from utils.it_bounds import wyner_ziv_bound, shannon_lower_bound
from utils.visualization import save_comparison_grid

CONFIG_PATH = "configs/default_config.yaml"
RATE_CONFIGS = [
    "configs/model_r_low.yaml",
    "configs/model_r_mid.yaml",
    "configs/model_r_high.yaml"
]

def compute_correlation(X, Y):
    x_flat = X.contiguous().view(-1)
    y_flat = Y.contiguous().view(-1)
    vx = x_flat - torch.mean(x_flat)
    vy = y_flat - torch.mean(y_flat)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return corr.item()

def load_best_model(model, save_dir):
    best_path = os.path.join(save_dir, "model_best.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  [+] Loaded BEST model (Epoch {checkpoint.get('epoch', 'N/A')})")
        return model
    else:
        raise FileNotFoundError(f"Không tìm thấy model_best.pth tại {save_dir}")

def evaluate_single_rate(config, rate_config, test_loader, device):
    rate_name = rate_config["compression"]["name"]
    codebook_size = rate_config["compression"]["codebook_size"]
    downsample_factor = rate_config["compression"]["downsample_factor"]
    print(f"\n{'='*60}\nEVALUATING: {rate_name.upper()} (K={codebook_size}, Factor={downsample_factor})\n{'='*60}")

    save_dir = os.path.join(config["experiment"]["save_dir"], rate_name)
    vis_dir = os.path.join(save_dir, "visualizations")
    
    # Cập nhật base_dims=64 và downsample_factor động
    model = DSCV2XModel(
        in_channels=3, latent_dim=64, num_embeddings=codebook_size, 
        base_dims=64, downsample_factor=downsample_factor
    ).to(device)
    
    try:
        model = load_best_model(model, save_dir)
    except FileNotFoundError as e:
        print(f"  [-] BỎ QUA {rate_name}: {e}")
        return None

    model.eval()
    eval_erasure_rates = config["channel"]["eval_erasure_rates"]
    eval_bit_flip_probs = config["channel"].get("eval_bit_flip_probs", [0.0, 0.01, 0.02])
    results = []

    # Quét qua mạng lưới: Erasure x Bit Flip
    for erasure in eval_erasure_rates:
        for bit_flip in eval_bit_flip_probs:
            metrics = {"psnr": 0.0, "ssim": 0.0, "bpp": 0.0, "mse": 0.0, "var_Y": 0.0, "corr": 0.0}
            pbar = tqdm(test_loader, desc=f"Test [E={erasure*100:.0f}%, BF={bit_flip*100:.1f}%]")
            
            with torch.no_grad():
                for batch_idx, (img_X, img_Y) in enumerate(pbar):
                    img_X, img_Y = img_X.to(device), img_Y.to(device)
                    
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        img_Y_hat, img_Y_prime, uncertainty_map, _, _, indices = model(
                            img_X, img_Y, erasure_rate=erasure, bit_flip_prob=bit_flip
                        )
                    
                    img_Y_f, img_Y_hat_f = img_Y.float(), img_Y_hat.float()

                    metrics["psnr"] += calculate_psnr(img_Y_hat_f, img_Y_f, max_val=2.0)
                    metrics["ssim"] += calculate_ssim(img_Y_hat_f, img_Y_f, max_val=2.0)
                    
                    num_bits = 8 if codebook_size <= 256 else (9 if codebook_size <= 512 else 10)
                    effective_bpp = calculate_bpp(indices, img_X.shape, num_bits=num_bits) * (1.0 - erasure)
                    metrics["bpp"] += effective_bpp

                    metrics["mse"] += torch.mean((img_Y_hat_f - img_Y_f) ** 2).item()
                    metrics["var_Y"] += torch.var(img_Y_f).item()
                    metrics["corr"] += compute_correlation(img_X.float(), img_Y_f)
                    
                    # Chỉ lưu ảnh trực quan khi không có bit flip để dễ phân tích trực quan rớt gói
                    if batch_idx == 0 and bit_flip == 0.0:
                        save_comparison_grid(
                            img_X, img_Y, img_Y_prime, img_Y_hat, uncertainty_map,
                            epoch=int(erasure*100), batch_idx=0, save_dir=vis_dir
                        )

            num_batches = len(test_loader)
            avg_psnr = metrics["psnr"] / num_batches
            avg_bpp = metrics["bpp"] / num_batches
            
            avg_mse = metrics["mse"] / num_batches
            avg_var = metrics["var_Y"] / num_batches
            avg_corr = metrics["corr"] / num_batches
            
            bound_wz = wyner_ziv_bound(avg_mse, avg_corr, avg_var)
            bound_shannon = shannon_lower_bound(avg_mse, avg_var)

            results.append({
                "Rate_Profile": rate_name,
                "Erasure(%)": int(erasure * 100),
                "Bit_Flip(%)": bit_flip * 100,
                "Eff_BPP": round(avg_bpp, 4),
                "PSNR": round(avg_psnr, 2),
                "SSIM": round(metrics["ssim"] / num_batches, 4),
                "Bound_WZ(BPP)": round(bound_wz, 4),
                "Bound_Shannon(BPP)": round(bound_shannon, 4)
            })

    return pd.DataFrame(results)

def plot_paper_figures(df):
    # Biểu đồ 1: PSNR vs Erasure (Cố định Bit Flip = 0%)
    plt.figure(figsize=(10, 6))
    df_erasure = df[df['Bit_Flip(%)'] == 0.0]
    for rate_name, group in df_erasure.groupby("Rate_Profile"):
        plt.plot(group["Erasure(%)"], group["PSNR"], marker='o', linewidth=2, label=rate_name.upper())
    plt.title("Performance under Packet Erasure (0% Bit Flip)", fontsize=14)
    plt.xlabel("Erasure Rate (%)", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("paper_figure_erasure.png", dpi=300)
    
    # Biểu đồ 2: PSNR vs Bit Flip (Cố định Erasure = 0%)
    plt.figure(figsize=(10, 6))
    df_bitflip = df[df['Erasure(%)'] == 0]
    for rate_name, group in df_bitflip.groupby("Rate_Profile"):
        plt.plot(group["Bit_Flip(%)"], group["PSNR"], marker='s', linewidth=2, linestyle='--', label=rate_name.upper())
    plt.title("Performance under Bit Flip Noise (0% Erasure)", fontsize=14)
    plt.xlabel("Bit Flip Probability (%)", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("paper_figure_bitflip.png", dpi=300)
    
    print("  [+] Đã xuất 2 biểu đồ: paper_figure_erasure.png và paper_figure_bitflip.png")

def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}")

    test_loader = build_dataloader(
        data_root=config["data"]["root"], split="test", batch_size=16, 
        img_size=tuple(config["data"]["img_size"]), num_workers=4, pin_memory=True
    )

    all_dfs = []
    for rate_path in RATE_CONFIGS:
        with open(rate_path, "r") as f:
            rate_config = yaml.safe_load(f)
        df = evaluate_single_rate(config, rate_config, test_loader, device)
        if df is not None:
            all_dfs.append(df)
            
    if all_dfs:
        final_report = pd.concat(all_dfs, ignore_index=True)
        print("\n" + "="*90)
        print("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (ERASURE & BIT FLIP) VÀ IT BOUNDS")
        print("="*90)
        print(final_report.to_string(index=False))
        
        final_report.to_csv("paper_evaluation_results.csv", index=False)
        plot_paper_figures(final_report)

if __name__ == "__main__":
    main()