import os
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.kitti_loader import build_dataloader
from utils.metrics import calculate_psnr, calculate_ssim, calculate_bpp
from utils.visualization import denormalize

# Import các Baselines
from baselines.classic_compression import evaluate_jpeg_compression
from baselines.standard_vqvae import StandardVQVAE
from baselines.naive_fusion import NaiveFusionVQVAE
from baselines.raft_fusion import RAFTFusionBaseline

CONFIG_PATH = "configs/default_config.yaml"
BASELINE_CKPT_DIR = "checkpoints/baselines" 

def evaluate_jpeg(test_loader):
    print(f"\n{'='*50}\nEVALUATING: BASELINE 1 - JPEG\n{'='*50}")
    qualities = [10, 30, 50, 70, 90]
    results = []
    
    for q in qualities:
        metrics = {"psnr": 0.0, "bpp": 0.0}
        pbar = tqdm(test_loader, desc=f"JPEG Quality {q}")
        
        for _, img_Y in pbar:
            batch_Y_np = (denormalize(img_Y).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            b_psnr, b_bpp = 0.0, 0.0
            
            for i in range(batch_Y_np.shape[0]):
                bpp, psnr = evaluate_jpeg_compression(batch_Y_np[i], quality=q)
                b_psnr += psnr
                b_bpp += bpp
                
            metrics["psnr"] += b_psnr / batch_Y_np.shape[0]
            metrics["bpp"] += b_bpp / batch_Y_np.shape[0]
            
        num_batches = len(test_loader)
        avg_psnr = metrics["psnr"] / num_batches
        avg_bpp = metrics["bpp"] / num_batches
        
        print(f"  -> BPP: {avg_bpp:.4f} | PSNR: {avg_psnr:.2f} dB")
        results.append({
            "Model": "JPEG",
            "Condition": f"Quality_{q}",
            "Erasure(%)": 0,
            "Bit_Flip(%)": 0.0,
            "Eff_BPP": round(avg_bpp, 4),
            "PSNR": round(avg_psnr, 2),
            "SSIM": 0.0 
        })
    return pd.DataFrame(results)

def evaluate_neural_baseline(model, model_name, ckpt_name, test_loader, device, erasure_rates, bit_flip_probs):
    print(f"\n{'='*50}\nEVALUATING: BASELINE - {model_name.upper()}\n{'='*50}")
    
    ckpt_path = os.path.join(BASELINE_CKPT_DIR, ckpt_name)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model_state_dict"])
        print(f"  [+] Loaded weights from {ckpt_path}")
    else:
        print(f"  [!] WARNING: Khong tim thay {ckpt_path}. Chay voi random weights!")
        
    model.to(device)
    model.eval()
    results = []

    # Quét lưới 2D như model chính
    for erasure in erasure_rates:
        for bit_flip in bit_flip_probs:
            metrics = {"psnr": 0.0, "ssim": 0.0, "bpp": 0.0}
            pbar = tqdm(test_loader, desc=f"Test [E={erasure*100:.0f}%, BF={bit_flip*100:.1f}%]")
            
            with torch.no_grad():
                for img_X, img_Y in pbar:
                    img_X, img_Y = img_X.to(device), img_Y.to(device)
                    
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        if model_name == "Standard_VQVAE":
                            out = model(img_Y, erasure_rate=erasure, bit_flip_prob=bit_flip)
                        else: 
                            out = model(img_X, img_Y, erasure_rate=erasure, bit_flip_prob=bit_flip)
                    
                    img_Y_hat = out[0]
                    indices = out[2]
                    
                    img_Y_f, img_Y_hat_f = img_Y.float(), img_Y_hat.float()

                    metrics["psnr"] += calculate_psnr(img_Y_hat_f, img_Y_f, max_val=2.0)
                    metrics["ssim"] += calculate_ssim(img_Y_hat_f, img_Y_f, max_val=2.0)
                    metrics["bpp"] += calculate_bpp(indices, img_X.shape, num_bits=8) * (1.0 - erasure)

            num_batches = len(test_loader)
            avg_psnr = metrics["psnr"] / num_batches
            avg_ssim = metrics["ssim"] / num_batches
            avg_bpp = metrics["bpp"] / num_batches
            
            results.append({
                "Model": model_name,
                "Condition": f"E_{int(erasure*100)}%_BF_{bit_flip*100:.1f}%",
                "Erasure(%)": int(erasure * 100),
                "Bit_Flip(%)": bit_flip * 100,
                "Eff_BPP": round(avg_bpp, 4),
                "PSNR": round(avg_psnr, 2),
                "SSIM": round(avg_ssim, 4)
            })
            
    return pd.DataFrame(results)

def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    erasure_rates = config["channel"]["eval_erasure_rates"]
    bit_flip_probs = config["channel"].get("eval_bit_flip_probs", [0.0, 0.01, 0.02])
    
    test_loader = build_dataloader(
        data_root=config["data"]["root"], split="test", batch_size=16, 
        img_size=tuple(config["data"]["img_size"]), num_workers=4
    )

    all_results = []
    
    # 1. JPEG Baseline (JPEG không giả lập bit flip được vì làm hỏng cấu trúc file tĩnh)
    df_jpeg = evaluate_jpeg(test_loader)
    all_results.append(df_jpeg)
    
    # 2. Standard VQ-VAE Baseline
    model_vq = StandardVQVAE(in_channels=3, latent_dim=64, num_embeddings=256, base_dims=64)
    df_vq = evaluate_neural_baseline(model_vq, "Standard_VQVAE", "vqvae_best.pth", test_loader, device, erasure_rates, bit_flip_probs)
    all_results.append(df_vq)
    
    # 3. Naive Fusion Baseline
    model_naive = NaiveFusionVQVAE(in_channels=3, latent_dim=64, num_embeddings=256, base_dims=64)
    df_naive = evaluate_neural_baseline(model_naive, "Naive_Fusion", "naive_best.pth", test_loader, device, erasure_rates, bit_flip_probs)
    all_results.append(df_naive)
    
    # 4. RAFT Fusion Baseline
    model_raft = RAFTFusionBaseline(in_channels=3, latent_dim=64, num_embeddings=256, base_dims=64)
    df_raft = evaluate_neural_baseline(model_raft, "RAFT_Late_Fusion", "raft_best.pth", test_loader, device, erasure_rates, bit_flip_probs)
    all_results.append(df_raft)
    
    # Tổng hợp báo cáo
    final_report = pd.concat(all_results, ignore_index=True)
    print("\n" + "="*90)
    print("BÁO CÁO KẾT QUẢ CÁC MÔ HÌNH BASELINE (TEST SET)")
    print("="*90)
    print(final_report.to_string(index=False))
    
    final_report.to_csv("baselines_evaluation_results.csv", index=False)
    print("\n[SUCCESS] Đã lưu kết quả dạng bảng vào: baselines_evaluation_results.csv")

if __name__ == "__main__":
    main()