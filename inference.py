import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image

from models.full_model import DSCV2XModel
from utils.visualization import denormalize

def get_inference_transforms(img_size=(256, 256)):
    """Biến đổi ảnh đầu vào khớp với tensor dùng lúc train"""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def save_single_output(tensor, path):
    """Lưu 1 tensor ảnh ra file đĩa cứng"""
    from torchvision.utils import save_image
    img = denormalize(tensor.squeeze(0).cpu())
    save_image(img, path)
    print(f"  [+] Đã lưu: {path}")

def main():
    parser = argparse.ArgumentParser(description="DSC-V2X Real-world Inference Tool")
    parser.add_argument("--img_X", type=str, required=True, help="Đường dẫn ảnh Side Info (Xe 1 - Cam Trái)")
    parser.add_argument("--img_Y", type=str, required=True, help="Đường dẫn ảnh Ground Truth (Xe 2 - Cam Phải)")
    parser.add_argument("--rate", type=str, choices=["low", "mid", "high"], default="low", help="Mức độ nén (low/mid/high)")
    parser.add_argument("--erasure", type=float, default=0.0, help="Tỉ lệ rớt mạng mô phỏng (0.0 đến 1.0)")
    parser.add_argument("--out_dir", type=str, default="./inference_output", help="Thư mục xuất ảnh")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Thiết lập codebook size tương ứng với mức rate
    codebook_map = {"low": 256, "mid": 512, "high": 1024}
    ckpt_path = f"checkpoints/rate_{args.rate}/model_best.pth"
    
    print(f"[*] Đang nạp mô hình hệ số: {args.rate.upper()} | Checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"[!] LỖI: Không tìm thấy file checkpoint: {ckpt_path}")
        return

    # Khởi tạo mô hình
    model = DSCV2XModel(in_channels=3, latent_dim=64, num_embeddings=codebook_map[args.rate], base_dims=32).to(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Xử lý ảnh đầu vào
    print("[*] Đang đọc và tiền xử lý ảnh...")
    transform = get_inference_transforms()
    img_X_pil = Image.open(args.img_X).convert('RGB')
    img_Y_pil = Image.open(args.img_Y).convert('RGB')
    
    # Thêm chiều Batch (B=1)
    img_X = transform(img_X_pil).unsqueeze(0).to(device)
    img_Y = transform(img_Y_pil).unsqueeze(0).to(device)

    # Chạy mô hình
    print(f"[*] Đang thực thi Inference (với mô phỏng rớt mạng: {args.erasure*100}%)...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            img_Y_hat, img_Y_prime, uncertainty_map, flow, _, _ = model(
                img_X, img_Y, erasure_rate=args.erasure
            )
            
    # Lưu toàn bộ các ảnh trung gian để phân tích
    print("[*] Đang kết xuất kết quả...")
    save_single_output(img_X, os.path.join(args.out_dir, "1_input_X_side_info.png"))
    save_single_output(img_Y, os.path.join(args.out_dir, "2_input_Y_ground_truth.png"))
    save_single_output(img_Y_prime, os.path.join(args.out_dir, "3_warped_Y_prime.png"))
    
    # Bản đồ bất định (Uncertainty) cần chuyển Heatmap
    U = uncertainty_map.squeeze(0).cpu() # [1, H, W]
    U_heatmap = U.repeat(3, 1, 1)        # Biến thành RGB đen trắng [3, H, W]
    save_single_output(U_heatmap.unsqueeze(0), os.path.join(args.out_dir, "4_uncertainty_map.png"))
    
    save_single_output(img_Y_hat, os.path.join(args.out_dir, "5_final_reconstructed_Y_hat.png"))
    print("\n[SUCCESS] Hoàn tất! Ảnh nằm trong thư mục:", args.out_dir)

if __name__ == "__main__":
    main()