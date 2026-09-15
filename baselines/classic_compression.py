import cv2
import numpy as np

def calculate_psnr_np(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def evaluate_jpeg_compression(img_Y_np, quality=50):
    """
    Mô phỏng Xe 2 nén ảnh bằng JPEG và gửi qua mạng.
    img_Y_np: numpy array RGB, chuẩn [0, 255], kích thước (H, W, 3)
    quality: Mức chất lượng JPEG từ 1 đến 100
    """
    # Encode (Nén)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img_Y_np, encode_param)
    
    if not result:
        return 0.0, 0.0
        
    # Tính BPP (Bits Per Pixel)
    file_size_bits = len(encimg) * 8
    bpp = file_size_bits / (img_Y_np.shape[0] * img_Y_np.shape[1])
    
    # Decode (Giải nén)
    decimg = cv2.imdecode(encimg, 1)
    
    # Tính PSNR
    psnr = calculate_psnr_np(img_Y_np, decimg)
    return bpp, psnr