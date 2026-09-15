import numpy as np

def wyner_ziv_bound(distortion, correlation_rho, variance=1.0):
    """
    Tính toán giới hạn Rate-Distortion lý thuyết của Wyner-Ziv
    cho nguồn Gauss có độ tương quan (correlation_rho).
    Công thức: R_WZ(D) = 0.5 * log2((sigma^2 * (1 - rho^2)) / D)
    """
    # Nếu méo cho phép (distortion) lớn hơn cả nhiễu dư thừa, không cần truyền gì (Rate = 0)
    if distortion >= variance * (1.0 - correlation_rho**2):
        return 0.0
        
    rate = 0.5 * np.log2((variance * (1.0 - correlation_rho**2)) / (distortion + 1e-9))
    return max(0.0, rate)

def shannon_lower_bound(distortion, variance=1.0):
    """
    Giới hạn nén độc lập (Không có Side Information từ Xe 1).
    Dùng để so sánh độ chênh lệch băng thông với Wyner-Ziv.
    """
    if distortion >= variance:
        return 0.0
    return 0.5 * np.log2(variance / (distortion + 1e-9))