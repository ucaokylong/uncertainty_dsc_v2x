import torchvision.transforms as transforms

def get_transforms(img_size=(256, 256)):
    """
    Chuẩn hóa ảnh về kích thước vuông cố định cho VQ-VAE và đưa giá trị pixel về dải [-1, 1].
    Toàn bộ khung nhìn (FOV) của ảnh stereo được giữ nguyên.
    """
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def denormalize(tensor):
    """Đưa tensor từ [-1, 1] về [0, 1] để đo lường PSNR/SSIM và trực quan hóa."""
    return (tensor * 0.5 + 0.5).clamp(0.0, 1.0)