import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from data.transforms import get_transforms

class KITTIRawStereoDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=(256, 256)):
        """
        data_root: Đường dẫn tới thư mục chứa các drive (ví dụ: .../data_raw/2011_09_26)
        split: 'train' (drives 0009, 0011, 0013) hoặc 'test' (drive 0014)
        """
        self.split = split
        self.transform = get_transforms(img_size=img_size)
        
        # Phân chia drive triệt để để chống rò rỉ dữ liệu chuỗi thời gian
        target_drives = ['0009', '0011', '0013'] if split == 'train' else ['0014']
        
        self.left_paths = []
        self.right_paths = []
        
        for drive_id in target_drives:
            drive_folder = f"2011_09_26_drive_{drive_id}_sync"
            left_dir = os.path.join(data_root, drive_folder, "image_02", "data")
            right_dir = os.path.join(data_root, drive_folder, "image_03", "data")
            
            # Chỉ nạp file .png, bỏ qua hoàn toàn file metadata timestamps.txt
            left_imgs = sorted(glob.glob(os.path.join(left_dir, "*.png")))
            
            for l_path in left_imgs:
                file_name = os.path.basename(l_path)
                r_path = os.path.join(right_dir, file_name)
                
                # Xác thực cặp ảnh tương ứng tồn tại
                if os.path.exists(r_path):
                    self.left_paths.append(l_path)
                    self.right_paths.append(r_path)

        if len(self.left_paths) == 0:
            raise FileNotFoundError(f"Không tìm thấy ảnh stereo hợp lệ trong: {data_root} cho split: {split}")

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        # Anchor X (Camera Trái - image_02)
        # Target Y (Camera Phải - image_03)
        img_X = Image.open(self.left_paths[idx]).convert('RGB')
        img_Y = Image.open(self.right_paths[idx]).convert('RGB')
        
        return self.transform(img_X), self.transform(img_Y)


def build_dataloader(data_root, split='train', batch_size=64, img_size=(256, 256), num_workers=4, pin_memory=True):
    """Hàm dựng DataLoader với cấu hình nạp dữ liệu đa luồng."""
    dataset = KITTIRawStereoDataset(data_root=data_root, split=split, img_size=img_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == 'train')
    )


if __name__ == "__main__":
    TEST_ROOT = "/home/s2410433/uncertainty_dsc_v2x/data_raw/2011_09_26"
    if os.path.exists(TEST_ROOT):
        train_loader = build_dataloader(TEST_ROOT, split='train', batch_size=64, num_workers=4)
        test_loader = build_dataloader(TEST_ROOT, split='test', batch_size=64, num_workers=4)

        print(f"Khởi tạo DataLoader thành công:")
        print(f"  - Số cặp ảnh Train (Drives 0009, 0011, 0013): {len(train_loader.dataset)}")
        print(f"  - Số cặp ảnh Test  (Drive 0014):              {len(test_loader.dataset)}")
        
        bx, by = next(iter(train_loader))
        print(f"  - Kích thước batch X: {bx.shape}")
        print(f"  - Kích thước batch Y: {by.shape}")
    else:
        print(f"Đường dẫn không tồn tại: {TEST_ROOT}")