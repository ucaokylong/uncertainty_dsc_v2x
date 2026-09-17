#!/bin/bash
#SBATCH -p GPU-1A
#SBATCH -J DSC_V2X_Train
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH -o DSC_Train_Output-%j.log

# Di chuyển vào thư mục submit job
cd $SLURM_SUBMIT_DIR

# ==========================================
# BƯỚC 0: DỌN DẸP MÔI TRƯỜNG
# ==========================================
unset PYTHONPATH
unset PYTHONHOME

# ==========================================
# BƯỚC 1: KÍCH HOẠT CONDA
# ==========================================
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate mamba_venv

# ==========================================
# BƯỚC 2: LIÊN KẾT ĐƯỜNG DẪN CUDA & THƯ VIỆN NỘI BỘ
# ==========================================
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export SSL_CERT_FILE=$(python -m certifi 2>/dev/null)
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

export HF_HOME=$SLURM_SUBMIT_DIR/.cache
mkdir -p $HF_HOME

echo "[*] Thoi gian bat dau: $(date)"
echo "[*] Node dang chay  : $(hostname)"
echo "[*] Python dang dung: $(which python)"
nvidia-smi

# ==========================================
# BƯỚC 3: KIỂM TRA NHANH (SMOKE TEST TRÊN A100)
# ==========================================
echo "=========================================================="
echo "[CHECK] Kiem tra import PyTorch, Torchvision & Mamba tren GPU..."
echo "=========================================================="
python -c "import torch, torchvision, causal_conv1d, mamba_ssm; \
print('CUDA Ready:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0)); \
from torchvision.ops import nms; print('Torchvision NMS: OK'); \
from mamba_ssm import Mamba; print('Mamba SSM: OK')"

# ==========================================
# BƯỚC 4: CHẠY TRAINING CHÍNH THỨC
# ==========================================
echo "=========================================================="
echo "[RUNNING] DSC-V2X Multi-Rate Training..."
echo "=========================================================="

export BN_DISABLE_MALLOC_STATS=1

# Dùng -u để log in trực tiếp ra file real-time
python -u train_all_rates.py

echo "=========================================================="
echo "[SUCCESS] Toan bo thi nghiem hoan thanh luc: $(date)"