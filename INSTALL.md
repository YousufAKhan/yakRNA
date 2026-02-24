# YAK RNA - Installation Guide

## System Requirements

- **OS**: Linux or macOS
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 16GB+
- **Python**: 3.10

## Installation

### Option 1: Micromamba/Conda (Recommended)

```bash
git clone https://github.com/YousufAKhan/yakRNA.git
cd yakRNA

# macOS
micromamba env create -f environment-macos.yml

# Linux (with CUDA)
micromamba env create -f environment-linux.yml

micromamba activate yakrna
```

### Option 2: pip

```bash
git clone https://github.com/YousufAKhan/yakRNA.git
cd yakRNA

python -m venv venv
source venv/bin/activate

pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Troubleshooting

### Flash Attention Error
```bash
pip install flash-attn==2.5.8
```

### xFormers Error
```bash
pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121
```

## Next Steps

See [README.md](README.md) for usage instructions.
