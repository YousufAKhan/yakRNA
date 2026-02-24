# YAK RNA: RNA Sequence Generation

A discrete diffusion model for conditional RNA sequence generation. Generate RNA sequences conditioned on secondary structure, consensus sequences, or GO terms.

## Features

- **Conditional Generation**: Structure-guided and GO term-conditioned generation
- **Multiple Generation Modes**:
  - Unconditional (length-based)
  - Structure-conditioned (dot-bracket notation)
  - Consensus-conditioned
  - GO term-conditioned
  - Sequence infilling
- **Interactive CLI**: Easy-to-use command-line interface

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

## Quick Start

### Download Pre-trained Model

*Instructions coming soon*

### Generate Sequences

```bash
# Basic generation
python inference/rna_sequence_generator.py \
    --config configs/inference.yaml \
    --checkpoint path/to/model.pt \
    --num_sequences 10 \
    --length 100

# Interactive mode
python inference/interactive_generator.py \
    --config configs/inference.yaml \
    --checkpoint path/to/model.pt
```

### Generation Options

```bash
# Structure-conditioned generation
python inference/rna_sequence_generator.py \
    --config configs/inference.yaml \
    --checkpoint path/to/model.pt \
    --secondary_structure "((((....))))" \
    --num_sequences 5

# With temperature control
python inference/rna_sequence_generator.py \
    --config configs/inference.yaml \
    --checkpoint path/to/model.pt \
    --temperature 0.8 \
    --num_sequences 10
```

## Project Structure

```
yakRNA/
├── inference/
│   ├── rna_sequence_generator.py   # Main generation script
│   ├── interactive_generator.py    # Interactive CLI
│   └── discrete_diffusion.py       # Diffusion components
├── configs/
│   └── inference.yaml              # Model config
├── training_data/processed/vocabulary_analysis/
│   └── unified_vocabulary.json     # Vocabulary file
├── environment-macos.yml
├── environment-linux.yml
├── requirements.txt
└── README.md
```

## System Requirements

- **OS**: Linux or macOS
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 16GB+
- **Python**: 3.10

## License

MIT License - see [LICENSE](LICENSE)

## Citation

If you use YAK RNA in your research, please cite:

```bibtex
@software{yakrna2025,
  author = {Khan, Yousuf},
  title = {YAK RNA: Discrete Diffusion for RNA Sequence Generation},
  year = {2025},
  url = {https://github.com/YousufAKhan/yakRNA}
}
```

## Contact

- **Author**: Yousuf Khan
- **GitHub**: [@YousufAKhan](https://github.com/YousufAKhan)
