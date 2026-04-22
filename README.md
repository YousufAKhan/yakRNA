# yakRNA Design: A semantic multimodal RNA composer

<p align="center">
  <img src="mascot.png" alt="YAK RNA Mascot" width="200"/>
</p>

An RNA designer. Generate RNA sequences conditioned on nothing or a combination of secondary structure, consensus sequences, or gene ontology terms

## Features

- **Conditional Generation**: Structure-guided and GO term-conditioned generation
- **Multiple Generation Modes**:
  - Unconditional (length-based)
  - Structure-conditioned (dot-bracket notation)
  - Consensus-conditioned
  - GO term-conditioned
  - Sequence infilling
- **Command-line interface**: Easy-to-use CLI for batch generation

## Try it in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YousufAKhan/yakRNA/blob/main/yakRNA_colab.ipynb)

No installation required — runs in your browser with free GPU access.

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

The model weights are hosted on Hugging Face. Download via the CLI:

```bash
pip install huggingface_hub
huggingface-cli download MasterYster/yakRNA-Design yakRNA_110M.pt --local-dir checkpoints/
```

Or in Python:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="MasterYster/yakRNA-Design", filename="yakRNA_110M.pt", local_dir="checkpoints/")
```

Model page: https://huggingface.co/MasterYster/yakRNA-Design

### Generate Sequences

```bash
# Basic generation
python inference/rna_sequence_generator.py \
    --config configs/inference.yaml \
    --checkpoint checkpoints/yakRNA_110M.pt \
    --num_sequences 10 \
    --length 100
```

### Single Modality

```bash
# Unconditional (length-based)
python inference/rna_sequence_generator.py \
    --config configs/inference.yaml \
    --checkpoint checkpoints/yakRNA_110M.pt \
    --num_sequences 100 \
    --length 78 \
    --temperature 1.0 \
    --fasta_output random_sequences.fasta

# Secondary structure only
python inference/rna_sequence_generator.py \
    --config configs/inference.yaml \
    --checkpoint checkpoints/yakRNA_110M.pt \
    --num_sequences 100 \
    --secondary_structure ":::::::<<<<<<<<<---[[[[[-->>>>>>>>><<<<<<<<<<_________>>>->>>>>>>::::]]]]]::::" \
    --constraint_set canonical \
    --temperature 1.0 \
    --fasta_output ss_prompt.fasta

# GO term only
python inference/rna_sequence_generator.py \
    --config configs/inference.yaml \
    --checkpoint checkpoints/yakRNA_110M.pt \
    --num_sequences 100 \
    --go_terms "GO:0075523" \
    --length 78 \
    --temperature 1.0 \
    --fasta_output go_prompt.fasta
```

### Multimodal

```bash
# Structure + Consensus + GO terms
python inference/rna_sequence_generator.py \
    --config configs/inference.yaml \
    --checkpoint checkpoints/yakRNA_110M.pt \
    --num_sequences 100 \
    --consensus "GAGUaaGGGGuuCuAGU...gcaGCcCgcCUaGaaCCCUGcgacacuGGuucuaaaaCagAugucgUuuuaAGgGCuUUUG" \
    --go_terms "GO:0075523" \
    --secondary_structure ":::::::<<<<<<<<<-:::--[[[[[-->>>>>>>>><<<<<<<<<<_________>>>->>>>>>>::::]]]]]::::" \
    --constraint_set canonical \
    --temperature 1.0 \
    --fasta_output all_modalities.fasta
```

## Base-Pairing Constraints

When generating with secondary structure (`--secondary_structure`), base-pairing constraints are automatically applied using the `canonical` set by default. You can override this with `--constraint_set`:

| Constraint Set | Allowed Base Pairs | Description |
|----------------|-------------------|-------------|
| `strict` | A:U, U:A, G:C, C:G | Watson-Crick pairs only |
| `canonical` | A:U, U:A, G:C, C:G, G:U, U:G | Watson-Crick + wobble pairs |
| `canonical+sheared` | canonical + G:A, A:G | Adds sheared G:A pairs |
| `canonical+common` | canonical+sheared + A:C, C:A | Adds common non-canonical pairs |
| `permissive` | canonical+common + U:C, C:U | Most permissive set |

Example:
```bash
python inference/rna_sequence_generator.py \
    --config configs/inference.yaml \
    --checkpoint checkpoints/yakRNA_110M.pt \
    --secondary_structure "((((....))))" \
    --constraint_set canonical \
    --num_sequences 5
```

## Project Structure

```
yakRNA/
├── inference/
│   ├── rna_sequence_generator.py   # Main generation script
│   └── discrete_diffusion.py       # Generation components
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
  title = {YAK RNA: An RNA Designer},
  year = {2026},
  url = {https://github.com/YousufAKhan/yakRNA}
}
```

## Contact

- **Author**: Yousuf Khan
- **GitHub**: [@YousufAKhan](https://github.com/YousufAKhan)
