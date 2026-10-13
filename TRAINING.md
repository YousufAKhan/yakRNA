# Training yakRNA Design

This document covers everything needed to go from a bare checkout to a trained
checkpoint: what the model is trained on, how to assemble the dataset from
Rfam, how to launch training, and how to generate with the result.

The released checkpoint (`checkpoints/yakRNA_110M.pt`, 114M parameters) was
produced by exactly the code in this repository with
[`configs/train_110M.yaml`](configs/train_110M.yaml) — that config was recovered
from the checkpoint itself, so it is the recipe, not a reconstruction of it.

---

## 1. How the model is trained

Each training sample is one Rfam sequence with its three companion modalities,
tokenized into a single stream with a shared 326-token vocabulary:

```
[CLS] [SEQ_START] G G A U ... [STRUCT_START] ( ( ( . . ) ) ) ... [CONS_START] gGaUc ... [GO_START] GO:0003735 ... [SEP]
```

| Modality | Token ids | Source |
|---|---|---|
| sequence | 9–15 | the aligned RNA sequence, gaps stripped, `A/C/G/U/N` |
| secondary structure | 16–31 | the family's `#=GC SS_cons` line, projected onto that sequence |
| consensus | 32–43 | the family's `#=GC RF` line, projected onto that sequence |
| GO terms | 44–325 | GO terms attached to the family by `rfam2go` |

Training runs in **two phases**, with the second continuing from the first's
weights (`training.two_phase_training`):

1. **MLM phase** — span masking across the whole token stream (geometric span
   lengths, 2–8 tokens, 35% of positions). Whole conditioning modalities are
   replaced with `[DROPPED]` beforehand, so the model learns to work from any
   subset of modalities. Loss is a weighted sum over the four modalities
   (`training.modality_loss_weights`), with a reduced penalty for consensus
   predictions that get the letter right but the case wrong.
2. **Diffusion phase** — discrete absorbing-state diffusion on the *sequence*
   modality only. A timestep is drawn per sample, a cosine schedule converts it
   into a corruption rate (up to 90%), those sequence positions become `[MASK]`,
   and the model is trained to recover the originals while reading whatever
   conditioning modalities survived dropout. This is the objective
   `inference/rna_sequence_generator.py` samples from at generation time.

Both phases use the same vocabulary-constrained head: logits for tokens that
belong to a different modality than the position being predicted are set to
`-inf` (`training.apply_vocab_constraints`).

Train/val/test splits are **by family**, so no Rfam family is split across
them, and every batch is assembled from distinct families
(`FamilyDiverseSampler`) so that large families cannot dominate a step.

---

## 2. Requirements

Training needs the inference environment plus a logger:

```bash
micromamba env create -f environment-linux.yml
micromamba activate yakrna
pip install -r requirements-training.txt
```

- **Weights & Biases is optional.** Without it, every metric still goes to
  stdout and `outputs/training_<timestamp>.log`. Set `wandb.mode: disabled` (or
  pass `--wandb_mode disabled`) to keep it installed but silent.
- **flash-attn / xformers are optional.** The configs request them; the trainer
  logs what it actually found and falls back to PyTorch SDPA otherwise.
- **Infernal** (`cmalign`, `cmfetch`) is needed only to rebuild the full-region
  dataset: `conda install -c bioconda infernal`.

Hardware used for the released model: 8 GPUs on one node, 96 sequences per GPU,
50,000 steps. A single 24GB GPU trains the same architecture at batch 16 (see
`configs/train_single_gpu.yaml`); a smoke run fits anywhere, CPU included.

---

## 3. Getting the dataset

The model is trained on **Rfam 15.0**. There are three ways to get a trainable
dataset, and all three end in the same place:
`training_data/processed/bert_ready_data.h5` plus `family_lookup.json`, which is
what the shipped configs point at.

| | **A. semRfam** (prebuilt) | B. `seed` | C. `full` |
|---|---|---|---|
| What | download the tokenized HDF5 | build from curated seed alignments | rebuild all full-region hits with `cmalign` |
| Rfam 15.0 size | 4,028 families / 3,978,586 sequences | 4,178 families / 92,781 sequences | ~4,000 families / ~4M sequences |
| Download | ~2.2 GB | ~5 MB | ~6 GB |
| Time | minutes | under a minute | hours to CPU-days |
| Disk | ~2.3 GB | ~250 MB | ~20 GB |
| Needs Infernal | no | no | yes |
| Reproduces the published training set | **exactly** | no | approximately |

**Option A is the recommended route, and the only one that reproduces the
published training set exactly.** Rfam's full-region membership changes between
releases and re-runs, so rebuilding with option C gives a *similar* but not
identical dataset — e.g. RF00044 has one sequence in the published set and six
in a fresh Rfam 15.0 build.

### Option A — download semRfam (recommended)

semRfam is the already-tokenized dataset the released model was trained on,
hosted on the Hugging Face Hub at
[**MasterYster/semRfam**](https://huggingface.co/datasets/MasterYster/semRfam):

```bash
python data_processing/download_rfam.py --stage prebuilt
```

That writes into `training_data/processed/`:

```
bert_ready_data.h5     the 3,978,586 tokenizable records (2.24 GB)
family_lookup.json     family -> row index map for the batch sampler (51 MB)
family_splits.json     the exact train/val/test family split: 3,625 / 201 / 202
                       families = 3,846,938 / 57,920 / 73,728 sequences (35 MB)
```

and nothing else is needed — skip to section 4 and train. Flags:
`--hf-repo <owner>/<name>` for a mirror or fork, `--processed-dir` to put it
somewhere else (update your config's `data_files` to match). Already-present
files are skipped, so the command is safe to re-run.

The derived dataset is redistributable because Rfam is released under
[CC0](http://creativecommons.org/publicdomain/zero/1.0/).

### Option B/C, step 1 — download Rfam

```bash
# quick path: GO mappings + seed alignments
python data_processing/download_rfam.py --stage seed-all

# published path: GO mappings + covariance models + per-family FASTA + cmalign
python data_processing/download_rfam.py --stage full-all --jobs 8 --cmalign-cpu 4
```

Either way you end up with:

```
training_data/multi_modality_Rfam/
├── rfam2go.txt
└── stockholm/RF00001/RF00001.sto, RF00002/RF00002.sto, ...
```

Useful flags: `--families RF00001,RF00005` or `--max-families 50` to try the
full path on a handful of families first; `--release` for a different Rfam
release; `--discard-archives` to delete each `.gz` after unpacking. Download and
alignment are both resumable — rerun the same command and finished families are
skipped. Families that fail are listed in
`training_data/multi_modality_Rfam/align_failures.txt` instead of stopping the
run, and the script refuses to mix seed and full alignments in one directory.

The `full` stage runs `cmalign --outformat Pfam`, which is required: the
Stockholm parser reads one `#=GC SS_cons` line per family, so interleaved
alignments would be silently truncated.

### Option B/C, step 2 — tokenizable records (HDF5)

```bash
python data_processing/rfam_data_preprocessing.py --stage all
```

Reads `rfam2go.txt` + `stockholm/`, strips alignment gaps while keeping
structure and consensus aligned to the remaining positions, and writes
`training_data/processed/bert_ready_data.h5` with one row per sequence and the
columns `sequence`, `ss`, `consensus`, `go_terms`, `rfam_id`, `sequence_id`.
Add `--remove-temp-files` to drop the intermediate JSON (which is several times
larger than the HDF5), or `--hdf5_output`/`--clean_output` etc. to write
somewhere other than `training_data/processed/`.

### Option B/C, step 3 — vocabulary

The repository already ships the vocabulary the released checkpoint uses:
`training_data/processed/vocabulary_analysis/unified_vocabulary.json` (326
tokens). **Keep it** if you are fine-tuning or resuming from
`yakRNA_110M.pt` — rebuilding assigns different token ids and the checkpoint
will no longer match.

To build one from your own data (training from scratch, a different Rfam
release, or extra GO terms):

```bash
python data_processing/vocabulary.py \
    --input training_data/processed/bert_ready_data.h5 \
    --build-unified-vocab \
    --output-dir training_data/processed/vocabulary_analysis
```

### Option B/C, step 4 — family lookup

The batch sampler needs a family → row-index map. This one is not created
automatically:

```bash
python data_processing/training_data_preparation.py \
    --config configs/train_single_gpu.yaml \
    --create-lookup
```

It writes `family_lookup.json` next to the HDF5 file. The train/val/test split
file (`family_splits.json`) *is* created automatically on the first training
run, using `data.train_split`/`val_split`/`test_split`, and reused afterwards —
delete it to re-split. You can also create it up front with `--test-splits`.

At this point `training_data/processed/` holds the four files the configs point
at:

```
bert_ready_data.h5
family_lookup.json
family_splits.json            # created by the first training run
vocabulary_analysis/unified_vocabulary.json
```

### Publishing semRfam (maintainers)

To (re)publish the prebuilt dataset, from a checkout that has the files:

```bash
pip install huggingface_hub
hf auth login
hf repo create semRfam --repo-type dataset          # once
hf upload semRfam training_data/processed/bert_ready_data.h5 semRfam.h5 --repo-type dataset
hf upload semRfam training_data/processed/family_lookup.json family_lookup.json --repo-type dataset
hf upload semRfam training_data/processed/family_splits.json family_splits.json --repo-type dataset
```

`--stage prebuilt` accepts either `semRfam.h5` or `bert_ready_data.h5` as the
name in the repo. A 2.2 GB file is well inside the Hub's limits (200 GB
recommended per file), and a dataset card is expected for dataset repos.

---

## 4. Training

```bash
# verify the whole pipeline in ~2 minutes (tiny model, 10 steps, no W&B)
python rna_gen/rna_gen_training.py --config configs/train_smoke.yaml

# one GPU
python rna_gen/rna_gen_training.py --config configs/train_single_gpu.yaml

# 8 GPUs on one node - the released recipe
torchrun --nproc_per_node=8 rna_gen/rna_gen_training.py --config configs/train_110M.yaml
```

Multi-GPU uses DistributedDataParallel with a distributed family-diverse
sampler; only rank 0 logs and writes checkpoints. `environment/train_cluster.slurm`
is a ready-to-adapt Slurm script for the 8-GPU case.

Any of these can be overridden on the command line without editing the YAML:

```bash
python rna_gen/rna_gen_training.py --config configs/train_110M.yaml \
    --batch_size 16 --max_steps 2000 --learning_rate 5e-5 \
    --print_freq 20 --validation_freq_steps 200 --wandb_mode disabled
```

### What you get

```
outputs/                                   # training.output_dir
├── training_2026-10-12_20-32-58.log       # full metric history
├── <checkpoint_prefix>_step_1000.pt       # every training.model_save_freq_steps
└── <checkpoint_prefix>_final.pt
```

Each checkpoint holds the weights, optimizer and scheduler state, the resolved
config, the architecture, and which phase it was written in — enough to resume
training *or* to generate from directly.

### Reading the logs

```
[PROGRESS] MLM Phase (2/45000) | Global Step: 2/50000 | Avg Loss: 1.8179 | Time/Iter: 0.2s | ETA: 2.5h | Seq/sec: 25.3
MLM METRICS - TRAINING - Step 100 ... SEQUENCE: Acc=0.42, Loss=1.31, Perp=3.71
DIFFUSION METRICS - VALIDATION - Step 46000 ... GENERATION: Acc=0.51, Loss=1.19, Perp=3.29
```

Per-modality accuracy/loss/perplexity are reported separately for each phase,
along with per-nucleotide precision/recall, accuracy bucketed by diffusion
timestep, corruption-rate statistics, and how many unique sequences and
families have been seen. `PHASE OVERLAP` at the end reports how much of the
diffusion phase's data the MLM phase had already seen — with
`use_all_data_for_training: true` that is expected to be high.

### Resuming and fine-tuning

```yaml
checkpoint:
  resume_from_checkpoint: true
  checkpoint_path: "outputs/yakRNA_110M_step_40000.pt"
  resume_training: true
  resume_optimizer: true
  resume_scheduler: true
  resume_step: true      # false to restart the step counter, e.g. for fine-tuning
```

To fine-tune the released model on your own families: keep the shipped
vocabulary, point `checkpoint_path` at `checkpoints/yakRNA_110M.pt`, set
`resume_step: false`, lower `learning_rate` (1e-5 to 3e-5 is a reasonable
starting range), and set the two-phase step counts to whatever budget you have.
`_orig_mod.` and `module.` prefixes from `torch.compile`/DDP are stripped on
load, so compiled and uncompiled checkpoints are interchangeable.

---

## 5. Generating from your own checkpoint

Training checkpoints are the same format the inference stack already reads:

```bash
python inference/rna_sequence_generator.py \
    --config configs/inference.yaml \
    --checkpoint outputs/yakRNA_single_gpu_final.pt \
    --num_sequences 10 --length 100
```

`configs/inference.yaml` describes the released 110M architecture. If you
trained a different size you do not need to edit it — the checkpoint records
its own architecture and the generator uses that. You *do* need
`configs/inference.yaml`'s `data_files.vocabulary_file` to point at the same
vocabulary the model was trained with.

---

## 6. Config reference

Everything is config-driven. The three shipped configs differ only in scale:

| File | Model | Batch | Steps | Purpose |
|---|---|---|---|---|
| `configs/train_110M.yaml` | 12L / 768h (114M) | 96 × 8 GPUs | 50k (45k MLM + 5k diffusion) | the released recipe |
| `configs/train_single_gpu.yaml` | 12L / 768h (114M) | 16 | 50k | one 24GB GPU |
| `configs/train_smoke.yaml` | 4L / 256h (~10M) | 4 | 10 | pipeline check |

Keys worth knowing:

| Key | Meaning |
|---|---|
| `data.max_position_embeddings` | total token budget per sample; longer samples are **skipped**, not truncated |
| `data.use_all_data_for_training` | `true` adds the val/test families to training (what the released run did) |
| `data.sequence_buckets` | lengths warmed up once at startup to avoid recompiling per shape |
| `data_files.load_data_into_ram` | read the whole HDF5 into RAM up front |
| `data_files.temporary_processing_directory` | copy the data to node-local scratch first (env vars expanded) |
| `training.apply_vocab_constraints` | mask out-of-modality logits |
| `training.modality_loss_weights` | MLM-phase weighting; must sum to 1.0 |
| `training.MLM_masking.*` | span-masking parameters and MLM-phase modality dropout |
| `training.diffusion.cumulative_masking` | corruption schedule for the diffusion phase |
| `training.diffusion.modality_dropout_rates` | per-modality dropout for conditional generation |
| `training.stale_validation` | pin validation samples/masks/noise so validations are comparable |
| `training.test_num_samples` | cap the final test pass (absent = the whole test split) |
| `training.lr_scheduler.type` | `cosine_with_restarts` or `linear_warmup`; anything else warns and uses linear |
| `optimizations.*` | flash-attn, xformers, gradient checkpointing, torch.compile, mixed precision |

Two notes on the published recipe, so the numbers in `train_110M.yaml` are not
misread:

- The optimizer is **AdamW** (fused when CUDA is available), with
  `training.learning_rate` and `training.weight_decay`. The original config
  carried an inert `optimizer: {type: StableAdamW, ...}` block that the trainer
  never read; it has been dropped rather than left in place suggesting
  otherwise.
- `lr_scheduler.type: cosine` is not one of the two recognized values, so the
  released run trained with linear warmup + linear decay. The value is kept
  as-is so the file reproduces that run; the trainer now warns when it falls
  back.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Training exits with "Training data is not ready" | follow the command it prints: `--stage prebuilt` to download semRfam, or options B/C in section 3 |
| `FileNotFoundError: .../family_lookup.json` | run option B/C step 4 (`--create-lookup`), or `--stage prebuilt` |
| `Found 0 families in train split` | `family_lookup.json` was built from a different HDF5 — rebuild it |
| First training step takes many minutes | `torch_compile.enabled: true` plus the bucket warmup; set it to `false` while iterating |
| CUDA OOM | lower `data.batch_size`, keep `memory.gradient_checkpointing: true`, or lower `data.max_position_embeddings` |
| `RuntimeError: ... size mismatch for ...tok_embeddings.weight` | the vocabulary does not match the checkpoint; use the shipped `unified_vocabulary.json` |
| `modality_loss_weights must sum to 1.0` | exactly what it says |
| Test evaluation never finishes | set `training.test_num_samples` (the full split is 5% of ~4M sequences) |
| `cmalign: DP matrix too big` | raise `--mxsize` on `download_rfam.py` |
| Many `Skipping sample N: Total concatenated length ... exceeds` | expected: samples above `max_position_embeddings` are dropped. Raise it (up to 8192) to keep them |
