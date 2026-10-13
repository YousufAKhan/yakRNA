"""
yakRNA Design - two-phase (MLM then discrete diffusion) training for the
multimodal ModernBERT RNA composer.

How to run:
    # single GPU
    python rna_gen/rna_gen_training.py --config configs/train_single_gpu.yaml

    # multi-GPU on one node (the recipe used for the released 110M model)
    torchrun --nproc_per_node=8 rna_gen/rna_gen_training.py --config configs/train_110M.yaml

    # quick end-to-end smoke test (tiny model, a handful of steps)
    python rna_gen/rna_gen_training.py --config configs/train_smoke.yaml

Phase 1 trains masked-language modelling over the four concatenated modalities
(sequence, secondary structure, consensus, GO terms); phase 2 keeps the same
weights and trains a discrete (absorbing-mask) diffusion objective on the
sequence modality, conditioned on the remaining modalities with per-modality
dropout. See TRAINING.md for the data pipeline and the full config reference.

Supported optimizations (all config-driven, see configs/*.yaml):
- Flash Attention 2
- Gradient checkpointing for memory efficiency
- torch.compile
- bf16/fp16 mixed precision
- xFormers attention
- DistributedDataParallel for multi-GPU training

Checkpoints are written to <training.output_dir> and are directly loadable by
inference/rna_sequence_generator.py.
"""
import sys
import os
import argparse
import yaml
import logging
import functools
from datetime import datetime
import numpy as np
# For eager shadow model used in validation
import copy
# For approximate unique counting with constant memory - using hash-based approach
import hashlib
import random
import torch
import contextlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    import torch._dynamo as _dynamo_mod  # noqa: F401
except Exception:  # pragma: no cover
    _dynamo_mod = None

def _disable_dynamo(fn):
    if _dynamo_mod is None:
        return fn
    try:
        return _dynamo_mod.disable(fn)
    except Exception:
        return fn

@_disable_dynamo
def to_float(tensor_value: torch.Tensor) -> float:
    return float(tensor_value.detach())

@_disable_dynamo
def to_int(tensor_value: torch.Tensor) -> int:
    return int(tensor_value.detach())

@_disable_dynamo
def to_list_cpu(tensor_value: torch.Tensor):
    return tensor_value.detach().to('cpu', non_blocking=True).tolist()

# --- Repo-local imports -------------------------------------------------------
# Make the sibling packages importable no matter where the script is launched
# from. `inference/` holds the discrete-diffusion components, which are shared
# verbatim between training and generation, so there is a single source of truth.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _extra_path in (
    os.path.dirname(os.path.abspath(__file__)),      # rna_gen/  (metrics_calculator)
    _REPO_ROOT,                                      # repo root (data_processing package)
    os.path.join(_REPO_ROOT, 'inference'),           # inference/ (discrete_diffusion)
    os.path.join(_REPO_ROOT, 'data_processing'),      # data_processing/ (direct imports)
):
    if _extra_path not in sys.path:
        sys.path.append(_extra_path)

from metrics_calculator import MLMMetricsCalculator, DiffusionMetricsCalculator
from torch.profiler import record_function
from data_processing.training_data_preparation import DistributedFamilyDiverseSampler, DistributedSimpleFamilyDiverseSampler

# Weights & Biases is optional. Without it, training still logs every metric to
# stdout and to outputs/training_<timestamp>.log; set `wandb.mode: disabled` in
# the config to keep it installed but silent.
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when wandb is absent
    WANDB_AVAILABLE = False

    class _NoOpWandbPlot:
        @staticmethod
        def bar(*args, **kwargs):
            return None

    class _NoOpWandb:
        """Stand-in for the small slice of the wandb API this script uses."""
        plot = _NoOpWandbPlot()

        @staticmethod
        def Table(*args, **kwargs):
            return None

        @staticmethod
        def init(*args, **kwargs):
            return None

        @staticmethod
        def log(*args, **kwargs):
            return None

        @staticmethod
        def finish(*args, **kwargs):
            return None

    wandb = _NoOpWandb()

# --- Distributed Utilities ---
def init_distributed(logger=None):
    """Initialize torch.distributed if launched with torchrun.
    Returns True if distributed is initialized, else False.
    """
    try:
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if world_size > 1 and dist.is_available() and not dist.is_initialized():
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend)
            if logger:
                logger.info(f"DDP initialized: backend={backend}, world_size={world_size}, rank={int(os.environ.get('RANK','0'))}")
            # Pin device to local rank for CUDA
            if torch.cuda.is_available():
                local_rank = int(os.environ.get('LOCAL_RANK', '0'))
                torch.cuda.set_device(local_rank)
        return dist.is_available() and dist.is_initialized()
    except Exception as e:
        if logger:
            logger.warning(f"Failed to initialize distributed: {e}")
        return False

def is_main_process() -> bool:
    try:
        return int(os.environ.get('RANK', '0')) == 0
    except Exception:
        return True

def distributed_sum(value: float, device: torch.device) -> float:
    """All-reduce sum a scalar float across ranks. Returns the global sum on all ranks.
    If DDP not initialized, returns value.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    t = torch.tensor([float(value)], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())

def cleanup_distributed():
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

def gather_union_size(local_set):
    """Gather Python sets from all ranks and return union size (rank 0 only).
    Returns local len on non-main ranks.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return len(local_set)
    try:
        object_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(object_list, local_set)
        # Union all members on rank 0
        if is_main_process():
            union_set = set()
            for s in object_list:
                try:
                    union_set.update(s)
                except Exception:
                    continue
            return len(union_set)
        else:
            return len(local_set)
    except Exception:
        # Fallback to local
        return len(local_set)

# --- Patch for Mac OpenMP error ---
def set_kmp_duplicate_lib_ok_from_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if config.get('training', {}).get('set_kmp_duplicate_lib_ok', False):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        print('[INFO] Set KMP_DUPLICATE_LIB_OK=TRUE for Mac OpenMP workaround')

# --- Modern Optimizations Setup ---
def setup_optimizations(config, logger):
    """Setup modern transformer optimizations based on config."""
    optimizations = config.get('optimizations', {})
    
    # Flash Attention setup
    flash_config = optimizations.get('flash_attention', {})
    if flash_config.get('enabled', False):
        try:
            import flash_attn
            logger.info(f"✅ Flash Attention {flash_config.get('version', 2)} available")
            os.environ['FLASH_ATTENTION_ENABLED'] = 'true'
            # Prefer FlashAttention-2 over PyTorch SDPA flash when available
            os.environ.setdefault('PYTORCH_SDP_DISABLE_FLASH', '1')
            logger.info("🔧 Disabled PyTorch SDPA flash to prefer FlashAttention-2 kernels")
        except ImportError:
            logger.warning("⚠️ Flash Attention not installed. Install with: pip install flash-attn --no-build-isolation")
            os.environ['FLASH_ATTENTION_ENABLED'] = 'false'
            # Fall back to PyTorch SDPA flash attention for speed
            try:
                torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
                logger.info("🔧 Enabled PyTorch SDPA flash attention (fallback)")
            except Exception:
                pass
    
    # xFormers setup
    xformers_config = optimizations.get('xformers', {})
    if xformers_config.get('enabled', False):
        try:
            import xformers
            logger.info("✅ xFormers available for memory-efficient attention")
            os.environ['XFORMERS_ENABLED'] = 'true'
        except ImportError:
            logger.warning("⚠️ xFormers not installed. Install with: pip install xformers")
            os.environ['XFORMERS_ENABLED'] = 'false'
    
    # Set precision for mixed precision training
    mixed_precision_config = optimizations.get('mixed_precision', {})
    if mixed_precision_config.get('enabled', False):
        precision = mixed_precision_config.get('precision', 'bf16')
        os.environ['MIXED_PRECISION_TYPE'] = precision
        logger.info(f"✅ Mixed precision training enabled with {precision}")
    
    return optimizations

def create_optimized_model_config(config, vocab_size, pad_token_id, mask_token_id, cls_token_id, logger):
    """Create ModernBERT config with compatible optimizations."""
    # Get model architecture from new model_architecture section, fallback to training.model_size
    model_arch = config.get('model_architecture', {})
    model_size = model_arch.get('model_size', config.get('training', {}).get('model_size', {}))
    optimizations = config.get('optimizations', {})
    
    # Base configuration for ModernBERT
    model_config_dict = {
        'vocab_size': vocab_size,
        'pad_token_id': pad_token_id,
        'mask_token_id': mask_token_id,
        'cls_token_id': cls_token_id,
        'num_hidden_layers': model_size['num_hidden_layers'],
        'hidden_size': model_size['hidden_size'],
        'num_attention_heads': model_size['num_attention_heads'],
        'intermediate_size': model_size['intermediate_size']
    }
    
    # Use explicit max_position_embeddings from config for model optimization
    max_position_embeddings = config['data']['max_position_embeddings']
    model_config_dict['max_position_embeddings'] = max_position_embeddings
    
    logger.info(f"ℹ️ ModernBERT configured for {max_position_embeddings} total tokens")
    logger.info(f"✅ RoPE positioning explicitly set for optimal memory/speed")
    
    # Apply dropout settings from training config if provided
    training_cfg = config.get('training', {})
    dropout_cfg = training_cfg.get('dropout', {})
    if 'attention_dropout' in dropout_cfg:
        model_config_dict['attention_dropout'] = float(dropout_cfg['attention_dropout'])
    if 'hidden_dropout' in dropout_cfg:
        model_config_dict['hidden_dropout'] = float(dropout_cfg['hidden_dropout'])

    # Flash Attention configuration (ModernBERT explicitly supports this)
    flash_config = optimizations.get('flash_attention', {})
    if flash_config.get('enabled', False) and os.environ.get('FLASH_ATTENTION_ENABLED') == 'true':
        model_config_dict['use_flash_attention_2'] = True
        # Set dtype for Flash Attention 2
        mixed_precision_config = optimizations.get('mixed_precision', {})
        if mixed_precision_config.get('enabled', False):
            precision = mixed_precision_config.get('precision', 'bf16')
            if precision == 'bf16':
                model_config_dict['torch_dtype'] = torch.bfloat16
            else:
                model_config_dict['torch_dtype'] = torch.float16
        else:
            model_config_dict['torch_dtype'] = torch.float32
        logger.info("✅ Flash Attention 2 enabled for ModernBERT with proper dtype")
    
    # Memory optimizations
    memory_config = optimizations.get('memory', {})
    if memory_config.get('use_cache', True) == False:
        model_config_dict['use_cache'] = False
        logger.info("✅ KV cache disabled for memory savings")
    
    return model_config_dict

def preflight_data_files(config, config_path, logger):
    """Fail fast, with the command that fixes it, when the dataset is not built.

    Nothing here downloads or builds anything: the full Rfam dataset is a
    multi-GB download and CPU-days of cmalign, which is not something a `train`
    invocation should start on its own. The point is to say exactly which file
    is missing and which command produces it, instead of surfacing an h5py
    traceback 30 seconds in.

    `family_splits_file` is deliberately not required - the first training run
    creates it from the HDF5 and later runs reuse it.
    """
    data_files = config.get('data_files', {}) or {}
    seed_note = "    # --stage seed-all for the ~93k-sequence seed set (minutes),\n" \
                "    # --stage full-all for the ~4M-sequence set the released model used (hours+)"
    required = [
        ('h5_file', 'Tokenizable Rfam records (semRfam)',
         "python data_processing/download_rfam.py --stage prebuilt\n"
         "    # downloads semRfam, the dataset the released model was trained on (~2.2 GB).\n"
         "    # To build it from Rfam locally instead:\n"
         f"    python data_processing/download_rfam.py --stage seed-all\n{seed_note}\n"
         "    python data_processing/rfam_data_preprocessing.py --stage all"),
        ('vocabulary_file', 'Unified multimodal vocabulary',
         "# this repository ships the vocabulary the released model uses at\n"
         "    #   training_data/processed/vocabulary_analysis/unified_vocabulary.json\n"
         "    # keep that one when resuming from yakRNA_110M.pt, or build your own:\n"
         "    python data_processing/vocabulary.py \\\n"
         "        --input <your bert_ready_data.h5> --build-unified-vocab"),
        ('family_lookup_file', 'Family -> row index map used by the batch sampler',
         "# included in --stage prebuilt; to build it from your own HDF5:\n"
         f"    python data_processing/training_data_preparation.py \\\n"
         f"        --config {config_path} --create-lookup"),
    ]

    missing = [(key, label, command) for key, label, command in required
               if not data_files.get(key) or not os.path.exists(data_files[key])]

    if missing:
        lines = [
            "",
            "=" * 78,
            f"Training data is not ready: {len(missing)} of {len(required)} required file(s) missing.",
            "=" * 78,
        ]
        for key, label, command in missing:
            lines += [
                "",
                f"  data_files.{key}: {data_files.get(key) or '<not set in config>'}",
                f"    {label}",
                f"    {command}",
            ]
        lines += [
            "",
            "The full data pipeline is documented in TRAINING.md (section 3).",
            "=" * 78,
            "",
        ]
        message = "\n".join(lines)
        if logger is not None:
            logger.error(message)
        else:
            print(message)
        sys.exit(1)

    splits_file = data_files.get('family_splits_file')
    if splits_file and not os.path.exists(splits_file):
        logger.info(f"📐 No split file at {splits_file} yet - creating family-level "
                    f"train/val/test splits from the HDF5 (this run only; reused afterwards)")


def resolve_output_dir(config):
    """Directory for checkpoints and logs.

    Prefers `training.output_dir` (where every shipped config sets it) and falls
    back to a top-level `output_dir`, then to `outputs/`.
    """
    training_cfg = config.get('training', {}) or {}
    return training_cfg.get('output_dir') or config.get('output_dir') or 'outputs'

def serializable_model_config(model_config_dict):
    """Strip non-YAML/JSON-friendly values (e.g. torch dtypes) from the model config."""
    clean = {}
    for key, value in model_config_dict.items():
        if isinstance(value, torch.dtype):
            clean[key] = str(value).replace('torch.', '')
        else:
            clean[key] = value
    return clean

def setup_gradient_checkpointing(model, config, logger):
    """Setup gradient checkpointing if enabled."""
    optimizations = config.get('optimizations', {})
    memory_config = optimizations.get('memory', {})
    
    if memory_config.get('gradient_checkpointing', False):
        if hasattr(model, 'gradient_checkpointing_enable'):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")
        else:
            logger.warning("⚠️ Model doesn't support gradient checkpointing")

def fix_checkpoint_keys(state_dict):
    """
    Fix checkpoint keys by removing _orig_mod prefix if present.
    
    This handles checkpoints saved from torch.compile wrapped models which
    add the _orig_mod prefix to all parameter names.
    
    Args:
        state_dict (dict): The state dict to fix
        
    Returns:
        dict: Fixed state dict with proper key names
    """
    fixed_state_dict = {}
    
    for key, value in state_dict.items():
        # Remove _orig_mod prefix if present
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
            fixed_state_dict[new_key] = value
        else:
            fixed_state_dict[key] = value
    
    return fixed_state_dict


def compile_model(model, config, logger):
    """Compile model with torch.compile if enabled."""
    optimizations = config.get('optimizations', {})
    compile_config = optimizations.get('torch_compile', {})
    
    if compile_config.get('enabled', False):
        try:
            if hasattr(torch, 'compile'):
                mode = compile_config.get('mode', 'default')
                model = torch.compile(model, mode=mode)
                logger.info(f"✅ Model compiled with torch.compile (mode: {mode})")
            else:
                logger.warning("⚠️ torch.compile not available (requires PyTorch 2.0+)")
        except Exception as e:
            logger.warning(f"⚠️ Failed to compile model: {e}")
    
    return model

def enable_cuda_graphs_debug_if_needed(config, logger):
    """Enable detailed CUDA Graphs logging to pinpoint capture skips.
    Controlled by config.debug.cuda_graphs_debug: true
    """
    try:
        debug_cfg = config.get('debug', {}) if isinstance(config, dict) else {}
        if debug_cfg.get('cuda_graphs_debug', False):
            existing = os.environ.get('TORCH_LOGS', '')
            if '+cuda_graphs' not in existing:
                new_val = (existing + (';' if existing and not existing.endswith(';') else '') + '+cuda_graphs').lstrip(';') if existing else '+cuda_graphs'
                os.environ['TORCH_LOGS'] = new_val
            logger.info(f"🔎 CUDA Graphs debug enabled. TORCH_LOGS={os.environ.get('TORCH_LOGS')}")
    except Exception as _:
        pass

def warmup_model_buckets_forward_only(model, config, device, vocab_size, logger):
    """Simple, resilient warmup: run 1 forward per configured sequence bucket to
    trigger initial compilation without risking backward/cudagraph issues.
    Skips silently on any error.
    """
    data_cfg = config.get('data', {})
    buckets = data_cfg.get('sequence_buckets') or []
    if not buckets:
        return
    batch_size = int(data_cfg.get('batch_size', 1))
    mp_cfg = config.get('optimizations', {}).get('mixed_precision', {})
    precision = (mp_cfg.get('precision', 'bf16') or 'bf16').lower()
    amp_dtype = torch.bfloat16 if precision == 'bf16' else (torch.float16 if precision in ('fp16', 'float16', 'half') else torch.float32)
    # Keep to valid range
    max_len = int(data_cfg.get('max_position_embeddings', 2048))
    buckets = [min(int(b), max_len) for b in buckets]
    logger.info(f"🔧 Startup warmup over {len(buckets)} buckets (forward-only)")
    for L in buckets:
        try:
            input_ids = torch.randint(0, vocab_size, (batch_size, L), device=device)
            attention_mask = torch.ones((batch_size, L), dtype=torch.long, device=device)
            labels = torch.randint(0, vocab_size, (batch_size, L), device=device)
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', dtype=amp_dtype):
                _ = model(input_ids, attention_mask=attention_mask, labels=labels)
            if device.type == 'cuda':
                torch.cuda.synchronize()
                # Clear cache after warmup to free memory
                torch.cuda.empty_cache()
        except Exception as _e:  # best-effort warmup; continue
            if device.type == 'cuda':
                torch.cuda.empty_cache()  # Cleanup on error too
            continue

def create_optimized_data_loader(dataset, config, sampler=None, collate_fn=None, training=True):
    """Create optimized data loader with modern settings."""
    data_config = config['data']
    optimizations = config.get('optimizations', {})
    dataloader_config = optimizations.get('dataloader', {})
    
    # Base settings
    loader_kwargs = {
        'dataset': dataset,
        'batch_size': data_config['batch_size'],
        'num_workers': data_config['num_workers'],
        'pin_memory': data_config.get('pin_memory', True),
    }
    
    # Add sampler if provided
    if sampler is not None:
        loader_kwargs['sampler'] = sampler
    else:
        loader_kwargs['shuffle'] = training
    
    # Add collate function if provided
    if collate_fn is not None:
        loader_kwargs['collate_fn'] = collate_fn
    
    # Optimization settings
    if dataloader_config.get('persistent_workers', True) and data_config['num_workers'] > 0:
        loader_kwargs['persistent_workers'] = True
    
    loader_kwargs['prefetch_factor'] = dataloader_config.get('prefetch_factor', 2)
    loader_kwargs['drop_last'] = bool(training)
    # Add timeout to prevent hanging worker processes
    if data_config['num_workers'] > 0:
        loader_kwargs['timeout'] = dataloader_config.get('timeout', 30)
    # Route pinned memory to CUDA explicitly when available (PyTorch 2.3+)
    try:
        if torch.cuda.is_available():
            loader_kwargs['pin_memory_device'] = 'cuda'
    except Exception:
        pass
    
    if training and dataloader_config.get('drop_last', False):
        loader_kwargs['drop_last'] = True
    
    return torch.utils.data.DataLoader(**loader_kwargs)

def setup_enhanced_scheduler(optimizer, config, total_steps):
    """Setup enhanced learning rate scheduler."""
    from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
    
    lr_config = config['training'].get('lr_scheduler', {})
    scheduler_type = lr_config.get('type', 'linear_warmup')
    warmup_steps = lr_config.get('warmup_steps', int(0.1 * total_steps))
    
    if scheduler_type not in ('cosine_with_restarts', 'linear_warmup'):
        logging.getLogger().warning(
            f"lr_scheduler.type '{scheduler_type}' is not recognized "
            "(supported: 'cosine_with_restarts', 'linear_warmup'); using linear warmup + decay"
        )

    if scheduler_type == 'cosine_with_restarts':
        cycles = lr_config.get('cycles', 1)
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps,
            num_cycles=cycles
        )
    else:
        # Default linear warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
    
    return scheduler

# --- Token decoding for readable logs ---
def decode_tokens(token_ids, vocab_json):
    special_tokens = {v: k for k, v in vocab_json['special_tokens'].items()}
    decoded = []
    for tid in token_ids:
        if tid in special_tokens:
            decoded.append(f"[{special_tokens[tid][1:-1]}]")
        else:
            found = False
            for modality, tokens in vocab_json['modality_vocabs'].items():
                offset = vocab_json['modality_offsets'][modality]
                if offset <= tid < offset + len(tokens):
                    decoded.append(tokens[tid - offset])
                    found = True
                    break
            if not found:
                decoded.append("?")
    return decoded

# --- Masked visualization ---
def visualize_masked(unmasked, masked, mask_token_id, vocab_json):
    special_tokens = {v: k for k, v in vocab_json['special_tokens'].items()}
    dropped_token_id = vocab_json['special_tokens']['[DROPPED]']
    vis = []
    for u, m in zip(unmasked, masked):
        if m == mask_token_id:
            vis.append("*")
        elif m == dropped_token_id:
            vis.append("X")
        elif m in special_tokens:
            vis.append(f"[{special_tokens[m][1:-1]}]")
        else:
            vis.append(u)
    return vis

def apply_modality_vocab_mask(logits, modality_type_ids, valid_token_indices_by_modality, apply_vocab_constraints):
    """
    Mask logits so that only valid tokens for each modality can be predicted at each position.
    Args:
        logits: (batch, seq_len, vocab_size)
        modality_type_ids: (batch, seq_len) int tensor
        valid_token_indices_by_modality: dict {modality_id: set of valid token indices}
        apply_vocab_constraints: bool
    Returns:
        masked_logits: same shape as logits
    """
    if not apply_vocab_constraints:
        return logits
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device
    # Build a mask of shape (batch, seq_len, vocab_size), True for valid, False for invalid
    mask = torch.zeros((batch_size, seq_len, vocab_size), dtype=torch.bool, device=device)
    for modality_id, valid_indices in valid_token_indices_by_modality.items():
        # Find positions in batch where this modality applies
        positions = (modality_type_ids == modality_id)  # (batch, seq_len) bool
        if positions.any():
            # For these positions, set valid token indices to True
            idxs = torch.nonzero(positions, as_tuple=False)  # (N, 2)
            for b, s in idxs:
                mask[b, s, list(valid_indices)] = True
    # Set logits for invalid tokens to -1e9 (large negative number, effectively zero probability)
    # This ensures numerical stability while preventing invalid token predictions
    logits = logits.masked_fill(~mask, -1e9)
    return logits

def load_data_into_ram(config, logger):
    """Load training data into RAM for faster access."""
    ram_config = config.get('optimizations', {}).get('ram_loading', {})
    if not ram_config.get('enabled', False):
        logger.info("📁 RAM loading disabled - using disk-based data loading")
        return None
    
    logger.info("🚀 Loading training data into RAM for maximum performance...")
    
    import h5py
    import time
    
    h5_path = config['data_files']['h5_file']
    start_time = time.time()
    
    try:
        # Load entire H5 file into RAM
        with h5py.File(h5_path, 'r') as f:
            # Load all datasets into memory
            ram_data = {}
            for key in f.keys():
                ram_data[key] = f[key][:]
        
        load_time = time.time() - start_time
        total_size_mb = sum(data.nbytes for data in ram_data.values()) / (1024**2)
        
        logger.info(f"✅ Data loaded into RAM in {load_time:.2f}s")
        logger.info(f"📊 Total RAM usage: {total_size_mb:.1f}MB")
        logger.info(f"⚡ RAM loading speed: {total_size_mb/load_time:.1f}MB/s")
        
        return ram_data
        
    except Exception as e:
        logger.error(f"❌ Failed to load data into RAM: {e}")
        logger.info("🔄 Falling back to disk-based loading")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ModernBERT on RNA multimodal data')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    
    # Add command line overrides for common parameters
    parser.add_argument('--validation_freq_steps', type=int, help='Override validation frequency (steps)')
    parser.add_argument('--print_freq', type=int, help='Override print frequency (steps)')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    parser.add_argument('--max_steps', type=int, help='Override max training steps')
    parser.add_argument('--wandb_mode', type=str, choices=['online', 'offline', 'disabled'],
                        help="Override wandb.mode (use 'disabled' to train without a W&B account)")
    
    args, remaining = parser.parse_known_args()
    set_kmp_duplicate_lib_ok_from_config(args.config)

    # Setup logging to file and console
    os.makedirs('outputs', exist_ok=True)
    log_name = f"outputs/training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_name),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger()
    logger.info(f"Logging to {log_name}")

    # --- WANDB INIT ---
    with open(args.config, 'r') as f:
        config_yaml = yaml.safe_load(f)
    
    # Apply command line overrides to config
    if args.validation_freq_steps is not None:
        config_yaml['training']['validation_freq_steps'] = args.validation_freq_steps
        print(f"🔄 Overriding validation_freq_steps: {args.validation_freq_steps}")
    
    if args.print_freq is not None:
        config_yaml['training']['print_freq'] = args.print_freq
        print(f"🔄 Overriding print_freq: {args.print_freq}")
    
    if args.batch_size is not None:
        config_yaml['data']['batch_size'] = args.batch_size
        print(f"🔄 Overriding batch_size: {args.batch_size}")
    
    if args.learning_rate is not None:
        config_yaml['training']['learning_rate'] = args.learning_rate
        print(f"🔄 Overriding learning_rate: {args.learning_rate}")
    
    if args.max_steps is not None:
        config_yaml['training']['max_steps'] = args.max_steps
        print(f"🔄 Overriding max_steps: {args.max_steps}")

    if args.wandb_mode is not None:
        config_yaml.setdefault('wandb', {})['mode'] = args.wandb_mode
        print(f"🔄 Overriding wandb.mode: {args.wandb_mode}")

    # train() and create_data_loaders() both read the config back from disk, so
    # the command-line overrides are materialized into a temporary YAML and
    # args.config is repointed at it. Without this the overrides above would
    # only affect what gets recorded in W&B.
    if any(value is not None for value in (args.validation_freq_steps, args.print_freq,
                                           args.batch_size, args.learning_rate,
                                           args.max_steps, args.wandb_mode)):
        import atexit
        import tempfile
        _fd, _effective_config_path = tempfile.mkstemp(prefix='yakrna_effective_config_', suffix='.yaml')
        with os.fdopen(_fd, 'w') as _f:
            yaml.safe_dump(config_yaml, _f, sort_keys=False)
        atexit.register(lambda path=_effective_config_path: os.path.exists(path) and os.remove(path))
        args.config = _effective_config_path
        print(f"📝 Effective config with overrides written to: {_effective_config_path}")

    # Check the dataset before anything with side effects (W&B runs, DDP
    # process groups): a missing dataset should cost nothing but a message.
    preflight_data_files(config_yaml, args.config, logger)

    # Initialize distributed before W&B so we can gate logging
    ddp_enabled = init_distributed(logger)
    # Suppress all stdout/logging on non-main ranks
    if ddp_enabled and not is_main_process():
        try:
            import builtins as _builtins
            # Disable print on non-main ranks
            _builtins.print = lambda *args, **kwargs: None
        except Exception:
            pass
        # Disable Python logging on non-main ranks
        try:
            logging.disable(logging.CRITICAL)
        except Exception:
            pass
    wandb_config = config_yaml.get('wandb', {})
    if is_main_process():
        if not WANDB_AVAILABLE:
            logger.info("wandb not installed - metrics will be logged to stdout and the log file only")
        # mode: "online" (default), "offline" (sync later with `wandb sync`), or
        # "disabled" (keep wandb installed but log nothing).
        wandb.init(
            project=wandb_config.get('project', 'rna-multimodal'),
            entity=wandb_config.get('entity'),
            name=wandb_config.get('run_name'),
            config=config_yaml,
            mode=wandb_config.get('mode', 'online'),
            reinit=True
        )
    else:
        os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', 'disabled')
        # Ensure wandb.log is a no-op on non-main ranks to avoid init errors
        try:
            wandb.log = (lambda *args, **kwargs: None)
            # Guard against accidental init calls on non-main
            wandb.init = (lambda *args, **kwargs: None)
        except Exception:
            pass
    
    # Set WandB logging frequency
    wandb_log_freq = wandb_config.get('log_frequency', 100)

    # Now import torch and the rest
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    from transformers import ModernBertConfig, ModernBertForMaskedLM, get_linear_schedule_with_warmup
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
    from training_data_preparation import create_data_loaders
    import json

    def get_sequence_length(input_ids, vocab_json):
        """Get the length of the sequence modality only (not total concatenated length)."""
        seq_start = vocab_json['special_tokens']['[SEQ_START]']
        struct_start = vocab_json['special_tokens']['[STRUCT_START]']
        # Find indices
        try:
            seq_idx = input_ids.index(seq_start)
            struct_idx = input_ids.index(struct_start)
            return struct_idx - seq_idx - 1  # Exclude [SEQ_START] itself
        except ValueError:
            return 0
    
    def create_token_prediction_bar_chart(top_token_predictions, vocab_json, modality_name, phase_prefix):
        """Create WandB bar chart for top token predictions."""
        if modality_name not in top_token_predictions:
            return None
        
        token_data = top_token_predictions[modality_name]
        if not token_data['token_ids']:
            return None
        
        # Convert token IDs to readable tokens
        token_names = []
        for token_id in token_data['token_ids']:
            # Check if it's a special token first
            token_name = None
            for special_name, special_id in vocab_json['special_tokens'].items():
                if token_id == special_id:
                    token_name = special_name
                    break
            
            # If not a special token, look in modality vocabularies
            if token_name is None:
                found = False
                for modality, tokens in vocab_json['modality_vocabs'].items():
                    offset = vocab_json['modality_offsets'][modality]
                    if offset <= token_id < offset + len(tokens):
                        token_name = tokens[token_id - offset]
                        found = True
                        break
                
                if not found:
                    token_name = f"UNK_{token_id}"  # Show the token ID for debugging
            
            token_names.append(token_name)
        
        # Create bar chart
        chart = wandb.plot.bar(
            wandb.Table(data=[[name, count] for name, count in zip(token_names, token_data['counts'])], 
                       columns=["Token", "Count"]),
            "Token", "Count",
            title=f"Top 4 Predicted Tokens - {modality_name.replace('_', ' ').title()}"
        )
        
        return chart

    def get_modality_slices(token_ids, vocab_json):
        """Return a dict: modality -> (start, end) indices in token_ids."""
        special_tokens = vocab_json['special_tokens']
        starts = {
            'sequence': special_tokens['[SEQ_START]'],
            'ss': special_tokens['[STRUCT_START]'],
            'consensus': special_tokens['[CONS_START]'],
            'go_terms': special_tokens['[GO_START]']
        }
        indices = {}
        for mod, tok in starts.items():
            try:
                indices[mod] = token_ids.index(tok)
            except ValueError:
                indices[mod] = None
        # Find boundaries
        slices = {}
        order = ['sequence', 'ss', 'consensus', 'go_terms']
        for i, mod in enumerate(order):
            start = indices[mod] + 1 if indices[mod] is not None else None
            # End is next start or end of list
            next_starts = [indices[o] for o in order[i+1:] if indices[o] is not None]
            end = min(next_starts) if next_starts else len(token_ids)
            if start is not None:
                slices[mod] = (start, end)
        return slices

    def extract_sequences_from_batch(batch, modality_type_ids, phase_name, global_tracker, padding_label_id=-100, vocab_json=None):
        """Extract unique sequences from batch and add to global tracker."""
        if 'original_input_ids' in batch:
            input_ids = batch['original_input_ids']
        elif 'input_ids' in batch:
            input_ids = batch['input_ids'] 
        else:
            return
        
        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            # Get sequence modality positions (modality_type_id == 0)
            seq_mask = (modality_type_ids[b] == 0)
            
            # Filter out padding tokens if we have labels - ensure same device
            if 'labels' in batch:
                labels_b = batch['labels'][b].to(seq_mask.device)
                seq_mask = seq_mask & (labels_b != padding_label_id)
            
            # Ensure mask is on same device as input_ids before indexing
            seq_mask = seq_mask.to(input_ids[b].device, non_blocking=True)
            if seq_mask.sum() > 0:
                # Use robust indexing to avoid rare TensorAdvancedIndexing asserts
                try:
                    seq_mask = seq_mask.bool().contiguous()
                    row = input_ids[b].contiguous()
                    idx = torch.nonzero(seq_mask, as_tuple=False).squeeze(-1)
                    seq_tokens = to_list_cpu(torch.index_select(row, 0, idx))
                except Exception:
                    # Fallback to boolean indexing on CPU
                    seq_tokens = to_list_cpu(input_ids[b].to('cpu')[seq_mask.to('cpu')])
                
                # Filter out special tokens if vocab_json is provided
                if vocab_json is not None:
                    special_token_ids = set(vocab_json['special_tokens'].values())
                    seq_tokens = [token for token in seq_tokens if token not in special_token_ids]
                
                if seq_tokens:
                    seq_tuple = tuple(seq_tokens)
                    if phase_name == 'MLM':
                        global_tracker['mlm_sequences'].add(seq_tuple)
                    elif phase_name == 'Diffusion':
                        global_tracker['diffusion_sequences'].add(seq_tuple)



    def update_overall_tracker(batch, modality_type_ids, overall_tracker, padding_label_id=-100, vocab_json=None):
        """Update global overall tracker (across phases) with sequences and families from this batch."""
        if 'input_ids' not in batch:
            return
        input_ids = batch['input_ids']
        batch_size = input_ids.shape[0]
        # Simple counters - no memory growth
        overall_tracker['total_sequences'] += int(batch_size)
        if 'family_ids' in batch and batch['family_ids'] is not None:
            # Count total families (simple counter)
            overall_tracker['total_families'] += int(batch_size)
            # Count unique families using hash-based tracking (single rank only)
            if is_main_process():
                for family_id in batch['family_ids']:
                    family_str = str(family_id)
                    family_hash = hashlib.md5(family_str.encode('utf-8')).hexdigest()
                    if len(overall_tracker['unique_families_hash']) < overall_tracker['max_unique_families']:
                        overall_tracker['unique_families_hash'].add(family_hash)
        
        # Count unique sequences using hash-based tracking (with configurable memory limit)
        # Extract sequence tokens (sequence modality only, exclude padding/specials)
        # Use original_input_ids if available (for diffusion phase), otherwise use input_ids
        sequence_tokens = batch.get('original_input_ids', input_ids)
        valid_sequences_counted = 0  # Track how many valid sequences we actually count
        
        for b in range(batch_size):
            seq_mask = (modality_type_ids[b] == 0)
            if 'labels' in batch:
                labels_b = batch['labels'][b].to(seq_mask.device)
                seq_mask = seq_mask & (labels_b != padding_label_id)
            seq_mask = seq_mask.to(sequence_tokens[b].device, non_blocking=True)
            
            # Always try to extract sequence tokens, even if mask is empty
            seq_tokens = []
            if seq_mask.sum() > 0:
                try:
                    seq_mask = seq_mask.bool().contiguous()
                    row = sequence_tokens[b].contiguous()
                    idx = torch.nonzero(seq_mask, as_tuple=False).squeeze(-1)
                    seq_tokens = to_list_cpu(torch.index_select(row, 0, idx))
                except Exception:
                    seq_tokens = to_list_cpu(sequence_tokens[b].to('cpu')[seq_mask.to('cpu')])
                
                # Filter out special tokens
                if vocab_json is not None:
                    special_token_ids = set(vocab_json['special_tokens'].values())
                    seq_tokens = [token for token in seq_tokens if token not in special_token_ids]
            
            # Count this sequence for unique tracking if it has valid tokens
            if seq_tokens:
                # Create a hashable representation of the sequence for hash-based tracking
                seq_str = ','.join(map(str, seq_tokens))
                seq_hash = hashlib.md5(seq_str.encode('utf-8')).hexdigest()
                
                # Track valid sequences for unique counting (single rank only)
                valid_sequences_counted += 1
                if is_main_process():
                    if len(overall_tracker['unique_sequences_hash']) < overall_tracker['max_unique_sequences']:
                        overall_tracker['unique_sequences_hash'].add(seq_hash)
        
        # Update total sequences sampled to match actual valid sequences counted
        overall_tracker['total_sequences_sampled'] += valid_sequences_counted
        
        # Update total sequences sampled to match actual valid sequences counted
        overall_tracker['total_sequences_sampled'] += valid_sequences_counted
        


    def validate(model, val_loader, device, config, vocab_json, valid_token_indices_by_modality, apply_vocab_constraints, logger, mlm_metrics_calculator=None, diffusion_metrics_calculator=None, current_phase="MLM", noise_scheduler=None, data_preparator=None, diffusion_loss=None):
        model.eval()
        total_loss = 0.0
        total_sequences = 0  # This tracks sequences for current validation run only
        total_tokens = 0  # Initialize total_tokens for MLM validation
        num_batches = 0
        padding_label_id = config['data']['padding_label_id']
        dropped_token_id = vocab_json['special_tokens']['[DROPPED]']  # For MLM modality dropout
        # Normalize validation_num_samples to int or None
        _num_samples_cfg = config['training'].get('validation_num_samples', 128)
        num_samples = None
        try:
            if _num_samples_cfg is None:
                num_samples = None
            elif isinstance(_num_samples_cfg, str):
                _val = _num_samples_cfg.strip().lower()
                if _val in ('none', 'null', '~', 'all', 'full'):
                    num_samples = None
                else:
                    num_samples = int(float(_num_samples_cfg))
            else:
                num_samples = int(_num_samples_cfg)
        except Exception:
            # Fallback to full validation if unparsable
            num_samples = None
        batch_size = config['data']['batch_size']
        batches_seen = 0
        
        # Reset validation metrics for current phase; optionally accumulate across runs based on config
        accumulate_validation = config.get('training', {}).get('accumulate_validation_stats', False)
        # Prepare shared loss config for validation to mirror training behavior
        modality_loss_weights = config['training'].get('modality_loss_weights', {})
        modality_id_to_name = {0: 'sequence', 1: 'secondary_structure', 2: 'consensus', 3: 'go_terms'}
        consensus_case_penalty_factor = config['training'].get('consensus_case_penalty_factor', 0.5)
        # Precompute consensus letter code maps for penalty (device-aware)
        try:
            consensus_vocab = vocab_json['modality_vocabs']['consensus']
            consensus_offset = vocab_json['modality_offsets']['consensus']
            lower_list = []
            upper_list = []
            for tok in consensus_vocab:
                if isinstance(tok, str) and len(tok) == 1 and tok.isalpha():
                    lower_list.append(ord(tok.lower()))
                    upper_list.append(ord(tok.upper()))
                else:
                    lower_list.append(-1)
                    upper_list.append(-1)
            consensus_lower_codes = torch.tensor(lower_list, device=device, dtype=torch.int16)
            consensus_upper_codes = torch.tensor(upper_list, device=device, dtype=torch.int16)
        except Exception:
            # Fallback safe defaults (no penalty applied)
            consensus_vocab = []
            consensus_offset = 0
            consensus_lower_codes = torch.empty(0, device=device, dtype=torch.int16)
            consensus_upper_codes = torch.empty(0, device=device, dtype=torch.int16)
        if current_phase == "MLM" and mlm_metrics_calculator is not None:
            # Store current sequence statistics before reset
            current_validation_stats = mlm_metrics_calculator.metrics.get('validation', {}).get('training_stats', {})
            # Store the accumulated totals before reset
            accumulated_unique_sequences = current_validation_stats.get('unique_sequences', set())
            accumulated_sequence_lengths = current_validation_stats.get('sequence_lengths', [])
            accumulated_unique_families = current_validation_stats.get('unique_families', set())
            accumulated_total_sequences = current_validation_stats.get('total_sequences', 0)
            
            mlm_metrics_calculator.reset_phase('validation')
            
            # Restore sequence statistics to continue accumulation if enabled
            if accumulate_validation:
                mlm_metrics_calculator.metrics['validation']['training_stats']['unique_sequences'] = accumulated_unique_sequences
                mlm_metrics_calculator.metrics['validation']['training_stats']['sequence_lengths'] = accumulated_sequence_lengths
                mlm_metrics_calculator.metrics['validation']['training_stats']['unique_families'] = accumulated_unique_families
                mlm_metrics_calculator.metrics['validation']['training_stats']['total_sequences'] = accumulated_total_sequences
            
        elif current_phase == "Diffusion" and diffusion_metrics_calculator is not None:
            # Store current sequence statistics before reset
            current_validation_stats = diffusion_metrics_calculator.metrics.get('validation', {}).get('training_stats', {})
            # Store the accumulated totals before reset
            accumulated_unique_sequences = current_validation_stats.get('unique_sequences', set())
            accumulated_sequence_lengths = current_validation_stats.get('sequence_lengths', [])
            accumulated_unique_families = current_validation_stats.get('unique_families', set())
            accumulated_total_sequences = current_validation_stats.get('total_sequences', 0)
            
            diffusion_metrics_calculator.reset_phase('validation')
            
            # Restore sequence statistics to continue accumulation if enabled
            if accumulate_validation:
                diffusion_metrics_calculator.metrics['validation']['training_stats']['unique_sequences'] = accumulated_unique_sequences
                diffusion_metrics_calculator.metrics['validation']['training_stats']['sequence_lengths'] = accumulated_sequence_lengths
                diffusion_metrics_calculator.metrics['validation']['training_stats']['unique_families'] = accumulated_unique_families
                diffusion_metrics_calculator.metrics['validation']['training_stats']['total_sequences'] = accumulated_total_sequences
        
        # Create a random sampler for validation to ensure diverse sampling
        val_dataset = val_loader.dataset
        _val_len = len(val_dataset)
        stale_validation = config.get('training', {}).get('stale_validation', False)
        if stale_validation:
            # Cache fixed indices in config for repeatability across runs in the same process
            if '_fixed_validation_indices' not in config:
                if num_samples is None or num_samples >= _val_len or num_samples <= 0:
                    fixed_idx = torch.arange(_val_len)
                else:
                    # Deterministic subset: first N indices
                    fixed_idx = torch.arange(_val_len)[:num_samples]
                config['_fixed_validation_indices'] = fixed_idx
            random_indices = config['_fixed_validation_indices']
        else:
            if num_samples is None or num_samples >= _val_len or num_samples <= 0:
                random_indices = torch.arange(_val_len)
            else:
                random_indices = torch.randperm(_val_len)[:num_samples]
        
        with torch.no_grad():
            for i in range(0, len(random_indices), batch_size):
                batch_indices = random_indices[i:i+batch_size]
                if len(batch_indices) == 0:
                    continue
                
                # Get batch from dataset using random indices
                batch_data = [val_dataset[idx] for idx in batch_indices]
                
                # Collate the batch manually
                batch = val_loader.collate_fn(batch_data)
                
                if batch['input_ids'].shape[0] == 0:
                    continue
                
                # Stop when we've processed enough sequences
                if num_samples is not None and total_sequences >= num_samples:
                    break
                
                if current_phase == "MLM":
                    # MLM validation (existing logic)
                    modality_type_ids = batch['modality_type_ids'].to(device)
                    
                    # Debug: Check if model is in eval mode
                    if batches_seen == 1 and is_main_process():
                        logger.info(f"🔍 Validation model training mode: {model.training}")
                        logger.info(f"🔍 Validation model device: {next(model.parameters()).device}")
                    
                    # Run validation forward pass in eager mode to avoid torch.compile/Inductor issues
                    def _val_forward_mlm(inp_ids, attn_mask):
                        return model(inp_ids, attention_mask=attn_mask)
                    try:
                        import torch._dynamo as _dyn
                        _val_forward_mlm = _dyn.disable(_val_forward_mlm)
                    except Exception:
                        pass
                    model_outputs = _val_forward_mlm(
                        batch['input_ids'].to(device),
                        batch['attention_mask'].to(device)
                    )
                    model_logits = model_outputs['logits'] if isinstance(model_outputs, dict) else model_outputs.logits
                    masked_logits = apply_modality_vocab_mask(model_logits, modality_type_ids, valid_token_indices_by_modality, apply_vocab_constraints)
                    labels = batch['labels'].to(device)
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=padding_label_id)
                    # Compute weighted per-modality loss identically to training
                    modality_losses = {}
                    weighted_loss = 0.0
                    for mod_id, mod_name in modality_id_to_name.items():
                        mod_mask = (modality_type_ids == mod_id) & (labels != padding_label_id)
                        if mod_mask.sum() > 0:
                            mod_logits = masked_logits[mod_mask]
                            mod_labels = labels[mod_mask]
                            if mod_name == 'consensus':
                                ce_loss = torch.nn.functional.cross_entropy(mod_logits, mod_labels, reduction='none')
                                pred_ids = mod_logits.argmax(dim=-1)
                                # Vectorized case-penalty (same as training)
                                def ids_to_codes(ids):
                                    rel = ids - consensus_offset
                                    in_range = (rel >= 0) & (rel < len(consensus_vocab))
                                    codes_lower = torch.full_like(rel, fill_value=-1, dtype=torch.int16)
                                    codes_upper = torch.full_like(rel, fill_value=-1, dtype=torch.int16)
                                    rel_clamped = rel.clamp(min=0, max=len(consensus_vocab)-1)
                                    codes_lower[in_range] = consensus_lower_codes[rel_clamped[in_range]]
                                    codes_upper[in_range] = consensus_upper_codes[rel_clamped[in_range]]
                                    return codes_lower, codes_upper, in_range
                                t_lower, t_upper, _ = ids_to_codes(mod_labels)
                                p_lower, p_upper, _ = ids_to_codes(pred_ids)
                                same_letter_ignoring_case = (t_lower == p_lower) & (t_lower != -1)
                                different_case = (t_upper != p_upper)
                                penalty_mask = same_letter_ignoring_case & different_case
                                penalties = torch.where(
                                    penalty_mask,
                                    torch.as_tensor(consensus_case_penalty_factor, device=ce_loss.device, dtype=ce_loss.dtype),
                                    torch.as_tensor(1.0, device=ce_loss.device, dtype=ce_loss.dtype),
                                )
                                ce_loss = ce_loss * penalties
                                mod_loss_scalar = ce_loss.mean()
                                weighted_loss = weighted_loss + modality_loss_weights[mod_name] * mod_loss_scalar
                                modality_losses[mod_name] = to_float(mod_loss_scalar)
                            else:
                                mod_loss_scalar = loss_fct(mod_logits, mod_labels)
                                weighted_loss = weighted_loss + modality_loss_weights[mod_name] * mod_loss_scalar
                                modality_losses[mod_name] = to_float(mod_loss_scalar)
                        else:
                            modality_losses[mod_name] = 0.0
                    loss = weighted_loss
                    total_loss += to_float(loss) * int(labels.shape[0])
                    num_batches += 1
                    batches_seen += 1
                    
                    # Debug: Log first few validation losses to see if they're changing
                    if batches_seen <= 3 and is_main_process():
                        logger.info(f"🔍 Validation batch {batches_seen}: loss={to_float(loss):.6f}, batch_size={labels.shape[0]}")
                    
                    # Update metrics if calculator is provided
                    if mlm_metrics_calculator is not None:
                        # Compute per-modality losses
                        modality_losses = {}
                        for mod_id, mod_name in zip([0,1,2,3], ['sequence','secondary_structure','consensus','go_terms']):
                            mod_mask = (modality_type_ids == mod_id) & (labels != padding_label_id) & (batch['input_ids'].to(device) != dropped_token_id)
                            if mod_mask.sum() > 0:
                                mod_logits = masked_logits[mod_mask]
                                mod_labels = labels[mod_mask]
                                mod_loss = to_float(loss_fct(mod_logits, mod_labels))
                                modality_losses[mod_name] = mod_loss
                        
                        mlm_metrics_calculator.update_batch_metrics(
                                predictions=masked_logits,
                                targets=labels,
                                modality_type_ids=modality_type_ids,
                                phase='validation',
                                modality_losses=modality_losses,
                            overall_loss=to_float(loss),
                                family_ids=batch.get('family_ids', None),
                                input_ids=batch['input_ids'].to(device),
                                mlm_dropout_info=batch.get('mlm_dropout_info', None),
                                original_input_ids=batch.get('original_input_ids', batch['input_ids']).to(device)
                            )
                    
                    # Update sequence count for MLM validation
                    total_sequences += batch['input_ids'].shape[0]  # Count sequences, not tokens
                
                elif current_phase == "Diffusion":
                    # Diffusion validation - use passed components to avoid re-initialization
                    if noise_scheduler is None or data_preparator is None or diffusion_loss is None:
                        logger.error("Diffusion components not provided for validation")
                        continue
                    
                    # CRITICAL FIX: Apply the same data preparation as training
                    # The batch from validation loader doesn't have modality dropout applied
                    # We need to apply it manually to match training conditions
                    base_batch = {
                        'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        'modality_type_ids': batch['modality_type_ids'],
                        'original_input_ids': batch.get('original_input_ids', batch['input_ids']),
                        'family_ids': batch.get('family_ids', [])
                    }
                    
                    # Apply modality dropout for validation (same as training); make deterministic if requested
                    base_seed = int(config.get('_resolved_random_seed', 1337))
                    dropout_seed_base = base_seed if config.get('training', {}).get('stale_validation', False) else None
                    sample_keys = batch.get('data_indices', None)
                    sample_keys_list = sample_keys.tolist() if isinstance(sample_keys, torch.Tensor) else None
                    diffusion_batch = data_preparator.prepare_conditional_batch(
                        base_batch, training=True,
                        dropout_seed_base=dropout_seed_base,
                        sample_keys=sample_keys_list
                    )
                    
                    # Get conditioning info from properly prepared batch
                    conditioning_info = diffusion_batch.get('conditioning_info', {})
                    sequence_mask = conditioning_info.get('sequence_mask', torch.ones_like(diffusion_batch['input_ids'], dtype=torch.bool)).to(device)
                    conditioning_mask = conditioning_info.get('conditioning_mask', torch.zeros_like(diffusion_batch['input_ids'], dtype=torch.bool)).to(device)
                    
                    # Timesteps: deterministic if stale_validation enabled
                    if config.get('training', {}).get('stale_validation', False) and sample_keys_list is not None:
                        base_seed = int(config.get('_resolved_random_seed', 1337))
                        ts = [int((base_seed ^ int(k)) % noise_scheduler.num_timesteps) for k in sample_keys_list]
                        timesteps = torch.tensor(ts, device=device, dtype=torch.long)
                        rng_seeds = torch.tensor([base_seed ^ int(k) for k in sample_keys_list], device=device, dtype=torch.long)
                    else:
                        timesteps = torch.randint(0, noise_scheduler.num_timesteps, (diffusion_batch['input_ids'].shape[0],), device=device)
                        rng_seeds = None
                    
                    # Use properly prepared tokens (with modality dropout applied)
                    modality_dropped_tokens = diffusion_batch['input_ids'].to(device)
                    original_tokens = diffusion_batch.get('original_input_ids', diffusion_batch['input_ids']).to(device)
                    
                    noisy_tokens, noise_info = noise_scheduler.add_noise(
                        modality_dropped_tokens, 
                        timesteps, 
                        sequence_mask,
                        rng_seeds_per_sample=rng_seeds
                    )
                    
                    # Forward pass with noisy tokens (disable compile for validation)
                    def _val_forward_diff(noisy, attn):
                        return model(noisy, attention_mask=attn)
                    try:
                        import torch._dynamo as _dyn
                        _val_forward_diff = _dyn.disable(_val_forward_diff)
                    except Exception:
                        pass
                    
                    # Debug: Check diffusion validation model state
                    if batches_seen == 1 and is_main_process():
                        logger.info(f"🔍 Diffusion validation model training mode: {model.training}")
                        logger.info(f"🔍 Diffusion validation model device: {next(model.parameters()).device}")
                    
                    model_outputs = _val_forward_diff(
                        noisy_tokens,
                        diffusion_batch['attention_mask'].to(device)
                    )
                    model_logits = model_outputs['logits'] if isinstance(model_outputs, dict) else model_outputs.logits
                    
                    # CRITICAL FIX: Apply vocabulary masking to prevent invalid token predictions
                    masked_logits = apply_modality_vocab_mask(model_logits, diffusion_batch['modality_type_ids'].to(device), valid_token_indices_by_modality, apply_vocab_constraints)
                    
                    # Compute diffusion loss (identical to training)
                    loss_dict = diffusion_loss(
                        predictions=masked_logits,  # Use masked logits
                        targets=original_tokens,
                        timesteps=timesteps, 
                        sequence_mask=sequence_mask,
                        conditioning_mask=conditioning_mask,
                        noise_info=noise_info
                    )
                    loss = loss_dict['total_loss']
                    sequence_loss = loss_dict['sequence_loss']
                    
                    # Debug: Log first few diffusion validation losses
                    if batches_seen <= 3 and is_main_process():
                        logger.info(f"🔍 Diffusion validation batch {batches_seen}: loss={to_float(loss):.6f}, batch_size={diffusion_batch['input_ids'].shape[0]}")
                    
                    # Update diffusion metrics for validation
                    if diffusion_metrics_calculator is not None:
                        diffusion_metrics_calculator.update_batch_metrics(
                                predictions=masked_logits,  # Use masked logits
                                targets=original_tokens,
                                timesteps=timesteps,
                                sequence_mask=sequence_mask,
                                noise_info=noise_info,
                                conditioning_info=conditioning_info,
                            loss=to_float(loss),
                                phase='validation',
                                family_ids=diffusion_batch.get('family_ids', None)
                        )
                    
                    total_loss += to_float(loss) * int(diffusion_batch['input_ids'].shape[0])
                    total_sequences += diffusion_batch['input_ids'].shape[0]  # Count sequences, not tokens
                    num_batches += 1
                    batches_seen += 1
                
                if num_samples is not None and total_sequences >= num_samples:
                    break
        
        # Calculate and log metrics if calculator is provided
        validation_results = None
        if current_phase == "MLM" and mlm_metrics_calculator is not None:
            validation_results = mlm_metrics_calculator.calculate_metrics(phase='validation')
            mlm_metrics_calculator.log_metrics('validation', 0, validation_results)
        elif current_phase == "Diffusion" and diffusion_metrics_calculator is not None:
            validation_results = diffusion_metrics_calculator.calculate_metrics(phase='validation')
            diffusion_metrics_calculator.log_metrics('validation', 0, validation_results)
        
        avg_loss = total_loss / max(1, total_sequences)
        logger.info(f"[VALIDATION] Loss: {avg_loss:.4f} over {total_sequences} sequences ({num_batches} batches)")
        
        # Log validation statistics (current run + accumulated)
        if validation_results and 'training_stats' in validation_results:
            stats = validation_results['training_stats']
            # calculate_metrics returns integers, not sets
            accumulated_unique_seen = stats.get('unique_sequences', 0)
            accumulated_total_seen = stats.get('total_sequences', 0)
            total_families_seen = stats.get('unique_families', 0)
            sequence_lengths = stats.get('sequence_lengths', [])
            
            # Calculate redundancy rate
            redundancy_rate = 0.0
            if accumulated_total_seen > 0:
                redundancy_rate = (accumulated_total_seen - accumulated_unique_seen) / accumulated_total_seen * 100.0
            
            if sequence_lengths:
                min_len = min(sequence_lengths)
                max_len = max(sequence_lengths)
                avg_len = sum(sequence_lengths) / len(sequence_lengths)
                logger.info(f"[VALIDATION STATS] Current run: {total_sequences} sequences")
                logger.info(f"[VALIDATION STATS] Accumulated: {accumulated_total_seen} total, {accumulated_unique_seen} unique, {total_families_seen} families")
                logger.info(f"[VALIDATION STATS] Redundancy: {redundancy_rate:.1f}% | Sequence lengths: min={min_len}, max={max_len}, avg={avg_len:.1f}")
            else:
                logger.info(f"[VALIDATION STATS] Current run: {total_sequences} sequences")
                logger.info(f"[VALIDATION STATS] Accumulated: {accumulated_total_seen} total, {accumulated_unique_seen} unique, {total_families_seen} families")
                logger.info(f"[VALIDATION STATS] Redundancy: {redundancy_rate:.1f}%")
        
        # Return metrics if available
        if validation_results is not None:
            return avg_loss, validation_results
        
        model.train()
        return avg_loss

    def evaluate(model, data_loader, device, config, vocab_json, valid_token_indices_by_modality, apply_vocab_constraints, logger, split_name="EVAL", num_samples=None, mlm_metrics_calculator=None, diffusion_metrics_calculator=None, current_phase="MLM", noise_scheduler=None, data_preparator=None, diffusion_loss=None):
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        padding_label_id = config['data']['padding_label_id']
        dropped_token_id = vocab_json['special_tokens']['[DROPPED]']  # For MLM modality dropout
        batches_seen = 0
        
        # Reset evaluation metrics for current phase (but preserve sequence statistics)
        if current_phase == "MLM" and mlm_metrics_calculator is not None:
            # Store current sequence statistics before reset
            current_eval_stats = mlm_metrics_calculator.metrics.get('evaluation', {}).get('training_stats', {})
            mlm_metrics_calculator.reset_phase('evaluation')
            # Restore sequence statistics (don't reset these)
            if current_eval_stats:
                mlm_metrics_calculator.metrics['evaluation']['training_stats']['unique_sequences'] = current_eval_stats.get('unique_sequences', set())
                mlm_metrics_calculator.metrics['evaluation']['training_stats']['sequence_lengths'] = current_eval_stats.get('sequence_lengths', [])
                mlm_metrics_calculator.metrics['evaluation']['training_stats']['unique_families'] = current_eval_stats.get('unique_families', set())
        elif current_phase == "Diffusion" and diffusion_metrics_calculator is not None:
            # Store current sequence statistics before reset
            current_eval_stats = diffusion_metrics_calculator.metrics.get('evaluation', {}).get('training_stats', {})
            diffusion_metrics_calculator.reset_phase('evaluation')
            # Restore sequence statistics (don't reset these)
            if current_eval_stats:
                diffusion_metrics_calculator.metrics['evaluation']['training_stats']['unique_sequences'] = current_eval_stats.get('unique_sequences', set())
                diffusion_metrics_calculator.metrics['evaluation']['training_stats']['sequence_lengths'] = current_eval_stats.get('sequence_lengths', [])
                diffusion_metrics_calculator.metrics['evaluation']['training_stats']['unique_families'] = current_eval_stats.get('unique_families', set())
        
        with torch.no_grad():
            for batch in data_loader:
                if batch['input_ids'].shape[0] == 0:
                    continue
                if num_samples is not None and total_tokens >= num_samples:
                    break
                
                if current_phase == "MLM":
                    # MLM evaluation (existing logic)
                    modality_type_ids = batch['modality_type_ids'].to(device)
                    model_logits = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))['logits']
                    masked_logits = apply_modality_vocab_mask(model_logits, modality_type_ids, valid_token_indices_by_modality, apply_vocab_constraints)
                    labels = batch['labels'].to(device)
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=padding_label_id)
                    
                    # CRITICAL: Exclude [DROPPED] token positions from validation loss calculation
                    dropped_token_id = vocab_json['special_tokens']['[DROPPED]']
                    active_loss = (labels.view(-1) != padding_label_id) & (batch['input_ids'].to(device).view(-1) != dropped_token_id)
                    active_logits = masked_logits.view(-1, masked_logits.size(-1))[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                    total_loss += to_float(loss) * int(active_labels.numel())
                    total_tokens += active_labels.numel()
                    num_batches += 1
                    batches_seen += 1
                    
                    # Update metrics if calculator is provided
                    if mlm_metrics_calculator is not None:
                        # Compute per-modality losses
                        modality_losses = {}
                        for mod_id, mod_name in zip([0,1,2,3], ['sequence','secondary_structure','consensus','go_terms']):
                            mod_mask = (modality_type_ids == mod_id) & (labels != padding_label_id) & (batch['input_ids'].to(device) != dropped_token_id)
                            if mod_mask.sum() > 0:
                                mod_logits = masked_logits[mod_mask]
                                mod_labels = labels[mod_mask]
                                mod_loss = to_float(loss_fct(mod_logits, mod_labels))
                                modality_losses[mod_name] = mod_loss
                        
                        mlm_metrics_calculator.update_batch_metrics(
                                predictions=masked_logits,
                                targets=labels,
                                modality_type_ids=modality_type_ids,
                                phase='evaluation',
                                modality_losses=modality_losses,
                            overall_loss=to_float(loss),
                                family_ids=batch.get('family_ids', None),
                                input_ids=batch['input_ids'].to(device),
                                mlm_dropout_info=batch.get('mlm_dropout_info', None),
                                original_input_ids=batch.get('original_input_ids', batch['input_ids']).to(device)
                        )
                
                elif current_phase == "Diffusion":
                    # Diffusion evaluation - use passed components to avoid re-initialization
                    if noise_scheduler is None or data_preparator is None or diffusion_loss is None:
                        logger.error("Diffusion components not provided for evaluation")
                        continue
                    
                    # CRITICAL FIX: Apply the same data preparation as training
                    # The batch from evaluation loader doesn't have modality dropout applied
                    # We need to apply it manually to match training conditions
                    base_batch = {
                        'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        'modality_type_ids': batch['modality_type_ids'],
                        'original_input_ids': batch.get('original_input_ids', batch['input_ids']),
                        'family_ids': batch.get('family_ids', [])
                    }
                    
                    # Apply modality dropout for evaluation (same as training)
                    diffusion_batch = data_preparator.prepare_conditional_batch(base_batch, training=True)
                    
                    # Get conditioning info from properly prepared batch
                    conditioning_info = diffusion_batch.get('conditioning_info', {})
                    sequence_mask = conditioning_info.get('sequence_mask', torch.ones_like(diffusion_batch['input_ids'], dtype=torch.bool)).to(device)
                    conditioning_mask = conditioning_info.get('conditioning_mask', torch.zeros_like(diffusion_batch['input_ids'], dtype=torch.bool)).to(device)
                    
                    # Use random timesteps for evaluation
                    timesteps = torch.randint(0, noise_scheduler.num_timesteps, (diffusion_batch['input_ids'].shape[0],), device=device)
                    
                    # Use properly prepared tokens (with modality dropout applied)
                    modality_dropped_tokens = diffusion_batch['input_ids'].to(device)
                    original_tokens = diffusion_batch.get('original_input_ids', diffusion_batch['input_ids']).to(device)
                    
                    noisy_tokens, noise_info = noise_scheduler.add_noise(
                        modality_dropped_tokens, 
                        timesteps, 
                        sequence_mask
                    )
                    
                    # Forward pass with noisy tokens
                    model_logits = model(noisy_tokens, attention_mask=diffusion_batch['attention_mask'].to(device))['logits']
                    
                    # CRITICAL FIX: Apply vocabulary masking to prevent invalid token predictions
                    masked_logits = apply_modality_vocab_mask(model_logits, diffusion_batch['modality_type_ids'].to(device), valid_token_indices_by_modality, apply_vocab_constraints)
                    
                    # Compute diffusion loss
                    loss_dict = diffusion_loss(
                        predictions=masked_logits,  # Use masked logits
                        targets=original_tokens,
                        timesteps=timesteps, 
                        sequence_mask=sequence_mask,
                        conditioning_mask=conditioning_mask,
                        noise_info=noise_info
                    )
                    loss = loss_dict['total_loss']
                    sequence_loss = loss_dict['sequence_loss']
                    
                    # Update diffusion metrics for evaluation
                    if diffusion_metrics_calculator is not None:
                        diffusion_metrics_calculator.update_batch_metrics(
                                predictions=masked_logits,  # Use masked logits
                                targets=original_tokens,
                                timesteps=timesteps,
                                sequence_mask=sequence_mask,
                                noise_info=noise_info,
                                conditioning_info=conditioning_info,
                            loss=to_float(loss),
                                phase='evaluation',
                                family_ids=diffusion_batch.get('family_ids', None)
                        )
                    
                    total_loss += to_float(loss) * int(diffusion_batch['input_ids'].shape[0])
                    total_tokens += diffusion_batch['input_ids'].shape[0]
                    num_batches += 1
                    batches_seen += 1
                
                if num_samples is not None and total_tokens >= num_samples:
                    break
        
        # Calculate and log metrics if calculator is provided
        evaluation_results = None
        if current_phase == "MLM" and mlm_metrics_calculator is not None:
            evaluation_results = mlm_metrics_calculator.calculate_metrics(phase='evaluation')
            mlm_metrics_calculator.log_metrics('evaluation', 0, evaluation_results)
        elif current_phase == "Diffusion" and diffusion_metrics_calculator is not None:
            evaluation_results = diffusion_metrics_calculator.calculate_metrics(phase='evaluation')
            diffusion_metrics_calculator.log_metrics('evaluation', 0, evaluation_results)
        
        avg_loss = total_loss / max(1, total_tokens)
        logger.info(f"[{split_name}] Loss: {avg_loss:.4f} over {total_tokens} tokens ({num_batches} batches)")
        
        # Return metrics if available
        if evaluation_results is not None:
            return avg_loss, evaluation_results
        
        model.train()
        return avg_loss

def setup_ram_data_loading(config, logger):
    """Load essential data files into RAM for maximum performance."""
    if not config['data_files'].get('load_data_into_ram', False):
        logger.info("💾 Using disk-based data loading (RAM loading disabled)")
        return config
    
    logger.info("🚀 Setting up RAM-based data loading for maximum performance...")
    
    # Files to load into RAM
    files_to_load = [
        'h5_file',
        'vocabulary_file', 
        'family_splits_file',
        'family_lookup_file'
    ]
    
    # Load files into RAM and update config
    import h5py
    import json
    import tempfile
    import time as _time
    
    # Check if /dev/shm is available (Linux shared memory)
    if not os.path.exists('/dev/shm'):
        logger.warning("⚠️ /dev/shm not available, falling back to /tmp for RAM-like storage")
        ram_base = '/tmp'
    else:
        ram_base = '/dev/shm'
    
    # Track RAM files for cleanup
    ram_files = []
    
    # Consider distributed if torch.distributed is initialized OR launcher env is set
    ddp_enabled = (dist.is_available() and dist.is_initialized()) or int(os.environ.get('WORLD_SIZE', '1')) > 1
    for file_key in files_to_load:
        original_path = config['data_files'][file_key]
        
        # Check if original file exists
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"❌ Original data file not found: {original_path}")
        
        filename = os.path.basename(original_path)
        original_size = os.path.getsize(original_path) / 1024**2  # MB
        
        if is_main_process():
            logger.info(f"📋 Loading {filename} ({original_size:.1f} MB) into RAM...")
        
        try:
            ram_path = f"{ram_base}/{filename}"
            tmp_path = f"{ram_path}.tmp"
            lock_path = f"{ram_path}.lock"
            if file_key == 'h5_file':
                # Single-writer lock using filesystem; atomic rename guards readers
                have_lock = False
                if ddp_enabled:
                    try:
                        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.close(fd)
                        have_lock = True
                    except FileExistsError:
                        have_lock = False
                else:
                    have_lock = True

                if have_lock:
                    # This process performs the copy into tmp, then atomically replaces final
                    with h5py.File(original_path, 'r') as src_file:
                        with h5py.File(tmp_path, 'w') as dst_file:
                            def copy_group(src_group, dst_group):
                                for key, item in src_group.items():
                                    if isinstance(item, h5py.Group):
                                        new_group = dst_group.create_group(key)
                                        copy_group(item, new_group)
                                    else:
                                        dst_group.create_dataset(key, data=item)
                            copy_group(src_file, dst_file)
                    # Atomic move into place
                    os.replace(tmp_path, ram_path)
                    # Release lock
                    try:
                        os.remove(lock_path)
                    except Exception:
                        pass
                    config['data_files'][file_key] = ram_path
                    ram_files.append(ram_path)
                    if is_main_process():
                        logger.info(f"✅ Loaded {filename} into RAM at {ram_path}")
                else:
                    # Wait for final file to appear (not tmp) and be readable
                    wait_start = _time.time()
                    while True:
                        if os.path.exists(ram_path):
                            try:
                                with h5py.File(ram_path, 'r') as _f:
                                    pass
                                break
                            except Exception:
                                # File exists but not yet complete; keep waiting
                                pass
                        _time.sleep(0.1)
                        if _time.time() - wait_start > 1800:  # 30 min safety
                            raise TimeoutError(f"Timeout waiting for RAM file {ram_path} to be ready")
                    config['data_files'][file_key] = ram_path
            else:
                # JSON copy with lock+atomic move
                have_lock = False
                if ddp_enabled:
                    try:
                        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.close(fd)
                        have_lock = True
                    except FileExistsError:
                        have_lock = False
                else:
                    have_lock = True

                if have_lock:
                    with open(original_path, 'r') as f:
                        data = json.load(f)
                    with open(tmp_path, 'w') as f:
                        json.dump(data, f)
                    os.replace(tmp_path, ram_path)
                    try:
                        os.remove(lock_path)
                    except Exception:
                        pass
                    config['data_files'][file_key] = ram_path
                    ram_files.append(ram_path)
                    if is_main_process():
                        logger.info(f"✅ Loaded {filename} into RAM at {ram_path}")
                else:
                    wait_start = _time.time()
                    while not os.path.exists(ram_path):
                        _time.sleep(0.1)
                        if _time.time() - wait_start > 300:
                            raise TimeoutError(f"Timeout waiting for RAM file {ram_path} to be ready")
                    config['data_files'][file_key] = ram_path
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load {original_path} into RAM: {e}") from e
    
    # Synchronize all ranks before proceeding so everyone sees the files
    if ddp_enabled:
        try:
            # Add timeout to barrier to prevent indefinite hangs
            dist.barrier(timeout=datetime.timedelta(seconds=300))  # 5 minute timeout
        except Exception as e:
            logger.warning(f"⚠️ Distributed barrier failed: {e}")
            # Continue anyway - may cause minor sync issues but prevents deadlocks
            pass
    # Store RAM files for cleanup (rank 0 only)
    config['_ram_files'] = ram_files
    if is_main_process():
        logger.info("🎉 RAM data loading complete! All essential data loaded into RAM for maximum performance.")
    return config

def cleanup_ram_files(config, logger):
    """Clean up RAM files after training."""
    ram_files = config.get('_ram_files', [])
    if not ram_files:
        return
    
    logger.info("🧹 Cleaning up RAM files...")
    for ram_path in ram_files:
        try:
            if os.path.exists(ram_path):
                os.remove(ram_path)
                logger.info(f"✅ Removed RAM file: {ram_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to remove RAM file {ram_path}: {e}")
    
    logger.info("🎉 RAM cleanup complete!")

def setup_local_data_if_needed(config, logger):
    """Copy data to local storage if temporary_processing_directory is specified."""
    temp_dir = config['data_files'].get('temporary_processing_directory')
    if not temp_dir:
        logger.info("📁 Using original data paths (no local storage setup)")
        return config
    
    # Expand environment variables (like $L_SCRATCH)
    temp_dir = os.path.expandvars(temp_dir)
    if not temp_dir:
        logger.warning("⚠️ temporary_processing_directory is empty after environment variable expansion")
        return config
    
    logger.info(f"🚀 Setting up local data storage at: {temp_dir}")
    
    # Create local directory
    try:
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"✅ Created local directory: {temp_dir}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to create local directory {temp_dir}: {e}")
    
    # Files to copy
    files_to_copy = [
        'h5_file',
        'vocabulary_file', 
        'family_splits_file',
        'family_lookup_file'
    ]
    
    # Copy files and update paths
    import shutil
    for file_key in files_to_copy:
        original_path = config['data_files'][file_key]
        
        # Check if original file exists
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"❌ Original data file not found: {original_path}")
        
        # Create local path
        filename = os.path.basename(original_path)
        local_path = os.path.join(temp_dir, filename)
        
        logger.info(f"📋 Copying {filename} to local storage...")
        
        try:
            # Copy file
            shutil.copy2(original_path, local_path)
            
            # Update config to use local path
            config['data_files'][file_key] = local_path
            
            # Log file sizes
            original_size = os.path.getsize(original_path) / 1024**2  # MB
            logger.info(f"✅ Copied {filename} ({original_size:.1f} MB) to {local_path}")
            
        except Exception as e:
            raise RuntimeError(f"❌ Failed to copy {original_path} to {local_path}: {e}")
    
    logger.info("🎉 Local data setup complete! All data files copied to fast local storage.")
    return config

def train(args):
    print("🔍 Starting training function...")
    # 1. Load config
    print(f"📋 Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Config loaded successfully")

    # Cheap re-check: __main__ already ran this before touching W&B or DDP, but
    # train() can also be called directly.
    preflight_data_files(config, args.config, logger)

    # Runtime perf toggles (A100-safe)
    try:
        import os as _os
        enable_cuda_graphs_debug_if_needed(config, logger)
        torch._dynamo.config.capture_scalar_outputs = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        _os.environ.setdefault('TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS', '1')
    except Exception:
        pass
    
    print("🔍 Setting up data loading strategy...")
    # 2. Setup data loading strategy (RAM first, then local if needed)
    if config['data_files'].get('load_data_into_ram', False):
        # Skip local data setup if RAM loading is enabled
        print("🚀 RAM loading enabled - skipping local data setup")
        logger.info("🚀 RAM loading enabled - skipping local data setup")
        config = setup_ram_data_loading(config, logger)
    else:
        # Only setup local data if RAM loading is disabled
        print("📁 Local data setup enabled")
        config = setup_local_data_if_needed(config, logger)
    print("✅ Data loading strategy setup complete")
    
    # Device selection from config
    device_config = config.get('device', 'auto')
    if (dist.is_available() and dist.is_initialized() and torch.cuda.is_available()):
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(local_rank)
    else:
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_config == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = torch.device('cpu')
        else:
            device = torch.device(device_config)
    
    logger.info(f"🔧 Using device: {device}")
    
    # Handle random seed setting
    random_seed = config.get('random_seed', 42)
    if random_seed == 'random':
        import random
        random_seed = random.randint(1, 1000000)
        logger.info(f"🎲 Using random seed: {random_seed}")
    else:
        random_seed = int(random_seed)
        logger.info(f"🎲 Using fixed seed: {random_seed}")
    # Expose resolved seed to the rest of the pipeline (used by stale_validation)
    config['_resolved_random_seed'] = int(random_seed)
    
    # Rank-aware seeding
    rank_offset = int(os.environ.get('RANK', '0')) if (dist.is_available() and dist.is_initialized()) else 0
    torch.manual_seed(random_seed + rank_offset)
    try:
        import random as _py_random
        _py_random.seed(random_seed + rank_offset)
        np.random.seed(random_seed + rank_offset)
    except Exception:
        pass

    # 2. Create data loaders using the tested pipeline
    h5_path = config['data_files']['h5_file']
    vocab_path = config['data_files']['vocabulary_file']
    family_lookup_path = config['data_files']['family_lookup_file']
    train_loader, val_loader, test_loader = create_data_loaders(args.config, h5_path, vocab_path, family_lookup_path)
    # Swap in DistributedSamplers when running DDP
    if dist.is_available() and dist.is_initialized():
        logger.info("🔗 Using DistributedFamilyDiverseSampler for DDP")
        train_collate = getattr(train_loader, 'collate_fn', None)
        val_collate = getattr(val_loader, 'collate_fn', None)
        test_collate = getattr(test_loader, 'collate_fn', None)
        # Get distributed training parameters
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        batch_size = config['data']['batch_size']
        random_seed = config.get('random_seed', 42)
        if random_seed == 'random':
            random_seed = 42  # Fallback to fixed seed for distributed training
        
        # Create distributed simple family diverse samplers
        train_sampler = DistributedSimpleFamilyDiverseSampler(
            train_loader.dataset, world_size, rank, batch_size, family_lookup_path, 
            shuffle=True, seed=random_seed, drop_last=True
        )
        val_sampler = DistributedSimpleFamilyDiverseSampler(
            val_loader.dataset, world_size, rank, batch_size, family_lookup_path, 
            shuffle=False, seed=random_seed, drop_last=False
        )
        test_sampler = DistributedSimpleFamilyDiverseSampler(
            test_loader.dataset, world_size, rank, batch_size, family_lookup_path, 
            shuffle=False, seed=random_seed, drop_last=False
        )
        train_loader = create_optimized_data_loader(train_loader.dataset, config, sampler=train_sampler, collate_fn=train_collate, training=True)
        val_loader   = create_optimized_data_loader(val_loader.dataset,   config, sampler=val_sampler,   collate_fn=val_collate,   training=False)
        test_loader  = create_optimized_data_loader(test_loader.dataset,  config, sampler=test_sampler,  collate_fn=test_collate,  training=False)

    # 3. Get vocab size and special token ids from the loaded vocabulary JSON
    with open(vocab_path, 'r') as f:
        vocab_json = json.load(f)
    vocab_size = max(int(v) for v in vocab_json['global_to_local'].keys()) + 1 if 'global_to_local' in vocab_json else 0
    pad_token_id = vocab_json['special_tokens']['[PAD]']
    mask_token_id = vocab_json['special_tokens']['[MASK]']
    cls_token_id = vocab_json['special_tokens']['[CLS]']

    # --- Precompute valid token indices for each modality ---
    # Map: modality_type_id -> set of valid global token indices
    valid_token_indices_by_modality = {
        0: set(range(vocab_json['modality_offsets']['sequence'], vocab_json['modality_offsets']['sequence'] + len(vocab_json['modality_vocabs']['sequence']))),
        1: set(range(vocab_json['modality_offsets']['ss'], vocab_json['modality_offsets']['ss'] + len(vocab_json['modality_vocabs']['ss']))),
        2: set(range(vocab_json['modality_offsets']['consensus'], vocab_json['modality_offsets']['consensus'] + len(vocab_json['modality_vocabs']['consensus']))),
        3: set(range(vocab_json['modality_offsets']['go_terms'], vocab_json['modality_offsets']['go_terms'] + len(vocab_json['modality_vocabs']['go_terms'])))
    }
    # --- Debug print: valid token indices for each modality ---
    logger.info("Valid token indices by modality:")
    for mod_id, indices in valid_token_indices_by_modality.items():
        logger.info(f"  Modality {mod_id}: min={min(indices)}, max={max(indices)}, count={len(indices)}")
        logger.info(f"    Example tokens: {[vocab_json['global_to_local'][str(i)] if str(i) in vocab_json['global_to_local'] else '?' for i in list(indices)[:5]]}")

    # 4. Setup Modern Optimizations
    logger.info("🚀 Setting up modern transformer optimizations...")
    optimizations = setup_optimizations(config, logger)
        
    # 5. Model with optimizations
    model_config_dict = create_optimized_model_config(config, vocab_size, pad_token_id, mask_token_id, cls_token_id, logger)
    model_config = ModernBertConfig(**model_config_dict)
    model = ModernBertForMaskedLM(model_config)
    model.resize_token_embeddings(vocab_size)
    
    # Simple forward-only warmup across buckets to precompile likely shapes
    try:
        warmup_model_buckets_forward_only(model, config, device, vocab_size, logger)
    except Exception:
        pass

    # Checkpoint resumption logic
    start_step = 0
    checkpoint_config = config.get('checkpoint', {})
    resume_from_checkpoint = checkpoint_config.get('resume_from_checkpoint', False)
        
    if resume_from_checkpoint:
        checkpoint_path = checkpoint_config.get('checkpoint_path')
        if not checkpoint_path:
            logger.error("❌ Checkpoint resumption enabled but no checkpoint_path specified")
            raise ValueError("checkpoint_path must be specified when resume_from_checkpoint is True")
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"❌ Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        logger.info(f"🔄 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            # Fix checkpoint keys if they have _orig_mod prefix (from torch.compile)
            model_state_dict = fix_checkpoint_keys(checkpoint['model_state_dict'])
            
            try:
                model.load_state_dict(model_state_dict)
                logger.info("✅ Model state loaded successfully")
            except RuntimeError as e:
                # If still failing, log the key mismatch for debugging
                logger.error(f"❌ Failed to load model state: {e}")
                
                # Log expected vs actual keys for debugging
                model_keys = set(model.state_dict().keys())
                checkpoint_keys = set(model_state_dict.keys())
                
                missing_keys = model_keys - checkpoint_keys
                unexpected_keys = checkpoint_keys - model_keys
                
                if missing_keys:
                    logger.error(f"Missing keys: {list(missing_keys)[:10]}...")  # Show first 10
                if unexpected_keys:
                    logger.error(f"Unexpected keys: {list(unexpected_keys)[:10]}...")  # Show first 10
                
                raise e
        else:
            logger.warning("⚠️ No model_state_dict found in checkpoint, using fresh model")
        
        # Get starting step
        if checkpoint_config.get('resume_step', True) and 'step' in checkpoint:
            start_step = checkpoint['step']
            logger.info(f"🔄 Resuming from step: {start_step}")
        else:
            logger.info("🔄 Starting from step 0")
        
        # Validate checkpoint if requested
        if checkpoint_config.get('validate_checkpoint', False):
            logger.info("🔍 Running checkpoint validation...")
            # Run a quick validation to check checkpoint quality
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                sample_output = model(sample_batch['input_ids'].to(device), 
                                    attention_mask=sample_batch['attention_mask'].to(device))
                logger.info("✅ Checkpoint validation passed")
    else:
        logger.info("🚀 Starting fresh training (no checkpoint resumption)")
    
    # Setup gradient checkpointing before moving to device
    setup_gradient_checkpointing(model, config, logger)
    
    model = model.to(device)
    
    # Compile model if enabled
    model = compile_model(model, config, logger)
    
    # Create validation model AFTER compilation to match parameter structure
    # Keep an eager shadow copy strictly for validation to avoid Inductor/Dynamo issues
    val_model = copy.deepcopy(model).to(device)
    # Wrap with DDP after compile
    if dist.is_available() and dist.is_initialized():
        local_rank = int(os.environ.get('LOCAL_RANK', '0')) if torch.cuda.is_available() else None
        model = DDP(
            model,
            device_ids=[local_rank] if local_rank is not None else None,
            output_device=local_rank if local_rank is not None else None,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        
    # Print detailed model size information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    logger.info("📊 Model Architecture Summary:")
    logger.info(f"  Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"  Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    if non_trainable_params > 0:
        logger.info(f"  Non-trainable Parameters: {non_trainable_params:,} ({non_trainable_params/1e6:.2f}M)")
    
    # Log optimization status (ModernBERT-aware)
    max_position_embeddings = config['data']['max_position_embeddings']  # Total token capacity
    
    # Show model config details
    model_size_info = config.get('model_architecture', {}).get('model_size', config.get('training', {}).get('model_size', {}))
    if model_size_info:
        logger.info("🏗️ Architecture Configuration:")
        logger.info(f"  Hidden Layers: {model_size_info.get('num_hidden_layers', 'N/A')}")
        logger.info(f"  Hidden Size: {model_size_info.get('hidden_size', 'N/A')}")
        logger.info(f"  Attention Heads: {model_size_info.get('num_attention_heads', 'N/A')}")
        logger.info(f"  Intermediate Size: {model_size_info.get('intermediate_size', 'N/A')}")
        logger.info(f"  Vocabulary Size: {vocab_size:,}")
        logger.info(f"  Max Position Embeddings: {max_position_embeddings:,}")
    
    efficiency_gain = 8192 / max_position_embeddings if max_position_embeddings > 0 else 1
    memory_reduction = (1 - (max_position_embeddings / 8192) ** 2) * 100 if max_position_embeddings < 8192 else 0
    
    logger.info("🔧 Optimization Status:")
    logger.info("📋 ModernBERT Built-in Features:")
    logger.info("  ✅ RoPE (Rotary Position Embeddings) - Native")
    logger.info("  ✅ Unpadding & Sequence Packing - Native") 
    logger.info("  ✅ Local-Global Alternating Attention - Native")
    logger.info("  ✅ Hardware-Optimized Architecture - Native")
    logger.info("⚡ Sequence Length Optimization:")
    logger.info(f"  Total Model Capacity: {max_position_embeddings} tokens (vs 8192 max)")
    logger.info(f"  Attention Speedup: ~{efficiency_gain:.1f}x faster")
    logger.info(f"  Memory Reduction: ~{memory_reduction:.0f}% less memory")
    logger.info("🔧 Additional Optimizations:")
    logger.info(f"  Flash Attention 2: {'✅' if os.environ.get('FLASH_ATTENTION_ENABLED') == 'true' else '❌'}")
    logger.info(f"  xFormers: {'✅' if os.environ.get('XFORMERS_ENABLED') == 'true' else '❌'}")
    logger.info(f"  Gradient Checkpointing: {'✅' if optimizations.get('memory', {}).get('gradient_checkpointing', False) else '❌'}")
    logger.info(f"  Mixed Precision: {'✅' if config['training'].get('mixed_precision', False) else '❌'}")
    logger.info(f"  Torch Compile: {'✅' if optimizations.get('torch_compile', {}).get('enabled', False) else '❌'}")

    # Initialize metrics calculators for both phases
    mlm_metrics_calculator = MLMMetricsCalculator(vocab_json, vocab_json['modality_offsets'])
    diffusion_metrics_calculator = DiffusionMetricsCalculator(vocab_json, vocab_json['modality_offsets'])
    logger.info("Initialized separate metrics calculators for MLM and Diffusion phases")
    
    # Global sequence tracking for phase overlap detection
    global_sequence_tracker = {
        'mlm_sequences': set(),
        'diffusion_sequences': set()
    }
    # Overall tracker (persists across phases) - using simple counters and single-rank unique counting
    # Get configurable memory limits
    unique_tracking_config = config['training'].get('unique_tracking', {})
    max_unique_sequences = unique_tracking_config.get('max_unique_sequences', 10000000)  # Default 10M
    max_unique_families = unique_tracking_config.get('max_unique_families', 10000)       # Default 10K
    
    overall_tracker = {
        'total_sequences': 0,
        'total_families': 0,
        'unique_families_hash': set(),  # Hash-based unique family tracking (single rank only)
        'unique_sequences_hash': set(),  # Hash-based unique sequence tracking (single rank only)
        'sequence_sampling_rate': 1.0,   # Track all sequences for better accuracy
        'total_sequences_sampled': 0,    # Track total sequences sampled
        'max_unique_sequences': max_unique_sequences,  # Configurable limit
        'max_unique_families': max_unique_families     # Configurable limit
    }

    # --- Two-Phase Training Setup ---
    two_phase_config = config['training'].get('two_phase_training', {})
    enable_two_phase = two_phase_config.get('enabled', False)
        
    if enable_two_phase:
        logger.info("🚀 Two-phase training enabled: MLM → Diffusion")
        
        # Import diffusion components
        from discrete_diffusion import create_diffusion_components, diffusion_collate_fn
        
        # Create diffusion components
        diffusion_config = config['training']['diffusion']
        # Create a config structure that matches what create_diffusion_components expects
        diffusion_components_config = {
            'training': {
                'discrete_diffusion': {
                    'num_timesteps': diffusion_config['num_timesteps'],
                    'loss_type': diffusion_config['loss_type'],
                    'masking_rates': diffusion_config.get('masking_rates', None),  # Legacy masking rates
                    'cumulative_masking': diffusion_config.get('cumulative_masking', None),  # New cumulative masking
                    'conditional_generation': {
                        'modality_dropout_rates': diffusion_config['modality_dropout_rates'],
                        'target_modality': 'sequence',
                        'conditioning_modalities': ['ss', 'consensus', 'go_terms']
                    },
                    'loss_weights': {
                        'sequence': diffusion_config['sequence_loss_weight']
                    }
                }
            }
        }
        noise_scheduler, data_preparator, diffusion_loss = create_diffusion_components(diffusion_components_config, vocab_json)
        logger.info("✅ Initialized diffusion components")
        
        # Phase configuration
        mlm_phase_steps = two_phase_config.get('mlm_phase_steps', 1000)
        diffusion_phase_steps = two_phase_config.get('diffusion_phase_steps', 2000)
        total_phase_steps = mlm_phase_steps + diffusion_phase_steps
        
        logger.info(f"📊 Phase Configuration:")
        logger.info(f"  MLM Phase: {mlm_phase_steps} steps")
        logger.info(f"  Diffusion Phase: {diffusion_phase_steps} steps")
        logger.info(f"  Total Steps: {total_phase_steps}")
        
        # Create diffusion data loader with custom collate function
        # Use functools.partial to avoid pickling issues with lambda
        diffusion_collate_with_args = functools.partial(
            diffusion_collate_fn, 
            config=config, 
            vocabulary=vocab_json, 
            data_preparator=data_preparator, 
            training=True
        )
        
        diffusion_train_loader = create_optimized_data_loader(
            train_loader.dataset,
            config,
            sampler=train_loader.sampler,
            collate_fn=diffusion_collate_with_args,
            training=True
        )
        # If DDP is enabled, rebuild with DistributedFamilyDiverseSampler aligned to train_loader's sampler
        if dist.is_available() and dist.is_initialized():
            # Get distributed training parameters
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            batch_size = config['data']['batch_size']
            random_seed = config.get('random_seed', 42)
            if random_seed == 'random':
                random_seed = 42  # Fallback to fixed seed for distributed training
            
            diffusion_sampler = DistributedSimpleFamilyDiverseSampler(
                diffusion_train_loader.dataset, world_size, rank, batch_size, family_lookup_path, 
                shuffle=True, seed=random_seed, drop_last=True
            )
            diffusion_train_loader = create_optimized_data_loader(
                diffusion_train_loader.dataset,
                config,
                sampler=diffusion_sampler,
                collate_fn=diffusion_collate_with_args,
                training=True
            )
        logger.info("✅ Created diffusion data loader")
    else:
        logger.info("📚 Single-phase training: MLM only")
        noise_scheduler = None
        data_preparator = None
        diffusion_loss = None
        diffusion_train_loader = None
        mlm_phase_steps = 0
        diffusion_phase_steps = 0
        # Calculate total_phase_steps based on max_steps if available, otherwise use default
        max_steps = config['training'].get('max_steps', None)
        if max_steps is not None:
            total_phase_steps = max_steps
        else:
            total_phase_steps = len(train_loader) * 10  # Default to 10 epochs worth of steps

    # Calculate steps per epoch for epoch tracking
    steps_per_epoch_mlm = len(train_loader)
    steps_per_epoch_diff = len(diffusion_train_loader) if diffusion_train_loader is not None else None
    
    # 5. Optimizer, scheduler, loss
    # Prefer fused AdamW on CUDA for better throughput (PyTorch 2.4+)
    lr_value = float(config['training']['learning_rate'])
    weight_decay_value = float(config['training'].get('weight_decay', 0.0))
    try:
        if torch.cuda.is_available():
            optimizer = optim.AdamW(model.parameters(), lr=lr_value, weight_decay=weight_decay_value, fused=True)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr_value, weight_decay=weight_decay_value)
    except TypeError:
        optimizer = optim.AdamW(model.parameters(), lr=lr_value, weight_decay=weight_decay_value)
    
    # Use max_steps from config for scheduler
    max_steps = config['training'].get('max_steps', None)
    logger.info(f"📊 Config values - max_steps: {max_steps}, mlm_phase_steps: {mlm_phase_steps}, diffusion_phase_steps: {diffusion_phase_steps}")
    
    if max_steps is not None:
        total_steps = max_steps
        logger.info(f"📊 Using max_steps from config: {total_steps}")
    elif enable_two_phase:
        total_steps = total_phase_steps
        logger.info(f"📊 Using total_phase_steps: {total_steps}")
    else:
        # Fallback: use a reasonable default if neither max_steps nor two_phase is configured
        total_steps = 10000  # Default to 10k steps
        logger.info(f"📊 Using default steps: {total_steps} (consider setting max_steps in config)")
        
    scheduler = setup_enhanced_scheduler(optimizer, config, total_steps)
    
    # Load optimizer and scheduler state from checkpoint if resuming
    if resume_from_checkpoint and 'checkpoint' in locals():
        if checkpoint_config.get('resume_optimizer', True) and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("✅ Optimizer state loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load optimizer state: {e}. Continuing with fresh optimizer state.")
        
        if checkpoint_config.get('resume_scheduler', True) and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("✅ Scheduler state loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load scheduler state: {e}. Continuing with fresh scheduler state.")
    
    # Prefer PyTorch SDPA flash only if flash-attn is unavailable
    if os.environ.get('FLASH_ATTENTION_ENABLED', 'false') != 'true':
        try:
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
        except Exception:
            pass

    # Enhanced mixed precision setup
    use_mixed_precision = config['training'].get('mixed_precision', False)
    precision_type = os.environ.get('MIXED_PRECISION_TYPE', 'bf16')
    
    if use_mixed_precision and torch.cuda.is_available():
        if precision_type == 'bf16' and torch.cuda.is_bf16_supported():
            scaler = None  # bfloat16 doesn't need gradient scaling
            logger.info("✅ Using bfloat16 mixed precision (no gradient scaling needed)")
        else:
            scaler = torch.cuda.amp.GradScaler()
            precision_type = 'fp16'  # fallback to fp16 with scaling
            logger.info("✅ Using float16 mixed precision with gradient scaling")
    else:
        scaler = None
        precision_type = 'fp32'

    # --- Modality loss weights ---
    modality_loss_weights = config['training'].get('modality_loss_weights', None)
    if modality_loss_weights is None:
        raise ValueError("modality_loss_weights must be specified in the config under training: (sequence, secondary_structure, consensus, go_terms)")
    # Ensure weights sum to 1.0
    weight_sum = sum(modality_loss_weights.values())
    if not np.isclose(weight_sum, 1.0, atol=1e-5):
        raise ValueError(f"modality_loss_weights must sum to 1.0, got {weight_sum}")
    # Map modality index to name
    modality_id_to_name = {0: 'sequence', 1: 'secondary_structure', 2: 'consensus', 3: 'go_terms'}
    modality_names = ['sequence', 'secondary_structure', 'consensus', 'go_terms']

    # --- Consensus case penalty ---
    consensus_case_penalty_factor = config['training'].get('consensus_case_penalty_factor', 0.5)
    # Precompute consensus letter code maps (lower/upper) on device for vectorized penalty.
    # Only apply case penalty for single alphabetic characters; mark others as -1 to skip penalty.
    consensus_vocab = vocab_json['modality_vocabs']['consensus']
    consensus_offset = vocab_json['modality_offsets']['consensus']
    lower_list = []
    upper_list = []
    for token_str in consensus_vocab:
        if isinstance(token_str, str) and len(token_str) == 1 and token_str.isalpha():
            lower_list.append(ord(token_str.lower()))
            upper_list.append(ord(token_str.upper()))
        else:
            lower_list.append(-1)
            upper_list.append(-1)
    consensus_lower_codes = torch.tensor(lower_list, device=device, dtype=torch.int16)
    consensus_upper_codes = torch.tensor(upper_list, device=device, dtype=torch.int16)
    

    # 6. Two-Phase Training Loop
    model.train()
    apply_vocab_constraints = config['training'].get('apply_vocab_constraints', False)
    validation_freq_steps = config['training'].get('validation_freq_steps', 100)
    validation_num_samples = config['training'].get('validation_num_samples', 128)
    print_freq = config['training'].get('print_freq', 100)
    max_steps = config['training'].get('max_steps', None)
    model_save_freq_steps = config['training'].get('model_save_freq_steps', 1000)
    step = start_step  # Use start_step from checkpoint resumption
    running_loss = 0.0
    
    # Timing variables for progress tracking
    import time
    start_time = time.time()
    last_progress_time = start_time
    
    # Sequences per second tracking
    sequences_per_second_start_time = start_time
    total_elapsed_time = 0.0

    # Initialize loss function once (avoid per-iter re-allocation)
    padding_id = config['data']['padding_label_id']
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=padding_id)
    
    # Phase tracking
    current_phase = "MLM"
    phase_step = 0
    
    # Initialize iterators to preserve state across phases
    mlm_iterator = None
    diffusion_iterator = None
    diffusion_seeded = False
    # Track sampler epochs for DDP deterministic shuffling
    mlm_epoch = 0
    diffusion_epoch = 0
    
    logger.info(f"🎯 Starting training with {'two-phase' if enable_two_phase else 'single-phase'} training")
    if resume_from_checkpoint:
        logger.info(f"🔄 RESUMING from step {start_step} (total steps: {total_steps})")
    else:
        logger.info(f"🚀 FRESH START from step 0 (total steps: {total_steps})")
    
    # Log masking rates for clarity
    mlm_masking_rate = config['training']['MLM_masking']['rate']
    logger.info(f"🎭 MLM Masking Rate: {mlm_masking_rate * 100:.1f}%")
    if enable_two_phase:
        diffusion_config = config['training']['diffusion']
        initial_rate = diffusion_config['cumulative_masking']['initial_rate']
        target_percentage = diffusion_config['cumulative_masking']['target_percentage']
        logger.info(f"🎭 Diffusion Masking: {initial_rate * 100:.1f}% → {target_percentage * 100:.1f}% (progressive)")
    
    # Main training loop
    while step < total_steps:
        # Determine current phase
        if enable_two_phase:
            if step < mlm_phase_steps:
                current_phase = "MLM"
                phase_step = step
                current_loader = train_loader
                phase_desc = f"MLM Phase ({step}/{mlm_phase_steps})"
                
                # Initialize or use existing MLM iterator
                if mlm_iterator is None:
                    mlm_iterator = iter(train_loader)
                current_iterator = mlm_iterator
            else:
                current_phase = "Diffusion"
                phase_step = step - mlm_phase_steps
                current_loader = diffusion_train_loader
                phase_desc = f"Diffusion Phase ({phase_step}/{diffusion_phase_steps})"
                
                # Initialize or use existing diffusion iterator
                if diffusion_iterator is None:
                    # Seed diffusion sampler differently to avoid repeating MLM order
                    try:
                        if hasattr(diffusion_train_loader, 'sampler') and hasattr(diffusion_train_loader.sampler, 'set_epoch') and not diffusion_seeded:
                            # Use a different epoch seed than MLM; base it on random_seed offset
                            diffusion_epoch = mlm_epoch + 100003
                            diffusion_train_loader.sampler.set_epoch(diffusion_epoch)
                            diffusion_seeded = True
                    except Exception:
                        pass
                    diffusion_iterator = iter(diffusion_train_loader)
                current_iterator = diffusion_iterator
        else:
            current_phase = "MLM"
            current_loader = train_loader
            phase_desc = f"MLM Training ({step}/{total_phase_steps})"
            
            # Initialize or use existing MLM iterator
            if mlm_iterator is None:
                mlm_iterator = iter(train_loader)
            current_iterator = mlm_iterator
        
        # Phase transition logging
        if step % print_freq == 0 and is_main_process():
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 {phase_desc}")
            logger.info(f"📊 Global Step: {step}/{total_steps}")
            logger.info(f"{'='*60}")
        
        # Get batch from appropriate loader using preserved iterator
        try:
            batch = next(current_iterator)
        except StopIteration:
            # Restart iterator if exhausted and advance distributed sampler epoch
            if current_phase == "MLM":
                if dist.is_available() and dist.is_initialized() and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                    mlm_epoch += 1
                    train_loader.sampler.set_epoch(mlm_epoch)
                mlm_iterator = iter(train_loader)
                current_iterator = mlm_iterator
            else:
                if diffusion_train_loader is not None and dist.is_available() and dist.is_initialized() and hasattr(diffusion_train_loader, 'sampler') and hasattr(diffusion_train_loader.sampler, 'set_epoch'):
                    diffusion_epoch += 1
                    diffusion_train_loader.sampler.set_epoch(diffusion_epoch)
                diffusion_iterator = iter(diffusion_train_loader)
                current_iterator = diffusion_iterator
            batch = next(current_iterator)
        
        if batch['input_ids'].shape[0] == 0:
            logger.info(f"Skipping empty batch (batch length: 0).")
            continue
        
        # Get batch size for tracking
        batch_size = batch['input_ids'].shape[0]
            
        # Extract sequences for tracking (MLM phase) and update overall tracker
        extract_sequences_from_batch(batch, batch['modality_type_ids'], 'MLM', global_sequence_tracker, config['data']['padding_label_id'], vocab_json)
        update_overall_tracker(batch, batch['modality_type_ids'], overall_tracker, config['data']['padding_label_id'], vocab_json)
            
        # Only log sample output every print_freq steps (rank 0)
        if step % print_freq == 0 and is_main_process():
            logger.info(f"\n--- Batch {step} ---")
            logger.info(f"input_ids shape: {batch['input_ids'].shape}")
            logger.info(f"attention_mask shape: {batch['attention_mask'].shape}")
            
            # Handle different batch structures for MLM vs Diffusion
            if 'labels' in batch:
                logger.info(f"labels shape: {batch['labels'].shape}")
            else:
                logger.info("labels not in batch (diffusion mode)")
            
            # Phase-specific logging
            if current_phase == "MLM":
                # MLM phase logging (use eager val_model; disable Dynamo/compile for logging)
                masked_input_ids = to_list_cpu(batch['input_ids'][0])
                labels = to_list_cpu(batch['labels'][0])
                original_input_ids = to_list_cpu(batch['original_input_ids'][0])
                # Unmasked: decode original_input_ids
                unmasked_decoded = decode_tokens(original_input_ids, vocab_json)
                # Masked: show * for masked positions, else show decoded
                masked_decoded = visualize_masked(unmasked_decoded, masked_input_ids, mask_token_id, vocab_json)
                # Model predictions (eager)
                with torch.no_grad():
                    def _log_forward_mlm(inp_ids, attn_mask):
                        return val_model(inp_ids, attention_mask=attn_mask)
                    try:
                        import torch._dynamo as _dyn
                        _log_forward_mlm = _dyn.disable(_log_forward_mlm)
                    except Exception:
                        pass
                    model_outputs = _log_forward_mlm(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                    model_logits = model_outputs['logits'] if isinstance(model_outputs, dict) else model_outputs.logits
                    masked_logits = apply_modality_vocab_mask(model_logits, batch['modality_type_ids'].to(device), valid_token_indices_by_modality, apply_vocab_constraints)
                    pred_ids = to_list_cpu(masked_logits[0].argmax(dim=-1))
                pred_decoded = decode_tokens(pred_ids, vocab_json)
                # Show which positions are masked (labels != -100)
                mask_positions = [j for j, l in enumerate(labels) if l != config['data']['padding_label_id']]
                # --- Modality-wise logging ---
                slices = get_modality_slices(original_input_ids, vocab_json)
                for mod, (start, end) in slices.items():
                    logger.info(f"[{mod.upper()}]")
                    logger.info(f"Unmasked:   {''.join(unmasked_decoded[start:start+50])}")
                    logger.info(f"Masked:     {''.join(masked_decoded[start:start+50])}")
                    # Predicted line shows model prediction at masked positions, original at others
                    pred_vis = []
                    for idx in range(start, min(end, start+50)):
                        if idx in mask_positions:
                            pred_vis.append(pred_decoded[idx])
                        else:
                            pred_vis.append(unmasked_decoded[idx])
                    logger.info(f"Predicted:  {''.join(pred_vis)}")
                logger.info(f"Masked positions: {mask_positions}")
            else:
                # Diffusion phase logging - Enhanced to match MLM format exactly
                # Get the first sample for detailed logging
                sample_idx = 0  # Always show sample 0 for consistency
                
                logger.info(f"🔄 Diffusion Phase - Step {phase_step}")
                logger.info(f"📊 Batch shape: {batch['input_ids'].shape}")
                logger.info(f"📋 Showing Sample {sample_idx} of {batch['input_ids'].shape[0]}")
                
                # Get the first sample for detailed logging
                original_tokens = to_list_cpu(batch.get('original_input_ids', batch['input_ids'])[sample_idx])
                
                # Apply diffusion noise to get actual noisy tokens for logging
                # Sample a random timestep for demonstration
                demo_timestep = torch.randint(0, noise_scheduler.num_timesteps, (1,), device=device)
                
                # Get sequence mask for the first sample
                conditioning_info = batch.get('conditioning_info', {})
                sequence_mask = conditioning_info.get('sequence_mask', torch.ones_like(batch['input_ids'][:1], dtype=torch.bool)).to(device)
                
                # Add noise to get noisy tokens for logging
                # Use the modality-dropped tokens (batch['input_ids']) instead of original_tokens
                modality_dropped_tokens_tensor = batch['input_ids'][:1].to(device)
                noisy_tokens_tensor, noise_info = noise_scheduler.add_noise(
                    modality_dropped_tokens_tensor, 
                    demo_timestep, 
                    sequence_mask
                )
                noisy_tokens = to_list_cpu(noisy_tokens_tensor[0])
                
                # Decode all versions
                original_decoded = decode_tokens(original_tokens, vocab_json)
                
                # Get modality-dropped tokens (before noise) to show dropped modalities with X
                modality_dropped_tokens = to_list_cpu(batch['input_ids'][sample_idx])
                modality_dropped_decoded = visualize_masked(original_decoded, modality_dropped_tokens, mask_token_id, vocab_json)
                
                # Apply mask visualization to noisy tokens
                noisy_decoded = visualize_masked(original_decoded, noisy_tokens, mask_token_id, vocab_json)
                
                # Get model predictions (use noisy tokens; eager val_model)
                with torch.no_grad():
                    def _log_forward_diff(noisy, attn):
                        return val_model(noisy, attention_mask=attn)
                    try:
                        import torch._dynamo as _dyn
                        _log_forward_diff = _dyn.disable(_log_forward_diff)
                    except Exception:
                        pass
                    model_outputs = _log_forward_diff(noisy_tokens_tensor, batch['attention_mask'][:1].to(device))
                    model_logits = model_outputs['logits'] if isinstance(model_outputs, dict) else model_outputs.logits
                    masked_logits = apply_modality_vocab_mask(model_logits, batch['modality_type_ids'][:1].to(device), valid_token_indices_by_modality, apply_vocab_constraints)
                    pred_ids = to_list_cpu(masked_logits[0].argmax(dim=-1))
                pred_decoded = decode_tokens(pred_ids, vocab_json)
                
                # Get modality slices
                slices = get_modality_slices(original_tokens, vocab_json)
                
                # Get noise info for logging
                timestep = int(demo_timestep)
                corruption_rate = noise_info.get('corruption_rate', 'unknown') if noise_info else 'unknown'
                
                logger.info(f"🎲 Timestep: {timestep}, Corruption Rate: {corruption_rate:.3f} (progressive masking)")
                logger.info(f"🎯 Note: Only sequence modality is being generated; other modalities are conditioning")
                
                # --- Modality-wise logging (EXACTLY like MLM format) ---
                for mod, (start, end) in slices.items():
                    logger.info(f"[{mod.upper()}]")
                    # Show 50 characters like MLM format
                    display_length = min(50, end - start)
                    logger.info(f"Original:   {''.join(original_decoded[start:start+display_length])}")
                    logger.info(f"Pre:        {''.join(modality_dropped_decoded[start:start+display_length])}")
                    logger.info(f"Noisy:      {''.join(noisy_decoded[start:start+display_length])}")
                    
                    # For diffusion phase, show predictions for ALL modalities (like MLM)
                    # This helps debug what the model is predicting at each position
                    logger.info(f"Predicted:  {''.join(pred_decoded[start:start+display_length])}")
                
                # Show which positions were corrupted (sequence only)
                sequence_slice = slices.get('sequence', (0, 0))
                if sequence_slice[1] > sequence_slice[0]:
                    corrupted_positions = []
                    for i in range(sequence_slice[0], min(sequence_slice[1], len(original_tokens))):
                        if original_tokens[i] != noisy_tokens[i]:
                            corrupted_positions.append(i - sequence_slice[0])  # Relative to sequence start
                    logger.info(f"Corrupted sequence positions: {corrupted_positions[:20]}{'...' if len(corrupted_positions) > 20 else ''}")
                
                # Show modality dropout info (simplified)
                if conditioning_info:
                    dropped_modalities = conditioning_info.get('dropped_modalities', [])
                    if dropped_modalities and any(dropped_modalities):
                        # Count how many samples had dropout
                        samples_with_dropout = sum(1 for dropped_mods in dropped_modalities if dropped_mods)
                        total_samples = len(dropped_modalities)
                        logger.info(f"Modality dropout: {samples_with_dropout}/{total_samples} samples had modalities dropped")
                    else:
                        logger.info("No modalities dropped")
            
        # Progress bar log with timing metrics; all ranks must participate in the allreduce
        if step % print_freq == 0:
            avg_loss = running_loss / print_freq if print_freq > 0 else running_loss
            current_time = time.time()
            time_since_last_progress = current_time - last_progress_time
            time_per_iter = time_since_last_progress / print_freq if print_freq > 0 else 0
            steps_remaining = total_steps - step
            estimated_time_remaining = steps_remaining * time_per_iter if time_per_iter > 0 else 0
            sequences_since_last = print_freq * config['data']['batch_size']
            # All ranks call allreduce
            total_sequences_global = distributed_sum(sequences_since_last, device)
            time_den = time_since_last_progress if time_since_last_progress > 0 else 1e-8
            sequences_per_second_global = total_sequences_global / time_den
            # Update last progress time on all ranks
            last_progress_time = current_time
            if is_main_process():
                def format_time(seconds):
                    if seconds < 60:
                        return f"{seconds:.1f}s"
                    elif seconds < 3600:
                        minutes = seconds / 60
                        return f"{minutes:.1f}m"
                    else:
                        hours = seconds / 3600
                        return f"{hours:.1f}h"
                logger.info(f"[PROGRESS] {phase_desc} | Global Step: {step}/{total_steps} | Avg Loss: {avg_loss:.4f} | Time/Iter: {format_time(time_per_iter)} | ETA: {format_time(estimated_time_remaining)} | Seq/sec: {sequences_per_second_global:.1f}")

            # Calculate and log epoch progress (continuous across phases)
            try:
                if current_phase == "MLM":
                    # MLM phase: calculate fractional epoch within MLM phase
                    mlm_frac = (phase_step % steps_per_epoch_mlm) / steps_per_epoch_mlm
                    epochs_progress = mlm_epoch + mlm_frac
                else:
                    # Diffusion phase: continue counting from where MLM left off
                    denom = steps_per_epoch_diff if steps_per_epoch_diff is not None else 1
                    diff_frac = (phase_step % denom) / denom
                    epochs_progress = mlm_epoch + diffusion_epoch + diff_frac
                
                if is_main_process():
                    wandb.log({
                        'overall/epochs': epochs_progress
                    }, step=step)
            except Exception:
                pass
            
            # Log training metrics every print_freq steps
            if current_phase == "MLM":
                # MLM phase: use standard metrics
                metrics_results = mlm_metrics_calculator.calculate_metrics(phase='training')
                mlm_metrics_calculator.log_metrics('training', step, metrics_results)
                # Unified overall metrics (global across ranks) from persistent overall_tracker
                overall_total_sequences = int(distributed_sum(overall_tracker['total_sequences'], device))
                # Get single-rank unique counts (only main process tracks unique items)
                overall_unique_families = len(overall_tracker['unique_families_hash']) if is_main_process() else 0
                overall_unique_sequences = len(overall_tracker['unique_sequences_hash']) if is_main_process() else 0
                
                # Get unique sequences count (now tracking all sequences up to 1M limit)
                estimated_total_unique_sequences = overall_unique_sequences
                
                if is_main_process():
                    # Overwrite overall tab with unified metrics only
                    wandb.log({
                        'overall/total_sequences': overall_total_sequences,
                        'overall/total_unique_families_singleRank': overall_unique_families,
                        'overall/total_unique_sequences_singleRank': estimated_total_unique_sequences
                    }, step=step)
                
                # --- WANDB LOG ---
                if 'overall' in metrics_results:
                    wandb.log({'train_mlm/overall_accuracy': metrics_results['overall']['accuracy']}, step=step)
                    wandb.log({'train_mlm/overall_loss': metrics_results['overall']['loss']}, step=step)
                    wandb.log({'train_mlm/overall_perplexity': metrics_results['overall']['perplexity']}, step=step)
                
                if 'modalities' in metrics_results:
                    for mod_name, metrics in metrics_results['modalities'].items():
                        wandb.log({f'train_mlm/{mod_name}_accuracy': metrics['accuracy']}, step=step)
                        wandb.log({f'train_mlm/{mod_name}_loss': metrics['loss']}, step=step)
                        wandb.log({f'train_mlm/{mod_name}_perplexity': metrics['perplexity']}, step=step)
                
                # Training stats
                if 'training_stats' in metrics_results:
                    stats = metrics_results['training_stats']
                    wandb.log({
                        'train_mlm/total_sequences': stats['total_sequences'],
                        'train_mlm/unique_sequences': stats['unique_sequences'],
                        'train_mlm/total_families': stats['total_families'],
                        'train_mlm/unique_families': stats['unique_families'],
                        'train_mlm/total_dropped_tokens': stats['total_dropped_tokens'],
                        'train_mlm/sequence_diversity': stats['unique_sequences'] / max(1, stats['total_sequences']),
                        'train_mlm/family_diversity': stats['unique_families'] / max(1, stats['total_families']),
                        'train_mlm/percent_sequence_masked': config['training']['MLM_masking']['rate'] * 100.0,  # Use actual MLM masking rate
                        'train_mlm/min_sequence_length': stats['min_sequence_length'],
                        'train_mlm/max_sequence_length': stats['max_sequence_length'],
                        'train_mlm/avg_sequence_length': stats['avg_sequence_length']
                    }, step=step)
                    
                    # Log top token prediction bar charts for each modality
                    if 'top_token_predictions' in metrics_results:
                        for modality_name in ['sequence', 'secondary_structure', 'consensus', 'go_terms']:
                            chart = create_token_prediction_bar_chart(
                                metrics_results['top_token_predictions'], 
                                vocab_json, 
                                modality_name, 
                                'train_mlm'
                            )
                            if chart:
                                wandb.log({f'train_mlm/top_tokens_{modality_name}_bar': chart}, step=step)
                    
                    # Log modality dropout combinations as bar chart (MLM training)
                    if 'modality_dropout_combinations' in metrics_results:
                        dropout_combos = metrics_results['modality_dropout_combinations']
                        if dropout_combos:
                            # Create bar chart data
                            combo_names = list(dropout_combos.keys())
                            combo_counts = list(dropout_combos.values())
                            
                            # Log as wandb bar chart
                            wandb.log({
                                "train_mlm/dropout_combinations_bar": wandb.plot.bar(
                                    wandb.Table(data=[[name, count] for name, count in zip(combo_names, combo_counts)], 
                                                columns=["Modality_Combination", "Count"]),
                                    "Modality_Combination", "Count",
                                    title="MLM Training Modality Dropout Combinations"
                                )
                            }, step=step)
                    
                    # Mirror sequences per second into overall for unified monitoring
                    if is_main_process():
                        wandb.log({
                            'overall/sequences_per_second': sequences_per_second_global
                        }, step=step)


            else:
                # Diffusion phase: log diffusion-specific metrics
                # Note: These will be logged in the diffusion training loop after metrics are calculated
                pass
            running_loss = 0.0
        
        optimizer.zero_grad(set_to_none=True)
        # Use the correct dtype for mixed precision
        autocast_dtype = torch.bfloat16 if precision_type == 'bf16' else torch.float16 if precision_type == 'fp16' else torch.float32
        with record_function("forward_backward_step"), torch.amp.autocast('cuda', enabled=use_mixed_precision, dtype=autocast_dtype):
            if current_phase == "MLM":
                # MLM phase training (existing logic)
                modality_type_ids = batch['modality_type_ids'].to(device, non_blocking=True)
                with record_function("model_forward_mlm"):
                    model_logits = model(
                        batch['input_ids'].to(device, non_blocking=True),
                        attention_mask=batch['attention_mask'].to(device, non_blocking=True)
                    )['logits']
                masked_logits = apply_modality_vocab_mask(model_logits, modality_type_ids, valid_token_indices_by_modality, apply_vocab_constraints)
                labels = batch['labels'].to(device)
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=config['data']['padding_label_id'])
                
                # CRITICAL: Exclude [DROPPED] token positions from loss calculation
                dropped_token_id = vocab_json['special_tokens']['[DROPPED]']
                # Flatten for loss: (batch*seq, vocab)
                active_loss = (labels.view(-1) != config['data']['padding_label_id']) & (batch['input_ids'].to(device).view(-1) != dropped_token_id)
                active_logits = masked_logits.view(-1, vocab_size)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                # Compute per-modality losses
                modality_losses = {}
                weighted_loss = 0.0
                for mod_id, mod_name in modality_id_to_name.items():
                    mod_mask = (modality_type_ids == mod_id) & (labels != config['data']['padding_label_id']) & (batch['input_ids'].to(device) != dropped_token_id)
                    if mod_mask.sum() > 0:
                        mod_logits = masked_logits[mod_mask]
                        mod_labels = labels[mod_mask]
                        if mod_name == 'consensus':
                            with record_function("consensus_case_loss"):
                                ce_loss = torch.nn.functional.cross_entropy(mod_logits, mod_labels, reduction='none')
                            pred_ids = mod_logits.argmax(dim=-1)
                            # Build a vectorized penalty mask using tensors
                            # Map token ids -> ascii code lower/upper via tensors
                            # Build lookup tensors once per step for consensus vocab range
                            vocab_len = len(consensus_vocab)
                            # Convert ids -> codes; out-of-range gets -1
                            def ids_to_codes(ids):
                                rel = ids - consensus_offset
                                in_range = (rel >= 0) & (rel < vocab_len)
                                # Default -1 where not in consensus range
                                codes_lower = torch.full_like(rel, fill_value=-1, dtype=torch.int16)
                                codes_upper = torch.full_like(rel, fill_value=-1, dtype=torch.int16)
                                rel_clamped = rel.clamp(min=0, max=vocab_len-1)
                                codes_lower[in_range] = consensus_lower_codes[rel_clamped[in_range]]
                                codes_upper[in_range] = consensus_upper_codes[rel_clamped[in_range]]
                                return codes_lower, codes_upper, in_range
                            t_lower, t_upper, t_in = ids_to_codes(mod_labels)
                            p_lower, p_upper, p_in = ids_to_codes(pred_ids)
                            # same letter ignoring case but different case
                            same_letter_ignoring_case = (t_lower == p_lower) & (t_lower != -1)
                            different_case = (t_upper != p_upper)
                            penalty_mask = same_letter_ignoring_case & different_case
                            penalties = torch.where(penalty_mask, torch.as_tensor(consensus_case_penalty_factor, device=ce_loss.device, dtype=ce_loss.dtype), torch.as_tensor(1.0, device=ce_loss.device, dtype=ce_loss.dtype))
                            ce_loss = ce_loss * penalties
                            mod_loss = to_float(ce_loss.mean())
                            weighted_loss += modality_loss_weights[mod_name] * ce_loss.mean()
                        else:
                            mod_loss = to_float(loss_fct(mod_logits, mod_labels))
                            weighted_loss += modality_loss_weights[mod_name] * loss_fct(mod_logits, mod_labels)
                        modality_losses[mod_name] = mod_loss
                    else:
                        modality_losses[mod_name] = 0.0
                # Use weighted loss for backprop
                loss = weighted_loss
                
                # Track sequences for global phase overlap detection
                if step % print_freq == 0:
                    extract_sequences_from_batch(batch, modality_type_ids, 'MLM', global_sequence_tracker, config['data']['padding_label_id'], vocab_json)
                
                # Update metrics for training phase
                mlm_metrics_calculator.update_batch_metrics(
                        predictions=masked_logits,
                        targets=labels,
                        modality_type_ids=modality_type_ids,
                        phase='training',
                        modality_losses=modality_losses,
                    overall_loss=to_float(loss),
                        family_ids=batch.get('family_ids', None),
                        input_ids=batch['input_ids'].to(device),
                        mlm_dropout_info=batch.get('mlm_dropout_info', None),
                        original_input_ids=batch.get('original_input_ids', batch['input_ids']).to(device)
                )
                
            else:
                # Diffusion phase training
                # Get conditioning info from batch
                conditioning_info = batch.get('conditioning_info', {})
                sequence_mask = conditioning_info.get('sequence_mask', torch.ones_like(batch['input_ids'], dtype=torch.bool)).to(device)
                conditioning_mask = conditioning_info.get('conditioning_mask', torch.zeros_like(batch['input_ids'], dtype=torch.bool)).to(device)
                
                # Sample random timestep
                timesteps = torch.randint(0, noise_scheduler.num_timesteps, (batch['input_ids'].shape[0],), device=device)
                
                # Add noise to modality-dropped tokens (preserve dropped modalities)
                modality_dropped_tokens = batch['input_ids'].to(device)  # These already have dropped modalities applied
                original_tokens = batch.get('original_input_ids', batch['input_ids']).to(device)
                
                # Quick check for debugging
                if step % print_freq == 0:
                    # Check if modality dropout is working
                    dropped_token_id = vocab_json['special_tokens']['[DROPPED]']
                    dropped_count = int((modality_dropped_tokens == dropped_token_id).sum())
                    logger.info(f"🔍 Modality dropout check: {int(dropped_count)} dropped tokens")
                    
                    # Check sequence mask
                    logger.info(f"🔍 Sequence mask: {int(sequence_mask.sum())} actual RNA positions")
                
                with record_function("add_noise"):
                    noisy_tokens, noise_info = noise_scheduler.add_noise(
                        modality_dropped_tokens,  # Use modality-dropped tokens instead of original
                        timesteps, 
                        sequence_mask
                    )
                
                # Quick noise addition check
                if step % print_freq == 0:
                    mask_token_id = vocab_json['special_tokens']['[MASK]']
                    masks_after = int((noisy_tokens == mask_token_id).sum())
                    logger.info(f"🔍 Noise addition: {masks_after} mask tokens added")
                
                # Forward pass with noisy tokens
                with record_function("model_forward_diffusion"):
                    model_logits = model(noisy_tokens, attention_mask=batch['attention_mask'].to(device))['logits']
                
                # CRITICAL FIX: Apply vocabulary masking to prevent invalid token predictions
                masked_logits = apply_modality_vocab_mask(model_logits, batch['modality_type_ids'].to(device), valid_token_indices_by_modality, apply_vocab_constraints)
                
                # Quick model prediction check (ONLY on corrupted positions)
                if step % print_freq == 0:
                    # Only evaluate on corrupted sequence positions (same as proper metrics)
                    noise_mask = noise_info.get('noise_mask', torch.zeros_like(original_tokens, dtype=torch.bool))
                    eval_mask = sequence_mask & noise_mask  # Only corrupted sequence positions
                    
                    if eval_mask.sum() > 0:
                        seq_predictions = masked_logits[eval_mask]  # Use masked logits
                        seq_targets = original_tokens[eval_mask]
                        pred_ids = seq_predictions.argmax(dim=-1)
                        correct = (pred_ids == seq_targets).sum()
                        total = seq_targets.numel()
                        accuracy = float(correct) / float(total) if total > 0 else 0.0
                        logger.info(f"🔍 Model accuracy: {accuracy:.4f} ({int(correct)}/{int(total)}) [CORRUPTED ONLY]")
                    else:
                        logger.info(f"🔍 Model accuracy: N/A (no corrupted positions)")
                
                # Compute diffusion loss
                loss_dict = diffusion_loss(
                    predictions=masked_logits,  # Use masked logits
                    targets=original_tokens,  # Use original tokens as targets
                    timesteps=timesteps, 
                    sequence_mask=sequence_mask,
                    conditioning_mask=conditioning_mask,
                    noise_info=noise_info
                )
                loss = loss_dict['total_loss']
                
                # Quick loss check
                if step % print_freq == 0:
                    logger.info(f"🔍 Loss: {to_float(loss):.4f}")
                
                # Track sequences for global phase overlap detection
                extract_sequences_from_batch(batch, batch['modality_type_ids'], 'Diffusion', global_sequence_tracker, vocab_json=vocab_json)
                update_overall_tracker(batch, batch['modality_type_ids'], overall_tracker, config['data']['padding_label_id'], vocab_json)
                
                # Update diffusion metrics for training
                diffusion_metrics_calculator.update_batch_metrics(
                        predictions=masked_logits,  # Use masked logits
                        targets=original_tokens,
                        timesteps=timesteps,
                        sequence_mask=sequence_mask,
                        noise_info=noise_info,
                        conditioning_info=conditioning_info,
                    loss=to_float(loss),
                        phase='training',
                        family_ids=batch.get('family_ids', None)
                )
                
                # Store loss components for logging (will be shown in main logging section)
                if step % print_freq == 0:
                    logger.info(f"🔄 Diffusion Loss: {to_float(loss):.4f}")
                    
                    # Calculate and log diffusion metrics
                    diffusion_metrics_results = diffusion_metrics_calculator.calculate_metrics(phase='training')
                    diffusion_metrics_calculator.log_metrics('training', step, diffusion_metrics_results)
                    
                    # WandB logging for diffusion metrics
                    if 'generation' in diffusion_metrics_results:
                        gen_metrics = diffusion_metrics_results['generation']
                        wandb.log({
                            'train_diff/generation_accuracy': gen_metrics['accuracy'],
                            'train_diff/generation_loss': gen_metrics['loss'],
                            'train_diff/generation_perplexity': gen_metrics['perplexity']
                        }, step=step)
                    
                    if 'conditioning' in diffusion_metrics_results:
                        for cond_type, metrics in diffusion_metrics_results['conditioning'].items():
                            wandb.log({
                                f'train_diff/conditioning_{cond_type}_accuracy': metrics['accuracy'],
                                f'train_diff/conditioning_{cond_type}_loss': metrics['loss']
                            }, step=step)
                    
                    # Training stats and modality dropout combinations
                    if 'training_stats' in diffusion_metrics_results:
                        stats = diffusion_metrics_results['training_stats']
                        wandb.log({
                            'train_diff/family_diversity': stats['unique_families'] / max(1, stats['total_families']),
                            'train_diff/avg_corruption_rate': stats['avg_corruption_rate'],
                            'train_diff/min_corruption_rate': stats['min_corruption_rate'],
                            'train_diff/max_corruption_rate': stats['max_corruption_rate'],
                            'train_diff/percent_sequence_masked': stats['avg_corruption_rate'] * 100.0,
                            'train_diff/min_sequence_length': stats['min_sequence_length'],
                            'train_diff/max_sequence_length': stats['max_sequence_length'],
                            'train_diff/avg_sequence_length': stats['avg_sequence_length']
                        }, step=step)
                        
                        # Log top token prediction bar chart for sequence modality (diffusion only generates sequence)
                        if 'top_token_predictions' in diffusion_metrics_results:
                            chart = create_token_prediction_bar_chart(
                                diffusion_metrics_results['top_token_predictions'], 
                                vocab_json, 
                                'sequence', 
                                'train_diff'
                            )
                            if chart:
                                wandb.log({f'train_diff/top_tokens_sequence_bar': chart}, step=step)
                        
                        # Log modality dropout combinations as bar chart
                        if 'modality_dropout_combinations' in diffusion_metrics_results:
                            dropout_combos = diffusion_metrics_results['modality_dropout_combinations']
                            if dropout_combos:
                                # Create bar chart data
                                combo_names = list(dropout_combos.keys())
                                combo_counts = list(dropout_combos.values())
                                
                                # Log as wandb bar chart
                                wandb.log({
                                    "train_diff/dropout_combinations_bar": wandb.plot.bar(
                                        wandb.Table(data=[[name, count] for name, count in zip(combo_names, combo_counts)], 
                                                    columns=["Modality_Combination", "Count"]),
                                        "Modality_Combination", "Count",
                                        title="Modality Dropout Combinations"
                                    )
                                }, step=step)
                        
                        # Mirror sequences per second into overall for unified monitoring
                        if is_main_process():
                            wandb.log({
                                'overall/sequences_per_second': sequences_per_second_global
                            }, step=step)


                        # Unified overall metrics (global) from persistent overall_tracker
                        overall_total_sequences = int(distributed_sum(overall_tracker['total_sequences'], device))
                        # Get single-rank unique counts (only main process tracks unique items)
                        overall_unique_families = len(overall_tracker['unique_families_hash']) if is_main_process() else 0
                        overall_unique_sequences = len(overall_tracker['unique_sequences_hash']) if is_main_process() else 0
                        
                        # Get unique sequences count (now tracking all sequences up to 1M limit)
                        estimated_total_unique_sequences = overall_unique_sequences
                        
                        wandb.log({
                            'overall/total_sequences': overall_total_sequences,
                            'overall/total_unique_families_singleRank': overall_unique_families,
                            'overall/total_unique_sequences_singleRank': estimated_total_unique_sequences
                        }, step=step)
                        
                        # Calculate and log epoch progress (continuous across phases)
                        try:
                            if current_phase == "MLM":
                                # MLM phase: calculate fractional epoch within MLM phase
                                mlm_frac = (phase_step % steps_per_epoch_mlm) / steps_per_epoch_mlm
                                epochs_progress = mlm_epoch + mlm_frac
                            else:
                                # Diffusion phase: continue counting from where MLM left off
                                denom = steps_per_epoch_diff if steps_per_epoch_diff is not None else 1
                                diff_frac = (phase_step % denom) / denom
                                epochs_progress = mlm_epoch + diffusion_epoch + diff_frac
                            
                            if is_main_process():
                                wandb.log({
                                    'overall/epochs': epochs_progress
                                }, step=step)
                        except Exception:
                            pass
        
        running_loss += to_float(loss)
        
        if torch.isnan(loss):
            logger.error("[ERROR] Loss is NaN!")
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Increment step counter
        step += 1
        
        # Periodic CUDA memory cleanup to prevent accumulation
        if step % 50 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Validation
        if step % validation_freq_steps == 0:
            if is_main_process():
                logger.info(f"\n[VALIDATION] Running validation at step {step}...")
            
            # Log phase overlap statistics
            mlm_seq_count = len(global_sequence_tracker['mlm_sequences'])
            diffusion_seq_count = len(global_sequence_tracker['diffusion_sequences'])
            if mlm_seq_count > 0 and diffusion_seq_count > 0:
                overlap_count = len(global_sequence_tracker['mlm_sequences'] & global_sequence_tracker['diffusion_sequences'])
                overlap_percent = (overlap_count / min(mlm_seq_count, diffusion_seq_count)) * 100
                logger.info(f"📊 PHASE OVERLAP: MLM={mlm_seq_count}, Diffusion={diffusion_seq_count}, "
                            f"Overlap={overlap_count} ({overlap_percent:.1f}%)")
            elif mlm_seq_count > 0:
                logger.info(f"📊 PHASE SEQUENCES: MLM={mlm_seq_count}, Diffusion={diffusion_seq_count}")
            elif diffusion_seq_count > 0:
                logger.info(f"📊 PHASE SEQUENCES: MLM={mlm_seq_count}, Diffusion={diffusion_seq_count}")
            
            # Sync eager validation model weights before running validation
            try:
                # Both models now have the same structure (compiled), so we can sync directly
                src_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                val_model.load_state_dict(src_state, strict=True)  # Use strict=True since structures should match
                if is_main_process():
                    logger.info(f"✅ Successfully synced validation model weights from training model")
                    # Debug: Check if validation model is in eval mode
                    val_model.eval()
                    logger.info(f"✅ Validation model set to eval mode")
            except Exception as e:
                if is_main_process():
                    logger.warning(f"⚠️ Failed to sync validation model weights: {e}")
                    logger.warning(f"⚠️ This may cause validation to use stale weights!")
            
            # Debug: Check if weights are actually synced (only on main process)
            if is_main_process():
                try:
                    # Compare a few key weights between training and validation models
                    train_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                    val_state = val_model.state_dict()
                    
                    # Check first few parameters (both models should have same structure now)
                    train_keys = list(train_state.keys())[:3]
                    for key in train_keys:
                        if key in val_state:
                            train_param = train_state[key]
                            val_param = val_state[key]
                            if torch.allclose(train_param, val_param, atol=1e-6):
                                logger.info(f"✅ Weight sync check: {key} - weights match")
                            else:
                                logger.warning(f"⚠️ Weight sync check: {key} - weights DIFFER!")
                                logger.warning(f"⚠️ Max diff: {torch.max(torch.abs(train_param - val_param)).item()}")
                        else:
                            logger.warning(f"⚠️ Weight sync check: {key} - missing in validation model (should not happen)")
                except Exception as e:
                    logger.warning(f"⚠️ Could not check weight sync: {e}")
            
            # Log overall sequences per second at validation (all ranks participate in allreduce)
            current_time = time.time()
            total_elapsed_time = current_time - sequences_per_second_start_time
            if is_main_process():
                logger.info(f"📊 OVERALL PERFORMANCE: {total_elapsed_time:.1f}s elapsed")
            
            # Debug: Double-check validation model state before validation
            if is_main_process():
                logger.info(f"🔍 Pre-validation check: val_model training mode = {val_model.training}")
                logger.info(f"🔍 Pre-validation check: val_model device = {next(val_model.parameters()).device}")
            
            val_result = validate(val_model, val_loader, device, config, vocab_json, valid_token_indices_by_modality, apply_vocab_constraints, logger, mlm_metrics_calculator, diffusion_metrics_calculator, current_phase, noise_scheduler, data_preparator, diffusion_loss)
            
            if current_phase == "MLM":
                if len(val_result) == 2:
                    val_loss, val_metrics = val_result
                    # --- WANDB LOG VALIDATION ---
                    if 'overall' in val_metrics:
                        wandb.log({'val_mlm/overall_accuracy': val_metrics['overall']['accuracy']}, step=step)
                        wandb.log({'val_mlm/overall_loss': val_metrics['overall']['loss']}, step=step)
                        wandb.log({'val_mlm/overall_perplexity': val_metrics['overall']['perplexity']}, step=step)
                    
                    # Log overall sequences per second at validation
                    if is_main_process():
                        wandb.log({
                            'overall/sequences_per_second': sequences_per_second_global,
                            'overall/total_elapsed_time': total_elapsed_time
                        }, step=step)
                    
                    if 'modalities' in val_metrics:
                        for mod_name, metrics in val_metrics['modalities'].items():
                            wandb.log({f'val_mlm/{mod_name}_accuracy': metrics['accuracy']}, step=step)
                            wandb.log({f'val_mlm/{mod_name}_loss': metrics['loss']}, step=step)
                            wandb.log({f'val_mlm/{mod_name}_perplexity': metrics['perplexity']}, step=step)
                    
                    # Validation stats
                    if 'training_stats' in val_metrics:
                        stats = val_metrics['training_stats']
                        wandb.log({
                            'val_mlm/total_sequences': stats['total_sequences'],
                            'val_mlm/unique_sequences': stats['unique_sequences'],
                            'val_mlm/total_dropped_tokens': stats['total_dropped_tokens'],
                            'val_mlm/percent_sequence_masked': config['training']['MLM_masking']['rate'] * 100.0,
                            'val_mlm/min_sequence_length': stats['min_sequence_length'],
                            'val_mlm/max_sequence_length': stats['max_sequence_length'],
                            'val_mlm/avg_sequence_length': stats['avg_sequence_length']
                        }, step=step)
                        
                        # Log top token prediction bar charts for each modality (validation)
                        if 'top_token_predictions' in val_metrics:
                            for modality_name in ['sequence', 'secondary_structure', 'consensus', 'go_terms']:
                                chart = create_token_prediction_bar_chart(
                                    val_metrics['top_token_predictions'], 
                                    vocab_json, 
                                    modality_name, 
                                    'val_mlm'
                                )
                                if chart:
                                    wandb.log({f'val_mlm/top_tokens_{modality_name}_bar': chart}, step=step)
                        
                        # Log modality dropout combinations as bar chart (MLM validation)
                        if 'modality_dropout_combinations' in val_metrics:
                            dropout_combos = val_metrics['modality_dropout_combinations']
                            if dropout_combos:
                                # Create bar chart data
                                combo_names = list(dropout_combos.keys())
                                combo_counts = list(dropout_combos.values())
                                
                                # Log as wandb bar chart
                                wandb.log({
                                    "val_mlm/dropout_combinations_bar": wandb.plot.bar(
                                        wandb.Table(data=[[name, count] for name, count in zip(combo_names, combo_counts)], 
                                                    columns=["Modality_Combination", "Count"]),
                                        "Modality_Combination", "Count",
                                        title="MLM Validation Modality Dropout Combinations"
                                    )
                                }, step=step)
                else:
                    val_loss = val_result
            else:
                # Diffusion phase validation
                if len(val_result) == 2:
                    val_loss, val_metrics = val_result
                    
                    # WandB logging for diffusion validation metrics
                    if 'generation' in val_metrics:
                        gen_metrics = val_metrics['generation']
                        wandb.log({
                            'val_diff/generation_accuracy': gen_metrics['accuracy'],
                            'val_diff/generation_loss': gen_metrics['loss'],
                            'val_diff/generation_perplexity': gen_metrics['perplexity']
                        }, step=step)
                    
                    # Log overall sequences per second at validation (same for diffusion)
                    if is_main_process():
                        wandb.log({
                            'overall/sequences_per_second': sequences_per_second_global,
                            'overall/total_elapsed_time': total_elapsed_time
                        }, step=step)
                    
                    if 'conditioning' in val_metrics:
                        for cond_type, metrics in val_metrics['conditioning'].items():
                            wandb.log({
                                f'val_diff/conditioning_{cond_type}_accuracy': metrics['accuracy'],
                                f'val_diff/conditioning_{cond_type}_loss': metrics['loss']
                            }, step=step)
                    
                    # Validation stats
                    if 'training_stats' in val_metrics:
                        stats = val_metrics['training_stats']
                        wandb.log({
                            'val_diff/total_samples': stats['total_samples'],
                            'val_diff/unique_sequences': stats['unique_sequences'],
                            'val_diff/avg_corruption_rate': stats['avg_corruption_rate'],
                            'val_diff/min_corruption_rate': stats['min_corruption_rate'],
                            'val_diff/max_corruption_rate': stats['max_corruption_rate'],
                            'val_diff/percent_sequence_masked': stats['avg_corruption_rate'] * 100.0,
                            'val_diff/min_sequence_length': stats['min_sequence_length'],
                            'val_diff/max_sequence_length': stats['max_sequence_length'],
                            'val_diff/avg_sequence_length': stats['avg_sequence_length']
                        }, step=step)
                        
                        # Log top token prediction bar chart for sequence modality (validation)
                        if 'top_token_predictions' in val_metrics:
                            chart = create_token_prediction_bar_chart(
                                val_metrics['top_token_predictions'], 
                                vocab_json, 
                                'sequence', 
                                'val_diff'
                            )
                            if chart:
                                wandb.log({f'val_diff/top_tokens_sequence_bar': chart}, step=step)
                        
                        # Log modality dropout combinations as bar chart (validation)
                        if 'modality_dropout_combinations' in val_metrics:
                            dropout_combos = val_metrics['modality_dropout_combinations']
                            if dropout_combos:
                                # Create bar chart data
                                combo_names = list(dropout_combos.keys())
                                combo_counts = list(dropout_combos.values())
                                
                                # Log as wandb bar chart
                                wandb.log({
                                    "val_diff/dropout_combinations_bar": wandb.plot.bar(
                                        wandb.Table(data=[[name, count] for name, count in zip(combo_names, combo_counts)], 
                                                    columns=["Modality_Combination", "Count"]),
                                        "Modality_Combination", "Count",
                                        title="Validation Modality Dropout Combinations"
                                    )
                                }, step=step)
                else:
                    val_loss = val_result
        
        # Periodic model saving
        if step % model_save_freq_steps == 0 and step > 0 and is_main_process():
            output_dir = resolve_output_dir(config)
            os.makedirs(output_dir, exist_ok=True)
            # Get checkpoint prefix from config, with fallback to default
            checkpoint_prefix = config.get('checkpoint', {}).get('checkpoint_prefix', 'modernbert_two_phase')
            ckpt_path = os.path.join(output_dir, f"{checkpoint_prefix}_step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                # Plain-dict record of the ModernBERT architecture so that
                # inference/rna_sequence_generator.py can rebuild the model
                # without being handed a matching inference YAML.
                'model_config_dict': serializable_model_config(model_config_dict),
                'current_phase': current_phase
            }, ckpt_path)
            logger.info(f"💾 Saved checkpoint at step {step}: {ckpt_path}")
        
        # Stop training if total_steps is reached (max_steps is used for scheduler, not early stopping)
        if step >= total_steps:
            logger.info(f"[TRAINING] Reached total_steps ({total_steps}). Stopping training.")
            break
    
    # Save final checkpoint (same layout as the periodic ones, so that resuming
    # from it and running inference on it both work)
    if is_main_process():
        output_dir = resolve_output_dir(config)
        os.makedirs(output_dir, exist_ok=True)
        final_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        # Get checkpoint prefix from config, with fallback to default
        checkpoint_prefix = config.get('checkpoint', {}).get('checkpoint_prefix', 'modernbert_two_phase')
        ckpt_path = os.path.join(output_dir, f"{checkpoint_prefix}_final.pt")
        torch.save({
            'step': step,
            'model_state_dict': final_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'model_config_dict': serializable_model_config(model_config_dict),
            'current_phase': current_phase
        }, ckpt_path)
        logger.info(f"Saved final checkpoint: {ckpt_path}")
    
    # Final phase overlap analysis
    mlm_seq_count = len(global_sequence_tracker['mlm_sequences'])
    diffusion_seq_count = len(global_sequence_tracker['diffusion_sequences'])
    if mlm_seq_count > 0 and diffusion_seq_count > 0:
        overlap_count = len(global_sequence_tracker['mlm_sequences'] & global_sequence_tracker['diffusion_sequences'])
        overlap_percent = (overlap_count / min(mlm_seq_count, diffusion_seq_count)) * 100
        logger.info(f"\n🔍 FINAL PHASE OVERLAP ANALYSIS:")
        logger.info(f"  MLM Phase: {mlm_seq_count} unique sequences")
        logger.info(f"  Diffusion Phase: {diffusion_seq_count} unique sequences") 
        logger.info(f"  Overlap: {overlap_count} sequences ({overlap_percent:.1f}%)")
        logger.info(f"  Data Leakage: {'HIGH' if overlap_percent > 50 else 'MODERATE' if overlap_percent > 10 else 'LOW'}")
    else:
        logger.info(f"\n🔍 FINAL SEQUENCE COUNTS: MLM={mlm_seq_count}, Diffusion={diffusion_seq_count}")
    
    # Check if test evaluation is enabled
    eval_on_test = config['training'].get('eval_on_test', True)
    if eval_on_test:
        # Run test set evaluation. `training.test_num_samples` caps how much of
        # the test split is consumed (None/absent walks all of it, which on the
        # full Rfam set is ~5% of 4M sequences and takes hours).
        test_num_samples = config['training'].get('test_num_samples', None)
        if test_num_samples is not None:
            logger.info(f"\n[TEST] Running evaluation on test set (capped at {test_num_samples})...")
        else:
            logger.info(f"\n[TEST] Running evaluation on the full test set "
                        f"({len(test_loader.dataset)} sequences); set training.test_num_samples to cap it...")
        test_result = evaluate(model, test_loader, device, config, vocab_json, valid_token_indices_by_modality, apply_vocab_constraints, logger, split_name="TEST", num_samples=test_num_samples, mlm_metrics_calculator=mlm_metrics_calculator, diffusion_metrics_calculator=diffusion_metrics_calculator, current_phase=current_phase, noise_scheduler=noise_scheduler, data_preparator=data_preparator, diffusion_loss=diffusion_loss)
    else:
        logger.info(f"\n[TEST] Test evaluation disabled (eval_on_test: {eval_on_test})")
        test_result = None
    
    # --- WANDB LOG EVAL ---
    if eval_on_test and test_result is not None:
        if current_phase == "MLM":
            if len(test_result) == 2:
                test_loss, test_metrics = test_result
                # --- WANDB LOG TEST ---
                if 'overall' in test_metrics:
                    wandb.log({'test_mlm/overall_accuracy': test_metrics['overall']['accuracy']}, step=step)
                    wandb.log({'test_mlm/overall_loss': test_metrics['overall']['loss']}, step=step)
                    wandb.log({'test_mlm/overall_perplexity': test_metrics['overall']['perplexity']}, step=step)
                
                if 'modalities' in test_metrics:
                    for mod_name, metrics in test_metrics['modalities'].items():
                        wandb.log({f'test_mlm/{mod_name}_accuracy': metrics['accuracy']}, step=step)
                        wandb.log({f'test_mlm/{mod_name}_loss': metrics['loss']}, step=step)
                        wandb.log({f'test_mlm/{mod_name}_perplexity': metrics['perplexity']}, step=step)
                
                # Test stats
                if 'training_stats' in test_metrics:
                    stats = test_metrics['training_stats']
                    wandb.log({
                        'test_mlm/total_sequences': stats['total_sequences'],
                        'test_mlm/unique_sequences': stats['unique_sequences'],
                        'test_mlm/percent_sequence_masked': config['training']['MLM_masking']['rate'] * 100.0,
                        'test_mlm/min_sequence_length': stats['min_sequence_length'],
                        'test_mlm/max_sequence_length': stats['max_sequence_length'],
                        'test_mlm/avg_sequence_length': stats['avg_sequence_length']
                    }, step=step)
                    
                    # Log top token prediction bar charts for each modality (test)
                    if 'top_token_predictions' in test_metrics:
                        for modality_name in ['sequence', 'secondary_structure', 'consensus', 'go_terms']:
                            chart = create_token_prediction_bar_chart(
                                test_metrics['top_token_predictions'], 
                                vocab_json, 
                                modality_name, 
                                'test_mlm'
                            )
                            if chart:
                                wandb.log({f'test_mlm/top_tokens_{modality_name}_bar': chart}, step=step)
            else:
                test_loss = test_result
        else:
            # Diffusion phase test evaluation
            if len(test_result) == 2:
                test_loss, test_metrics = test_result
                
                # WandB logging for diffusion test metrics
                if 'generation' in test_metrics:
                    gen_metrics = test_metrics['generation']
                    wandb.log({
                        'test_diff/generation_accuracy': gen_metrics['accuracy'],
                        'test_diff/generation_loss': gen_metrics['loss'],
                        'test_diff/generation_perplexity': gen_metrics['perplexity']
                    }, step=step)
                
                if 'conditioning' in test_metrics:
                    for cond_type, metrics in test_metrics['conditioning'].items():
                        wandb.log({
                            f'test_diff/conditioning_{cond_type}_accuracy': metrics['accuracy'],
                            f'test_diff/conditioning_{cond_type}_loss': metrics['loss']
                        }, step=step)
                
                # Test stats
                if 'training_stats' in test_metrics:
                    stats = test_metrics['training_stats']
                    wandb.log({
                        'test_diff/total_samples': stats['total_samples'],
                        'test_diff/unique_sequences': stats['unique_sequences'],
                        'test_diff/avg_corruption_rate': stats['avg_corruption_rate'],
                        'test_diff/min_corruption_rate': stats['min_corruption_rate'],
                        'test_diff/max_corruption_rate': stats['max_corruption_rate'],
                        'test_diff/percent_sequence_masked': stats['avg_corruption_rate'] * 100.0,
                        'test_diff/min_sequence_length': stats['min_sequence_length'],
                        'test_diff/max_sequence_length': stats['max_sequence_length'],
                        'test_diff/avg_sequence_length': stats['avg_sequence_length']
                    }, step=step)
                    
                    # Log top token prediction bar chart for sequence modality (test)
                    if 'top_token_predictions' in test_metrics:
                        chart = create_token_prediction_bar_chart(
                            test_metrics['top_token_predictions'], 
                            vocab_json, 
                            'sequence', 
                            'test_diff'
                        )
                        if chart:
                            wandb.log({f'test_diff/top_tokens_sequence_bar': chart}, step=step)
            else:
                test_loss = test_result
    else:
        logger.info(f"📊 Test evaluation skipped (eval_on_test: {eval_on_test})")

    # Cleanup RAM files if they were created (rank 0 only) and finish W&B
    if is_main_process():
        cleanup_ram_files(config, logger)
        try:
            wandb.finish()
        except Exception:
            pass
    cleanup_distributed()

    # End training cleanly; do NOT recurse
    return 0

if __name__ == "__main__":
    # `args` (including any --<name> overrides) comes from the argument parser at
    # the top of this module's __main__ block, which also set up logging and W&B.
    sys.exit(train(args))
