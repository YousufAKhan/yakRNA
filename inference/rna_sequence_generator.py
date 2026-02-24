"""
RNA Sequence Generator

Comprehensive inference script for generating RNA sequences using the trained multimodal diffusion model.
Supports conditional generation with optional secondary structure, consensus sequence, and GO terms.

Usage:
    # Generate 5 sequences of length 50 with no conditioning
    python inference/rna_sequence_generator.py --config configs/cluster_training.yaml --checkpoint outputs/model.pt --num_sequences 5 --length 50
    
    # Generate with secondary structure conditioning
    python inference/rna_sequence_generator.py --config configs/cluster_training.yaml --checkpoint outputs/model.pt --num_sequences 3 --length 100 --secondary_structure "(((...)))"
    
    # Fill in masked sequence
    python inference/rna_sequence_generator.py --config configs/cluster_training.yaml --checkpoint outputs/model.pt --masked_sequence "AUGC***CGAU" --secondary_structure "(((...)))"
    
    # Generate with all modalities
    python inference/rna_sequence_generator.py --config configs/cluster_training.yaml --checkpoint outputs/model.pt --num_sequences 2 --length 200 --secondary_structure "(((...)))" --consensus "AUGC" --go_terms "GO:0003676,GO:0005515"

Features:
- Generate x sequences of length n (up to 640 nucleotides)
- Support for conditional generation with secondary structure, consensus, and GO terms
- Handle masked sequences (e.g., "AUG***CG") for infilling
- Automatically handle missing modalities with [DROPPED] tokens
- Progressive denoising using diffusion sampling
- Temperature control for generation diversity
- Batch generation for efficiency
"""

import sys
import os
import argparse
import random
import yaml
import torch
import json
import logging
from typing import Dict, List, Union, Optional, Set
import torch.nn.functional as F
from datetime import datetime
from collections import defaultdict

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rna_gen'))

from transformers import ModernBertConfig, ModernBertForMaskedLM
from discrete_diffusion import create_diffusion_components
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# -----------------------
# Constraint utilities
# -----------------------

BRACKET_PAIRS = {'(': ')', '<': '>', '[': ']', '{': '}'}
OPENING_BRACKETS = set(BRACKET_PAIRS.keys())
CLOSING_BRACKETS = set(BRACKET_PAIRS.values())
REVERSE_BRACKET_PAIRS = {v: k for k, v in BRACKET_PAIRS.items()}

PAIR_SETS = {
    'strict': {'A:U', 'U:A', 'G:C', 'C:G'},
    'canonical': {'A:U', 'U:A', 'G:C', 'C:G', 'G:U', 'U:G'},
    'canonical+sheared': {'A:U', 'U:A', 'G:C', 'C:G', 'G:U', 'U:G', 'G:A', 'A:G'},
    'canonical+common': {'A:U', 'U:A', 'G:C', 'C:G', 'G:U', 'U:G', 'G:A', 'A:G', 'A:C', 'C:A'},
    'permissive': {'A:U', 'U:A', 'G:C', 'C:G', 'G:U', 'U:G', 'G:A', 'A:G', 'A:C', 'C:A', 'U:C', 'C:U'}
}


def get_predefined_constraint_set(name: str) -> Set[str]:
    """Return allowed base pairs for a named constraint set."""
    if name not in PAIR_SETS:
        raise ValueError(f"Unknown constraint set '{name}'. Options: {list(PAIR_SETS.keys())}")
    return PAIR_SETS[name]


class StructureConstrainedGenerator:
    """Minimal constraint helper for base-pair validation/masking."""

    def __init__(self, vocabulary, allowed_pairs: Optional[Set[str]] = None):
        self.vocab = vocabulary
        self.allowed_pairs = allowed_pairs if allowed_pairs is not None else PAIR_SETS['canonical']
        self._build_compatibility_map()

    def _build_compatibility_map(self) -> None:
        self.base_to_partners = defaultdict(set)
        for pair in self.allowed_pairs:
            if ':' in pair and len(pair) == 3:
                a, b = pair.split(':')
                self.base_to_partners[a].add(b)
                self.base_to_partners[b].add(a)

    def parse_structure(self, structure: str) -> Dict[int, int]:
        stacks = {b: [] for b in OPENING_BRACKETS}
        pairs: Dict[int, int] = {}
        for i, char in enumerate(structure):
            if char in OPENING_BRACKETS:
                stacks[char].append(i)
            elif char in CLOSING_BRACKETS:
                opening = REVERSE_BRACKET_PAIRS[char]
                if stacks[opening]:
                    start = stacks[opening].pop()
                    pairs[start] = i
                    pairs[i] = start
        return pairs

    def get_compatible_bases(self, base: str) -> Set[str]:
        return self.base_to_partners.get(base, {'A', 'U', 'G', 'C'})

    def check_constraints(self, sequence: str, structure: str) -> (bool, List[str]):
        pairs = self.parse_structure(structure)
        violations = []
        for i, j in pairs.items():
            if i < j:
                pair = f"{sequence[i]}:{sequence[j]}"
                if pair not in self.allowed_pairs:
                    violations.append(f"{i}-{j}:{pair}")
        return len(violations) == 0, violations

    def calculate_constraint_satisfaction(self, sequence: str, structure: str) -> float:
        pairs = self.parse_structure(structure)
        if not pairs:
            return 1.0
        checked = 0
        ok = 0
        for i, j in pairs.items():
            if i < j:
                checked += 1
                if f"{sequence[i]}:{sequence[j]}" in self.allowed_pairs:
                    ok += 1
        return ok / checked if checked else 1.0


class RNASequenceGenerator:
    """
    RNA sequence generator using trained multimodal diffusion model.
    """
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "auto"):
        """
        Initialize the RNA sequence generator.
        
        Args:
            config_path: Path to training configuration file
            checkpoint_path: Path to trained model checkpoint (relative or absolute)
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.config_path = os.path.abspath(config_path)
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        
        # Validate paths
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info("Using device: %s", self.device)
        
        # Load configuration and model
        self._load_config()
        self._load_vocabulary()
        self._load_model()
        self._setup_diffusion_components()
        
        logger.info("RNA sequence generator initialized successfully")
    
    def _load_config(self) -> None:
        """Load training configuration."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("Loaded configuration from %s", self.config_path)
    
    def _load_vocabulary(self) -> None:
        """Load unified vocabulary."""
        # Try different possible vocabulary path keys
        if 'data' in self.config and 'unified_vocab_path' in self.config['data']:
            vocab_path = self.config['data']['unified_vocab_path']
        elif 'data_files' in self.config and 'vocabulary_file' in self.config['data_files']:
            vocab_path = self.config['data_files']['vocabulary_file']
        else:
            raise KeyError("Could not find vocabulary path in config. Expected 'data.unified_vocab_path' or 'data_files.vocabulary_file'")
        
        logger.info("Loading vocabulary from %s", vocab_path)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab_json = json.load(f)
        
        # Get special tokens
        self.special_tokens = self.vocab_json['special_tokens']
        self.modality_offsets = self.vocab_json['modality_offsets']
        self.modality_vocabs = self.vocab_json['modality_vocabs']
        
        # Get vocabulary size
        self.vocab_size = self.vocab_json['vocab_size']
        
        logger.info("Loaded vocabulary with %d tokens", self.vocab_size)
    
    def _load_model(self) -> None:
        """Load the trained model."""
        # Get model architecture parameters - handle nested structure
        arch_config = self.config['model_architecture']
        if 'model_size' in arch_config:
            # Use nested model_size structure
            model_size = arch_config['model_size']
            hidden_size = model_size['hidden_size']
            num_hidden_layers = model_size['num_hidden_layers']
            num_attention_heads = model_size['num_attention_heads']
            intermediate_size = model_size['intermediate_size']
        else:
            # Use flat structure
            hidden_size = arch_config['hidden_size']
            num_hidden_layers = arch_config['num_hidden_layers']
            num_attention_heads = arch_config['num_attention_heads']
            intermediate_size = arch_config['intermediate_size']
        
        # Get max position embeddings from data config
        if 'max_position_embeddings' in arch_config:
            max_position_embeddings = arch_config['max_position_embeddings']
        elif 'data' in self.config and 'max_position_embeddings' in self.config['data']:
            max_position_embeddings = self.config['data']['max_position_embeddings']
        else:
            max_position_embeddings = 2048  # Default fallback
        
        # Load checkpoint first to check for saved config
        logger.info("Loading checkpoint from %s", self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Try to get model config from checkpoint first
        if 'model_config' in checkpoint:
            logger.info("Using model config from checkpoint")
            model_config = checkpoint['model_config']
            # Update vocab size to match current vocabulary
            model_config.vocab_size = self.vocab_size
        else:
            logger.info("No model config in checkpoint, creating from inference config")
            # Setup ModernBERT configuration with compatible settings
            model_config = ModernBertConfig(
                vocab_size=self.vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                max_position_embeddings=max_position_embeddings,
                attention_dropout=self.config['training']['dropout']['attention_dropout'],
                hidden_dropout=self.config['training']['dropout']['hidden_dropout'],
                use_cache=False,
                pad_token_id=self.special_tokens['[PAD]'],
                classifier_dropout=0.1,
                norm_bias=False,
                embedding_dropout=0.1,
                hidden_activation='gelu',  # Use 'gelu' instead of 'swiglu' for compatibility
                attention_bias=False,
                normalization='RMSNorm',
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        
        # Create model with the config
        self.model = ModernBertForMaskedLM(model_config)
        self.model.to(self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove any prefixes from DDP or torch.compile
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key
            # Remove 'module.' prefix from DDP
            if clean_key.startswith('module.'):
                clean_key = clean_key[7:]
            # Remove '_orig_mod.' prefix from torch.compile
            if clean_key.startswith('_orig_mod.'):
                clean_key = clean_key[10:]
            cleaned_state_dict[clean_key] = value
        
        self.model.load_state_dict(cleaned_state_dict, strict=True)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def _setup_diffusion_components(self) -> None:
        """Setup diffusion components for sampling."""
        # Create diffusion components using the same configuration structure
        diffusion_config = self.config['training']['diffusion']
        diffusion_components_config = {
            'training': {
                'discrete_diffusion': {
                    'num_timesteps': diffusion_config['num_timesteps'],
                    'loss_type': diffusion_config['loss_type'],
                    'masking_rates': diffusion_config.get('masking_rates', None),
                    'cumulative_masking': diffusion_config.get('cumulative_masking', None),
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
        
        self.noise_scheduler, self.data_preparator, self.diffusion_loss = create_diffusion_components(
            diffusion_components_config, self.vocab_json
        )
        
        logger.info("Diffusion components initialized")
    
    def encode_sequence_input(self, 
                            sequence: str = None,
                            secondary_structure: str = None,
                            consensus: str = None,
                            go_terms: List[str] = None,
                            length: int = None) -> Dict[str, torch.Tensor]:
        """
        Encode input for generation.
        
        Args:
            sequence: RNA sequence (if None, will be filled with masks for specified length)
            secondary_structure: Secondary structure string
            consensus: Consensus sequence
            go_terms: List of GO term strings
            length: Desired sequence length (used when sequence is None or contains masks)
        
        Returns:
            Dictionary with encoded tensors
        """
        tokens = []
        modality_type_ids = []
        
        # Add CLS token
        tokens.append(self.special_tokens['[CLS]'])
        modality_type_ids.append(-1)  # Special token
        
        # Handle sequence modality
        tokens.append(self.special_tokens['[SEQ_START]'])
        modality_type_ids.append(-1)  # Special token
        
        if sequence is not None:
            # Encode provided sequence (may contain masks)
            seq_offset = self.modality_offsets['sequence']
            seq_vocab = self.modality_vocabs['sequence']
            
            for char in sequence:
                if char == '*':
                    # Mask token for infilling
                    tokens.append(self.special_tokens['[MASK]'])
                    modality_type_ids.append(0)
                elif char.upper() in seq_vocab:
                    char_idx = seq_vocab.index(char.upper())
                    tokens.append(seq_offset + char_idx)
                    modality_type_ids.append(0)
                else:
                    # Unknown character - use RNA-UNK
                    unk_idx = seq_vocab.index('[RNA-UNK]')
                    tokens.append(seq_offset + unk_idx)
                    modality_type_ids.append(0)
        else:
            # Generate entirely from masks
            if length is None:
                raise ValueError("Either sequence or length must be provided")
            
            # Determine actual sequence length based on conditioning modalities
            target_seq_length = length
            
            # Validate consistent lengths for provided modalities
            modality_lengths = []
            if secondary_structure is not None:
                modality_lengths.append(("secondary_structure", len(secondary_structure)))
                target_seq_length = len(secondary_structure)
            if consensus is not None:
                modality_lengths.append(("consensus", len(consensus)))
                if target_seq_length is None:
                    target_seq_length = len(consensus)
            
            # Check for length mismatches
            if len(modality_lengths) > 1:
                lengths = [length for _, length in modality_lengths]
                if len(set(lengths)) > 1:
                    length_info = ", ".join([f"{name}: {length}" for name, length in modality_lengths])
                    logger.warning("Modality length mismatch detected: %s. Using secondary structure length if available.", length_info)
            
            logger.info("Target sequence length: %d", target_seq_length)
            logger.info("Adding %d mask tokens to sequence", target_seq_length)
            for i in range(target_seq_length):
                tokens.append(self.special_tokens['[MASK]'])
                modality_type_ids.append(0)
            logger.info("After adding masks: tokens=%d, modality_ids=%d", len(tokens), len(modality_type_ids))
        
        # Handle secondary structure modality
        tokens.append(self.special_tokens['[STRUCT_START]'])
        modality_type_ids.append(1)  # SS modality
        
        if secondary_structure is not None:
            ss_offset = self.modality_offsets['ss']
            ss_vocab = self.modality_vocabs['ss']
            logger.info("Processing secondary structure: '%s' (length: %d)", secondary_structure, len(secondary_structure))
            for i, char in enumerate(secondary_structure):
                if char in ss_vocab:
                    char_idx = ss_vocab.index(char)
                    tokens.append(ss_offset + char_idx)
                    modality_type_ids.append(1)
                else:
                    logger.warning("SS char '%s' at position %d not found in vocab. Available: %s", char, i, ss_vocab[:10])
                    unk_idx = ss_vocab.index('[STRUC-UNK]')
                    tokens.append(ss_offset + unk_idx)
                    modality_type_ids.append(1)
        else:
            # No secondary structure provided - use [DROPPED] tokens matching sequence length
            for _ in range(target_seq_length):
                tokens.append(self.special_tokens['[DROPPED]'])
                modality_type_ids.append(1)
        
        # Handle consensus modality
        tokens.append(self.special_tokens['[CONS_START]'])
        modality_type_ids.append(2)  # Consensus modality
        
        if consensus is not None:
            cons_offset = self.modality_offsets['consensus']
            cons_vocab = self.modality_vocabs['consensus']
            logger.info("Processing consensus: '%s' (length: %d)", consensus, len(consensus))
            for i, char in enumerate(consensus):
                if char in cons_vocab:
                    char_idx = cons_vocab.index(char)
                    tokens.append(cons_offset + char_idx)
                    modality_type_ids.append(2)
                else:
                    logger.warning("Consensus char '%s' at position %d not found in vocab. Available: %s", char, i, cons_vocab[:10])
                    unk_idx = cons_vocab.index('[CONS-UNK]')
                    tokens.append(cons_offset + unk_idx)
                    modality_type_ids.append(2)
        else:
            # No consensus provided - use [DROPPED] tokens matching sequence length
            for _ in range(target_seq_length):
                tokens.append(self.special_tokens['[DROPPED]'])
                modality_type_ids.append(2)
        
        # Handle GO terms modality
        tokens.append(self.special_tokens['[GO_START]'])
        modality_type_ids.append(3)  # GO terms modality
        
        if go_terms is not None and len(go_terms) > 0:
            go_offset = self.modality_offsets['go_terms']
            go_vocab = self.modality_vocabs['go_terms']
            for term in go_terms:
                if term in go_vocab:
                    term_idx = go_vocab.index(term)
                    tokens.append(go_offset + term_idx)
                    modality_type_ids.append(3)
                else:
                    unk_idx = go_vocab.index('[GO-UNK]')
                    tokens.append(go_offset + unk_idx)
                    modality_type_ids.append(3)
        else:
            # No GO terms provided - use single [DROPPED] token
            tokens.append(self.special_tokens['[DROPPED]'])
            modality_type_ids.append(3)
        
        # Add SEP token
        tokens.append(self.special_tokens['[SEP]'])
        modality_type_ids.append(-1)  # Special token
        
        # Debug: Log final token and modality counts
        logger.info("Final encoding: %d tokens, %d modality_ids", len(tokens), len(modality_type_ids))
        sequence_count = sum(1 for mid in modality_type_ids if mid == 0)
        logger.info("Sequence modality positions: %d", sequence_count)
        
        # Debug: Find which positions are marked as sequence (type 0)
        sequence_positions_debug = [i for i, mid in enumerate(modality_type_ids) if mid == 0]
        logger.info("Sequence positions (first 10): %s", sequence_positions_debug[:10])
        logger.info("Sequence positions (last 10): %s", sequence_positions_debug[-10:])
        
        # Check if there are any unexpected sequence positions
        expected_seq_start = 2  # Should start after [CLS] and [SEQ_START] tokens
        expected_seq_end = expected_seq_start + target_seq_length
        unexpected_positions = [pos for pos in sequence_positions_debug if pos < expected_seq_start or pos >= expected_seq_end]
        if unexpected_positions:
            logger.warning("Unexpected sequence positions found: %s", unexpected_positions)
            # Debug: Show what tokens are at the unexpected positions
            for pos in unexpected_positions:
                token_id = tokens[pos]
                mod_id = modality_type_ids[pos]
                logger.warning("Position %d: token_id=%d, modality_id=%d", pos, token_id, mod_id)
                # Try to decode the token
                if token_id == self.special_tokens['[STRUCT_START]']:
                    logger.warning("  -> This is [STRUCT_START] but marked as sequence!")
                elif token_id == self.special_tokens['[MASK]']:
                    logger.warning("  -> This is [MASK] but outside expected sequence range!")
                else:
                    logger.warning("  -> Unknown token at unexpected position")
        
        # Convert to tensors
        input_ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        modality_type_ids = torch.tensor(modality_type_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'modality_type_ids': modality_type_ids
        }
    
    def decode_input_for_display(self, input_ids: torch.Tensor, modality_type_ids: torch.Tensor) -> str:
        """
        Decode the input tokens to show what the model sees, with mask tokens as *** and dropped modalities as X.
        
        Args:
            input_ids: Input token IDs
            modality_type_ids: Modality type IDs
            
        Returns:
            Human-readable string showing model input
        """
        display_parts = []
        tokens = input_ids.squeeze(0)
        mod_ids = modality_type_ids.squeeze(0)
        
        current_modality = None
        current_part = []
        
        for token_id, mod_id in zip(tokens, mod_ids):
            token_id = token_id.item()
            mod_id = mod_id.item()
            
            # Handle modality transitions
            if mod_id != current_modality:
                if current_part:
                    if current_modality == 0:  # sequence
                        display_parts.append(f"SEQ: {''.join(current_part)}")
                    elif current_modality == 1:  # secondary structure
                        display_parts.append(f"STRUCT: {''.join(current_part)}")
                    elif current_modality == 2:  # consensus
                        display_parts.append(f"CONS: {''.join(current_part)}")
                    elif current_modality == 3:  # GO terms
                        display_parts.append(f"GO: {''.join(current_part)}")
                current_part = []
                current_modality = mod_id
            
            # Decode token based on modality
            if mod_id == -1:  # Special tokens
                if token_id == self.special_tokens['[CLS]']:
                    continue
                elif token_id == self.special_tokens['[SEP]']:
                    continue
                elif token_id == self.special_tokens['[SEQ_START]']:
                    continue
                elif token_id == self.special_tokens['[STRUCT_START]']:
                    continue
                elif token_id == self.special_tokens['[CONS_START]']:
                    continue
                elif token_id == self.special_tokens['[GO_START]']:
                    continue
            elif mod_id == 0:  # sequence modality
                if token_id == self.special_tokens['[MASK]']:
                    current_part.append('*')
                elif token_id == self.special_tokens['[DROPPED]']:
                    current_part.append('X')
                else:
                    # Decode sequence token
                    seq_offset = self.modality_offsets['sequence']
                    seq_vocab = self.modality_vocabs['sequence']
                    if seq_offset <= token_id < seq_offset + len(seq_vocab):
                        local_idx = token_id - seq_offset
                        char = seq_vocab[local_idx]
                        if char not in ['[RNA-UNK]', '[SEQ-GAP]']:
                            current_part.append(char)
                        else:
                            current_part.append('?')
            elif mod_id == 1:  # secondary structure modality
                if token_id == self.special_tokens['[DROPPED]']:
                    current_part.append('X')
                else:
                    # Decode SS token
                    ss_offset = self.modality_offsets['ss']
                    ss_vocab = self.modality_vocabs['ss']
                    if ss_offset <= token_id < ss_offset + len(ss_vocab):
                        local_idx = token_id - ss_offset
                        char = ss_vocab[local_idx]
                        if char not in ['[STRUC-UNK]', '[STRUC-GAP]']:
                            current_part.append(char)
                        else:
                            current_part.append('?')
            elif mod_id == 2:  # consensus modality
                if token_id == self.special_tokens['[DROPPED]']:
                    current_part.append('X')
                else:
                    # Decode consensus token
                    cons_offset = self.modality_offsets['consensus']
                    cons_vocab = self.modality_vocabs['consensus']
                    if cons_offset <= token_id < cons_offset + len(cons_vocab):
                        local_idx = token_id - cons_offset
                        char = cons_vocab[local_idx]
                        if char not in ['[CONS-UNK]', '[CONS-GAP]']:
                            current_part.append(char)
                        else:
                            current_part.append('?')
            elif mod_id == 3:  # GO terms modality
                if token_id == self.special_tokens['[DROPPED]']:
                    current_part.append('X')
                else:
                    # Decode GO term
                    go_offset = self.modality_offsets['go_terms']
                    go_vocab = self.modality_vocabs['go_terms']
                    if go_offset <= token_id < go_offset + len(go_vocab):
                        local_idx = token_id - go_offset
                        term = go_vocab[local_idx]
                        if term not in ['[GO-UNK]', '[GO-GAP]']:
                            current_part.append(term)
                        else:
                            current_part.append('?')
        
        # Handle final modality
        if current_part:
            if current_modality == 0:  # sequence
                display_parts.append(f"SEQ: {''.join(current_part)}")
            elif current_modality == 1:  # secondary structure
                display_parts.append(f"STRUCT: {''.join(current_part)}")
            elif current_modality == 2:  # consensus
                display_parts.append(f"CONS: {''.join(current_part)}")
            elif current_modality == 3:  # GO terms
                display_parts.append(f"GO: {','.join(current_part)}")
        
        return " | ".join(display_parts)
    
    def create_sequence_mask(self, modality_type_ids: torch.Tensor) -> torch.Tensor:
        """Create mask for sequence modality positions."""
        return (modality_type_ids == 0)
    
    def decode_sequence(self, tokens: torch.Tensor, modality_type_ids: torch.Tensor) -> str:
        """
        Decode tokens back to RNA sequence string.
        
        Args:
            tokens: Token IDs tensor
            modality_type_ids: Modality type IDs tensor
        
        Returns:
            RNA sequence string
        """
        sequence_chars = []
        seq_offset = self.modality_offsets['sequence']
        seq_vocab = self.modality_vocabs['sequence']
        
        # Find sequence positions
        sequence_mask = (modality_type_ids == 0)
        sequence_positions = torch.where(sequence_mask)[0].tolist()
        
        logger.debug("Sequence positions found: %d", len(sequence_positions))
        
        for i in range(len(tokens)):
            if sequence_mask[i]:
                token_id = tokens[i].item()
                # Convert global token ID to local vocabulary index
                if seq_offset <= token_id < seq_offset + len(seq_vocab):
                    local_idx = token_id - seq_offset
                    char = seq_vocab[local_idx]
                    # Skip special tokens like [RNA-UNK], [SEQ-GAP]
                    if char not in ['[RNA-UNK]', '[SEQ-GAP]']:
                        sequence_chars.append(char)
        
        result_sequence = ''.join(sequence_chars)
        
        # Debug: Print detailed decoding info for sequences that are too long
        if len(result_sequence) != len(sequence_positions):
            print("🔍 DECODING MISMATCH:")
            print(f"  Expected sequence positions: {len(sequence_positions)}")
            print(f"  Actual decoded length: {len(result_sequence)}")
            print(f"  Sequence mask sum: {sequence_mask.sum().item()}")
            print(f"  First 10 sequence chars: {''.join(sequence_chars[:10])}")
            print(f"  Last 10 sequence chars: {''.join(sequence_chars[-10:])}")
        
        return result_sequence
    
    def apply_modality_vocab_mask(self, logits: torch.Tensor, modality_type_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply vocabulary constraints to prevent invalid token predictions.
        
        Args:
            logits: Model logits (batch_size, seq_len, vocab_size)
            modality_type_ids: Modality type IDs (batch_size, seq_len)
        
        Returns:
            Masked logits with invalid tokens set to -inf
        """
        _, _, vocab_size = logits.shape
        masked_logits = logits.clone()
        
        # Create masks for each modality
        for modality_name, modality_id in [('sequence', 0), ('ss', 1), ('consensus', 2), ('go_terms', 3)]:
            modality_mask = (modality_type_ids == modality_id)
            if modality_mask.sum() == 0:
                continue
            
            # Get valid token range for this modality
            offset = self.modality_offsets[modality_name]
            vocab_size_mod = len(self.modality_vocabs[modality_name])
            
            # Create valid indices tensor
            valid_indices = torch.arange(offset, offset + vocab_size_mod, device=logits.device)
            
            # Also allow special tokens
            special_indices = torch.tensor([
                self.special_tokens['[MASK]'],
                self.special_tokens['[PAD]'],
                self.special_tokens['[DROPPED]']
            ], device=logits.device)
            
            all_valid_indices = torch.cat([valid_indices, special_indices])
            
            # Create mask for invalid tokens
            invalid_mask = torch.ones(vocab_size, dtype=torch.bool, device=logits.device)
            invalid_mask[all_valid_indices] = False
            
            # Apply mask to logits
            masked_logits[modality_mask] = masked_logits[modality_mask].masked_fill(
                invalid_mask.unsqueeze(0).expand(modality_mask.sum(), -1), -float('inf')
            )
        
        return masked_logits
    
    @torch.no_grad()
    def generate_sequences(self,
                         num_sequences: int = 1,
                         length: int = None,
                         sequence: str = None,
                         secondary_structure: str = None,
                         consensus: str = None,
                         go_terms: Union[str, List[str]] = None,
                         num_inference_steps: int = None,
                         temperature: float = 1.0,
                         top_k: int = None,
                         top_p: float = None) -> List[Dict[str, str]]:
        """
        Generate RNA sequences using diffusion sampling.
        
        Args:
            num_sequences: Number of sequences to generate
            length: Desired sequence length (required if sequence is None)
            sequence: Existing sequence with optional masks (*) for infilling
            secondary_structure: Secondary structure constraint
            consensus: Consensus sequence constraint
            go_terms: GO terms (string or list of strings)
            num_inference_steps: Number of denoising steps (default: use model's timesteps)
            temperature: Sampling temperature for diversity
            top_k: Top-k sampling
            top_p: Nucleus sampling
        
        Returns:
            List of generated sequences with metadata
        """
        # Parse GO terms
        if isinstance(go_terms, str):
            go_terms = [term.strip() for term in go_terms.split(',') if term.strip()]
        
        # Set default inference steps
        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.num_timesteps
        
        results = []
        
        # Generate sequences in batch
        for batch_start in range(0, num_sequences, 8):  # Process in batches of 8
            batch_end = min(batch_start + 8, num_sequences)
            batch_size = batch_end - batch_start
            
            # Encode input for this batch
            batch_inputs = []
            for _ in range(batch_size):
                encoded = self.encode_sequence_input(
                    sequence=sequence,
                    secondary_structure=secondary_structure,
                    consensus=consensus,
                    go_terms=go_terms,
                    length=length
                )
                batch_inputs.append(encoded)
            
            # Stack batch inputs
            batch_input_ids = torch.cat([inp['input_ids'] for inp in batch_inputs], dim=0)
            batch_attention_mask = torch.cat([inp['attention_mask'] for inp in batch_inputs], dim=0)
            batch_modality_type_ids = torch.cat([inp['modality_type_ids'] for inp in batch_inputs], dim=0)
            
            # Create sequence mask
            sequence_mask = self.create_sequence_mask(batch_modality_type_ids)
            
            # Debug: Show what the model input looks like (only for the first sample)
            if batch_start == 0:
                input_display = self.decode_input_for_display(batch_input_ids[0:1], batch_modality_type_ids[0:1])
                print(f"\n🔍 Model Input: {input_display}")
                
                # Count sequence positions and mask tokens
                seq_mask_count = sequence_mask[0].sum().item()
                mask_tokens_count = (batch_input_ids[0] == self.special_tokens['[MASK]']).sum().item()
                print(f"🔍 Sequence positions: {seq_mask_count}, Mask tokens: {mask_tokens_count}")
                
                # Show the modality type IDs to debug
                mod_type_ids = batch_modality_type_ids[0].cpu().tolist()
                sequence_positions = [i for i, mid in enumerate(mod_type_ids) if mid == 0]
                print(f"🔍 Sequence position indices: {sequence_positions[:10]}...{sequence_positions[-10:]} (showing first/last 10)")
                print(f"🔍 Total sequence positions found: {len(sequence_positions)}")
            
            # Start with fully masked sequence (all [MASK] tokens in sequence positions)
            noisy_tokens = batch_input_ids.clone()
            mask_token_id = self.special_tokens['[MASK]']
            
            # If we're doing infilling, keep the non-masked parts
            if sequence is not None and '*' in sequence:
                # Keep original tokens where sequence doesn't have masks
                pass  # The encoding already handles this
            else:
                # For full generation, start with all masks in sequence positions
                noisy_tokens[sequence_mask] = mask_token_id
            
            # Diffusion sampling loop
            timesteps = torch.linspace(num_inference_steps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device)
            
            for i, t in enumerate(timesteps):
                # Current timestep for all samples in batch
                t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                
                # Model forward pass
                with torch.no_grad():
                    model_logits = self.model(noisy_tokens, attention_mask=batch_attention_mask)['logits']
                    
                    # Apply vocabulary constraints
                    masked_logits = self.apply_modality_vocab_mask(model_logits, batch_modality_type_ids)
                    
                    # Apply temperature
                    if temperature != 1.0:
                        masked_logits = masked_logits / temperature
                    
                    # Apply top-k and top-p filtering if specified
                    if top_k is not None or top_p is not None:
                        for b in range(batch_size):
                            for s in range(masked_logits.shape[1]):
                                if sequence_mask[b, s]:
                                    logits_pos = masked_logits[b, s]
                                    
                                    if top_k is not None:
                                        top_k_logits, _ = torch.topk(logits_pos, min(top_k, logits_pos.size(-1)))
                                        logits_pos[logits_pos < top_k_logits[-1]] = -float('inf')
                                
                                if top_p is not None:
                                    sorted_logits, sorted_indices = torch.sort(logits_pos, descending=True)
                                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                                    sorted_indices_to_remove = cumulative_probs > top_p
                                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                    sorted_indices_to_remove[..., 0] = 0
                                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                                    logits_pos[indices_to_remove] = -float('inf')
                                
                                masked_logits[b, s] = logits_pos
                    
                    # Use the noise scheduler's remove_noise method
                    if i < len(timesteps) - 1:  # Not the last step
                        denoised_tokens = self.noise_scheduler.remove_noise(
                            noisy_tokens, t_batch, masked_logits, sequence_mask
                        )
                        noisy_tokens = denoised_tokens
                    else:
                        # Final step - sample directly from logits
                        for b in range(batch_size):
                            seq_positions = sequence_mask[b]
                            if seq_positions.sum() == 0:
                                continue
                            
                            seq_indices = torch.where(seq_positions)[0]
                            seq_logits = masked_logits[b, seq_indices]
                            seq_probs = F.softmax(seq_logits, dim=-1)
                            sampled_tokens = torch.multinomial(seq_probs, num_samples=1).squeeze(-1)
                            noisy_tokens[b, seq_indices] = sampled_tokens
            
            # Decode generated sequences
            for b in range(batch_size):
                generated_sequence = self.decode_sequence(noisy_tokens[b], batch_modality_type_ids[b])
                
                # Debug: Show final token counts for the first sequence
                if batch_start == 0 and b == 0:
                    final_seq_positions = (batch_modality_type_ids[b] == 0).sum().item()
                    print(f"🔍 Final sequence positions in output: {final_seq_positions}")
                    print(f"🔍 Generated sequence length: {len(generated_sequence)}")
                
                result = {
                    'sequence': generated_sequence,
                    'length': len(generated_sequence),
                    'secondary_structure': secondary_structure,
                    'consensus': consensus,
                    'go_terms': go_terms,
                    'generation_params': {
                        'num_inference_steps': num_inference_steps,
                        'temperature': temperature,
                        'top_k': top_k,
                        'top_p': top_p
                    }
                }
                results.append(result)
        
        return results[:num_sequences]  # Return exactly the requested number


class ConstrainedRNASequenceGenerator(RNASequenceGenerator):
    """
    RNA sequence generator with structure-based base-pairing constraints.
    """

    def __init__(self, config_path: str, checkpoint_path: str, device: str = "auto",
                 constraint_set: str = "canonical+sheared"):
        super().__init__(config_path, checkpoint_path, device)

        self.constraint_set_name = constraint_set
        self.allowed_pairs = get_predefined_constraint_set(constraint_set)

        # Lightweight vocab wrapper for constraint checker
        class VocabWrapper:
            def __init__(self, special_tokens, modality_offsets, modality_vocabs, vocab_size):
                self.special_tokens = special_tokens
                self.modality_offsets = modality_offsets
                self.modality_vocabs = modality_vocabs
                self.vocab_size = vocab_size

            def get_global_token_id(self, token: str, modality: str) -> int:
                if token in self.special_tokens:
                    return self.special_tokens[token]
                if modality not in self.modality_vocabs:
                    return -1
                tokens = self.modality_vocabs[modality]
                if token in tokens:
                    local_id = tokens.index(token)
                    return self.modality_offsets[modality] + local_id
                return -1

        self.constraint_generator = StructureConstrainedGenerator(
            vocabulary=VocabWrapper(
                self.special_tokens,
                self.modality_offsets,
                self.modality_vocabs,
                self.vocab_size
            ),
            allowed_pairs=self.allowed_pairs
        )

        logger.info("Using constraint set: %s", constraint_set)
        logger.info("Allowed base pairs: %s", self.allowed_pairs)

    def apply_structure_constraints(self,
                                    logits: torch.Tensor,
                                    current_tokens: torch.Tensor,
                                    modality_type_ids: torch.Tensor,
                                    secondary_structure: str) -> torch.Tensor:
        """
        Apply base-pairing constraints to logits based on secondary structure.
        """
        if secondary_structure is None:
            return logits

        pair_positions = self.constraint_generator.parse_structure(secondary_structure)
        if not pair_positions:
            return logits

        batch_size = logits.shape[0]
        constrained_logits = logits.clone()

        for b in range(batch_size):
            seq_positions = (modality_type_ids[b] == 0).nonzero(as_tuple=True)[0]
            if len(seq_positions) != len(secondary_structure):
                logger.warning(
                    "Structure length (%d) doesn't match sequence positions (%d). Skipping constraints.",
                    len(secondary_structure), len(seq_positions)
                )
                continue

            for struct_pos in range(len(secondary_structure)):
                if struct_pos not in pair_positions:
                    continue

                paired_struct_pos = pair_positions[struct_pos]
                if paired_struct_pos >= struct_pos:
                    continue

                token_pos = seq_positions[struct_pos].item()
                paired_token_pos = seq_positions[paired_struct_pos].item()

                paired_token_id = current_tokens[b, paired_token_pos].item()
                paired_base = self._token_id_to_base(paired_token_id)

                if paired_base is None or paired_base == '[MASK]':
                    continue

                compatible_bases = self.constraint_generator.get_compatible_bases(paired_base)
                if not compatible_bases:
                    continue

                vocab_mask = torch.ones(self.vocab_size, dtype=torch.bool, device=logits.device)
                for base in ['A', 'U', 'G', 'C']:
                    base_token_id = self._base_to_token_id(base)
                    if base_token_id >= 0:
                        vocab_mask[base_token_id] = base in compatible_bases

                constrained_logits[b, token_pos][~vocab_mask] = -float('inf')

        return constrained_logits

    def _token_id_to_base(self, token_id: int) -> str:
        """Convert token ID to base character."""
        for modality, offset in self.modality_offsets.items():
            vocab = self.modality_vocabs[modality]
            if offset <= token_id < offset + len(vocab):
                local_id = token_id - offset
                return vocab[local_id]

        for token, tid in self.special_tokens.items():
            if tid == token_id:
                return token

        return None

    def _base_to_token_id(self, base: str) -> int:
        """Convert base character to token ID."""
        return self.constraint_generator.vocab.get_global_token_id(base, 'sequence')

    def generate_sequences(self,
                           num_sequences: int = 1,
                           length: int = None,
                           sequence: str = None,
                           secondary_structure: str = None,
                           consensus: str = None,
                           go_terms: Union[str, List[str]] = None,
                           num_inference_steps: int = None,
                           temperature: float = 1.0,
                           top_k: int = None,
                           top_p: float = None) -> List[Dict[str, str]]:
        """
        Generate RNA sequences with structure constraints.
        """
        if isinstance(go_terms, str):
            go_terms = [term.strip() for term in go_terms.split(',') if term.strip()]

        if secondary_structure and length is None:
            length = len(secondary_structure)
            logger.info("Auto-detected length %d from secondary structure", length)

        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.num_timesteps

        if secondary_structure:
            logger.info("Generating %d sequences with %s constraints", num_sequences, self.constraint_set_name)
        else:
            logger.info("Generating %d sequences (no structure provided, constraints not applied)", num_sequences)

        results: List[Dict[str, str]] = []
        target_length = length if length is not None else (len(secondary_structure) if secondary_structure else None)
        max_attempts = num_sequences * 10
        attempts = 0

        while len(results) < num_sequences:
            attempts += 1
            if attempts > max_attempts:
                msg = (
                    f"Exceeded max attempts ({max_attempts}) while trying to generate "
                    f"{num_sequences} sequences with target length {target_length} "
                    f"and constraints={secondary_structure is not None}. "
                    "Likely due to repeated length/constraint mismatches."
                )
                logger.error(msg)
                raise RuntimeError(msg)

            batch_size = min(8, num_sequences - len(results))

            batch_inputs = []
            for _ in range(batch_size):
                encoded = self.encode_sequence_input(
                    sequence=sequence,
                    secondary_structure=secondary_structure,
                    consensus=consensus,
                    go_terms=go_terms,
                    length=length
                )
                batch_inputs.append(encoded)

            batch_input_ids = torch.cat([inp['input_ids'] for inp in batch_inputs], dim=0)
            batch_attention_mask = torch.cat([inp['attention_mask'] for inp in batch_inputs], dim=0)
            batch_modality_type_ids = torch.cat([inp['modality_type_ids'] for inp in batch_inputs], dim=0)

            sequence_mask = self.create_sequence_mask(batch_modality_type_ids)

            noisy_tokens = batch_input_ids.clone()
            mask_token_id = self.special_tokens['[MASK]']

            if sequence is not None and '*' in sequence:
                pass
            else:
                noisy_tokens[sequence_mask] = mask_token_id

            timesteps = torch.linspace(num_inference_steps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device)

            for i, t in enumerate(timesteps):
                t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

                with torch.no_grad():
                    model_logits = self.model(noisy_tokens, attention_mask=batch_attention_mask)['logits']
                    masked_logits = self.apply_modality_vocab_mask(model_logits, batch_modality_type_ids)

                    if secondary_structure is not None:
                        masked_logits = self.apply_structure_constraints(
                            masked_logits,
                            noisy_tokens,
                            batch_modality_type_ids,
                            secondary_structure
                        )

                    if temperature != 1.0:
                        masked_logits = masked_logits / temperature

                    if top_k is not None or top_p is not None:
                        for b in range(batch_size):
                            for s in range(masked_logits.shape[1]):
                                if sequence_mask[b, s]:
                                    logits_pos = masked_logits[b, s]

                                    if top_k is not None:
                                        top_k_logits, _ = torch.topk(logits_pos, min(top_k, logits_pos.size(-1)))
                                        logits_pos[logits_pos < top_k_logits[-1]] = -float('inf')

                                    if top_p is not None:
                                        sorted_logits, sorted_indices = torch.sort(logits_pos, descending=True)
                                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                                        sorted_indices_to_remove = cumulative_probs > top_p
                                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                        sorted_indices_to_remove[..., 0] = 0
                                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                                        logits_pos[indices_to_remove] = -float('inf')

                                    masked_logits[b, s] = logits_pos

                    if i < len(timesteps) - 1:
                        denoised_tokens = self.noise_scheduler.remove_noise(
                            noisy_tokens, t_batch, masked_logits, sequence_mask
                        )
                        noisy_tokens = denoised_tokens
                    else:
                        for b in range(batch_size):
                            seq_positions = sequence_mask[b]
                            if seq_positions.sum() == 0:
                                continue

                            seq_indices = torch.where(seq_positions)[0]
                            seq_logits = masked_logits[b, seq_indices]
                            seq_probs = F.softmax(seq_logits, dim=-1)
                            sampled_tokens = torch.multinomial(seq_probs, num_samples=1).squeeze(-1)
                            noisy_tokens[b, seq_indices] = sampled_tokens

            for b in range(batch_size):
                generated_sequence = self.decode_sequence(noisy_tokens[b], batch_modality_type_ids[b])

                if secondary_structure is not None:
                    if len(generated_sequence) < len(secondary_structure):
                        logger.warning(
                            "Generated sequence shorter than structure (seq=%d, struct=%d); retrying...",
                            len(generated_sequence), len(secondary_structure)
                        )
                        continue  # retry by skipping append

                    is_valid, violations = self.constraint_generator.check_constraints(
                        generated_sequence, secondary_structure
                    )
                    satisfaction = self.constraint_generator.calculate_constraint_satisfaction(
                        generated_sequence, secondary_structure
                    )
                else:
                    is_valid = True
                    violations = []
                    satisfaction = 1.0

                if target_length and len(generated_sequence) != target_length:
                    logger.warning(
                        "Generated sequence length %d != target %d; retrying...",
                        len(generated_sequence), target_length
                    )
                    continue  # retry by not appending

                result = {
                    'sequence': generated_sequence,
                    'length': len(generated_sequence),
                    'secondary_structure': secondary_structure,
                    'consensus': consensus,
                    'go_terms': go_terms,
                    'constraint_satisfaction': satisfaction,
                    'constraint_valid': is_valid,
                    'constraint_violations': violations if not is_valid else [],
                    'constraint_set': self.constraint_set_name,
                    'generation_params': {
                        'num_inference_steps': num_inference_steps,
                        'temperature': temperature,
                        'top_k': top_k,
                        'top_p': top_p
                    }
                }
                results.append(result)

        return results[:num_sequences]


def main():
    parser = argparse.ArgumentParser(
        description='Generate RNA sequences using trained diffusion model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  
  # Generate 5 sequences of length 100
  python %(prog)s --config configs/cluster_training.yaml --checkpoint outputs/model.pt --num_sequences 5 --length 100
  
  # Generate with secondary structure (length auto-detected)
  python %(prog)s --config configs/cluster_training.yaml --checkpoint outputs/model.pt --secondary_structure "((((....))))"
  
  # Generate with multiple conditioning modalities
  python %(prog)s --config configs/cluster_training.yaml --checkpoint outputs/model.pt --secondary_structure ":::<<<___>>>" --consensus "AUGcaugc" --go_terms "GO:0075523"
  
  # Infill masked sequence
  python %(prog)s --config configs/cluster_training.yaml --checkpoint outputs/model.pt --masked_sequence "AUG***CG"
  
  # Generate with custom sampling parameters
  python %(prog)s --config configs/cluster_training.yaml --checkpoint outputs/model.pt --length 50 --temperature 0.8 --top_k 10
  
  # Save to FASTA file
  python %(prog)s --config configs/cluster_training.yaml --checkpoint outputs/model.pt --num_sequences 10 --length 100 --fasta_output sequences.fasta
  
  # Free generation with variable lengths
  python %(prog)s --config configs/cluster_training.yaml --checkpoint outputs/model.pt --num_sequences 10 --length_range 50 200

LENGTH DETERMINATION (priority order):
  1. --masked_sequence length (highest priority)
  2. --secondary_structure length  
  3. --consensus length
  4. --length parameter
  5. --length_range (random sampling, lowest priority)
  
MODALITY FORMATS:
  --secondary_structure: Use symbols like ().<>[]{}:~-_,
  --consensus: Use ACGU (uppercase) or acgu (lowercase), dots, tildes
  --go_terms: Use format "GO:0123456" or comma-separated "GO:0123456,GO:0789012"
  --masked_sequence: Use * for positions to infill, e.g., "AUG***CG"
  
Maximum sequence length: 640 nucleotides
        """)
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to training config file')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint (e.g., outputs/model.pt or absolute path)')
    
    # Generation arguments
    parser.add_argument('--num_sequences', type=int, default=1, 
                       help='Number of sequences to generate (default: 1)')
    parser.add_argument('--length', type=int, 
                       help='Desired sequence length (1-640). Auto-determined from conditioning modalities if not provided')
    parser.add_argument('--length_range', nargs=2, type=int, metavar=('MIN', 'MAX'),
                       help='Generate sequences with random lengths between MIN and MAX (e.g., --length_range 50 150)')
    parser.add_argument('--masked_sequence', type=str, 
                       help='Sequence with * for infilling (e.g., "AUG***CG"). Length auto-detected')
    
    # Conditioning arguments
    parser.add_argument('--secondary_structure', type=str, 
                       help='Secondary structure constraint using ().<>[]{}:~-_, symbols. Length auto-detected')
    parser.add_argument('--consensus', type=str, 
                       help='Consensus sequence constraint using ACGU/acgu nucleotides. Length auto-detected')
    parser.add_argument('--go_terms', type=str, 
                       help='GO terms (e.g., "GO:0075523" or "GO:0003676,GO:0005515")')
    
    # Sampling arguments
    parser.add_argument('--num_inference_steps', type=int, 
                       help='Number of denoising steps (default: use model\'s configured timesteps)')
    parser.add_argument('--temperature', type=float, default=1.0, 
                       help='Sampling temperature for randomness (default: 1.0, lower=less random)')
    parser.add_argument('--top_k', type=int, 
                       help='Top-k sampling: only consider k most likely tokens')
    parser.add_argument('--top_p', type=float, 
                       help='Nucleus sampling: only consider tokens with cumulative probability <= p')
    parser.add_argument('--constraint_set', type=str,
                       choices=['strict', 'canonical', 'canonical+sheared', 'canonical+common', 'permissive'],
                       help='Apply base-pair constraints (optional; defaults to unconstrained)')
    
    # Output arguments
    parser.add_argument('--output', type=str, 
                       help='Output file path in JSON format (optional)')
    parser.add_argument('--fasta_output', type=str, 
                       help='Output file path in FASTA format (optional)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], 
                       help='Device to use for generation (default: auto)')
    
    args = parser.parse_args()
    
    # Auto-determine length from conditioning modalities or validate arguments
    auto_length = None
    
    # Try to determine length from masked sequence
    if args.masked_sequence:
        auto_length = len(args.masked_sequence)
        logger.info("Auto-determined length from masked sequence: %d", auto_length)
    
    # Try to determine length from secondary structure
    elif args.secondary_structure:
        auto_length = len(args.secondary_structure)
        logger.info("Auto-determined length from secondary structure: %d", auto_length)
    
    # Try to determine length from consensus sequence
    elif args.consensus:
        auto_length = len(args.consensus)
        logger.info("Auto-determined length from consensus sequence: %d", auto_length)
    
    # Use provided length or auto-determined length
    final_length = args.length if args.length else auto_length
    
    # Handle length range for free generation
    use_length_range = False
    if not final_length and args.length_range:
        min_len, max_len = args.length_range
        if min_len < 1 or max_len > 640 or min_len > max_len:
            parser.error("Invalid length range. Must satisfy: 1 <= MIN <= MAX <= 640")
        use_length_range = True
        logger.info("Will generate sequences with random lengths between %d and %d", min_len, max_len)
    elif not final_length:
        parser.error("Must provide either --length, --length_range, --masked_sequence, --secondary_structure, or --consensus to determine sequence length")
    
    if final_length and final_length > 640:
        parser.error("Maximum sequence length is 640 nucleotides")
    
    # Initialize generator (constrained if requested)
    logger.info("Initializing RNA sequence generator...")
    if args.constraint_set:
        generator = ConstrainedRNASequenceGenerator(
            args.config, args.checkpoint, device=args.device, constraint_set=args.constraint_set
        )
    else:
        generator = RNASequenceGenerator(args.config, args.checkpoint, device=args.device)
    
    # Parse generation parameters
    go_terms = None
    if args.go_terms:
        go_terms = [term.strip() for term in args.go_terms.split(',')]
    
    # Generate sequences
    logger.info("Generating %d RNA sequences...", args.num_sequences)
    
    if use_length_range:
        # Generate sequences one at a time with random lengths
        results = []
        min_len, max_len = args.length_range
        
        for i in range(args.num_sequences):
            # Sample random length for this sequence
            seq_length = random.randint(min_len, max_len)
            logger.info("Sequence %d: using length %d", i+1, seq_length)
            
            # Generate single sequence with this length
            seq_results = generator.generate_sequences(
                num_sequences=1,
                length=seq_length,
                sequence=args.masked_sequence,
                secondary_structure=args.secondary_structure,
                consensus=args.consensus,
                go_terms=go_terms,
                num_inference_steps=args.num_inference_steps,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            results.extend(seq_results)
    else:
        # Generate all sequences with fixed length
        results = generator.generate_sequences(
            num_sequences=args.num_sequences,
            length=final_length,
            sequence=args.masked_sequence,
            secondary_structure=args.secondary_structure,
            consensus=args.consensus,
            go_terms=go_terms,
            num_inference_steps=args.num_inference_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
    
    # Display results
    print("\n" + "="*80)
    print("🧬 GENERATED RNA SEQUENCES")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\nSequence {i}:")
        print(f"  RNA:      {result['sequence']}")
        print(f"  Length:   {result['length']}")
        
        if result['secondary_structure']:
            print(f"  Structure: {result['secondary_structure']}")
        if result['consensus']:
            print(f"  Consensus: {result['consensus']}")
        if result['go_terms']:
            print(f"  GO Terms:  {', '.join(result['go_terms'])}")
        if 'constraint_set' in result and result.get('constraint_set'):
            sat = result.get('constraint_satisfaction', None)
            valid = result.get('constraint_valid', None)
            if sat is not None:
                print(f"  Constraint Satisfaction: {sat:.1%}")
            if valid is not None:
                print(f"  Constraint Valid: {'✅' if valid else '❌'}")
            print(f"  Constraint Set: {result['constraint_set']}")
    
    print(f"\n📊 Generation completed: {len(results)} sequences generated")
    
    # Save to file if requested
    if args.output:
        output_data = {
            'generated_sequences': results,
            'generation_metadata': {
                'config_path': args.config,
                'checkpoint_path': args.checkpoint,
                'generation_timestamp': datetime.now().isoformat(),
                'device': str(generator.device)
            }
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info("Results saved to %s", args.output)
    
    # Save to FASTA format if requested
    if args.fasta_output:
        with open(args.fasta_output, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results, 1):
                # Create FASTA header with metadata
                header_parts = [f"seq_{i}", f"length_{result['length']}"]
                
                # Add conditioning information to header
                if result['secondary_structure']:
                    header_parts.append("ss_provided")
                if result['consensus']:
                    header_parts.append("cons_provided")
                if result['go_terms']:
                    go_terms_str = ','.join(result['go_terms']) if isinstance(result['go_terms'], list) else result['go_terms']
                    header_parts.append(f"go_{go_terms_str}")
                
                # Add generation parameters
                gen_params = result['generation_params']
                if gen_params['temperature'] != 1.0:
                    header_parts.append(f"temp_{gen_params['temperature']}")
                if gen_params['top_k']:
                    header_parts.append(f"topk_{gen_params['top_k']}")
                if gen_params['top_p']:
                    header_parts.append(f"topp_{gen_params['top_p']}")
                
                header = f">{'|'.join(header_parts)}"
                
                # Write FASTA entry
                f.write(header + '\n')
                f.write(result['sequence'] + '\n')
        
        logger.info("FASTA sequences saved to %s", args.fasta_output)


if __name__ == "__main__":
    main()