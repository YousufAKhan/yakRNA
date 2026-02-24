"""
Discrete Diffusion Components for RNA Multimodal Model

This module implements:
1. Discrete noise scheduler for token-level corruption
2. Conditional data preparation with modality dropout
3. Discrete diffusion loss function
4. Forward and reverse diffusion processes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DiscreteNoiseScheduler:
    """
    Discrete noise scheduler for token-level diffusion.
    
    Gradually corrupts sequence tokens over T timesteps using discrete noise.
    Only applies corruption to the sequence modality, preserving conditioning modalities.
    """
    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_schedule: str = "linear",
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 vocabulary: Dict = None,
                 masking_rates: Dict = None,
                 cumulative_masking: Dict = None):
        """
        Initialize discrete noise scheduler.
        
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_schedule: Noise schedule type ("linear", "cosine", "sigmoid")
            beta_start: Starting noise level
            beta_end: Ending noise level
            vocabulary: Unified vocabulary dictionary for token sampling
            masking_rates: Optional dict with explicit masking rate control (legacy)
            cumulative_masking: Optional dict with cumulative masking control
                - target_percentage: Total percentage to mask by t=max (0.95 = 95%)
                - schedule_type: Distribution type ("linear", "cosine", "exponential")
                - initial_rate: Initial masking rate at t=0 (default: 0.0)
        """
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.vocabulary = vocabulary
        
        # Handle cumulative masking control (new approach)
        self.use_cumulative_masking = cumulative_masking is not None
        if self.use_cumulative_masking:
            self.target_percentage = cumulative_masking.get('target_percentage', 0.95)
            self.initial_rate = cumulative_masking.get('initial_rate', 0.0)
            self.masking_schedule_type = cumulative_masking.get('schedule_type', 'linear')
            
            # Precompute masking rates for all timesteps
            self.masking_rates = self._compute_cumulative_masking_schedule()
            
            logger.info(f"Using cumulative masking: target={self.target_percentage:.2f}, initial={self.initial_rate:.2f}, schedule={self.masking_schedule_type}")
        
        # Handle explicit masking rate control (legacy)
        elif masking_rates is not None:
            self.use_explicit_masking = True
            self.t0_rate = masking_rates.get('t0_rate', 0.0)
            self.tmax_rate = masking_rates.get('tmax_rate', 1.0)
            self.masking_schedule_type = masking_rates.get('schedule_type', 'linear')
            
            # Precompute masking rates for all timesteps
            self.masking_rates = self._compute_masking_schedule()
            
            logger.info(f"Using explicit masking rates: t0={self.t0_rate:.2f}, tmax={self.tmax_rate:.2f}, schedule={self.masking_schedule_type}")
        else:
            self.use_explicit_masking = False
            logger.info("Using beta-based masking rates (traditional diffusion)")
        
        # Compute noise schedule
        self.betas = self._compute_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute values for forward and reverse processes
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        logger.info(f"Initialized discrete noise scheduler with {num_timesteps} timesteps")
        logger.info(f"Beta schedule: {beta_schedule}, range: [{beta_start:.6f}, {beta_end:.6f}]")
    
    def _compute_cumulative_masking_schedule(self) -> torch.Tensor:
        """
        Compute progressive masking rates that increase from 0% to target_percentage.
        
        This creates a progressive corruption schedule where:
        - t=0: 0% corruption
        - t=max: target_percentage% corruption
        - Intermediate timesteps: interpolated corruption rates
        """
        # Create progressive masking rates that increase from 0 to target_percentage
        if self.masking_schedule_type == "linear":
            # Linear progression from 0 to target_percentage
            masking_rates = torch.linspace(0.0, self.target_percentage, self.num_timesteps)
        elif self.masking_schedule_type == "cosine":
            # Cosine progression: slower at start, faster in middle
            t = torch.linspace(0, 1, self.num_timesteps)
            masking_rates = self.target_percentage * (1 - torch.cos(t * torch.pi)) / 2
        elif self.masking_schedule_type == "exponential":
            # Exponential progression: very slow at start, very fast at end
            t = torch.linspace(0, 1, self.num_timesteps)
            masking_rates = self.target_percentage * (torch.exp(t) - 1) / (torch.exp(torch.tensor(1.0)) - 1)
        else:
            raise ValueError(f"Unknown masking schedule type: {self.masking_schedule_type}")
        
        # Add initial rate if specified (overrides the computed rate at t=0)
        if self.initial_rate > 0:
            masking_rates[0] = self.initial_rate
        
        # Ensure rates are within valid range [0, 1]
        masking_rates = torch.clamp(masking_rates, 0.0, 1.0)
        
        # Log the schedule for verification
        logger.info(f"Progressive masking schedule: t=0: {masking_rates[0]:.3f}, t=max: {masking_rates[-1]:.3f}")
        logger.info(f"  Schedule type: {self.masking_schedule_type}, Target: {self.target_percentage:.3f}")
        
        return masking_rates
    
    def _compute_masking_schedule(self) -> torch.Tensor:
        """Compute explicit masking rates for all timesteps."""
        if self.masking_schedule_type == "linear":
            return torch.linspace(self.t0_rate, self.tmax_rate, self.num_timesteps)
        elif self.masking_schedule_type == "cosine":
            # Cosine interpolation for smoother transitions
            t = torch.linspace(0, 1, self.num_timesteps)
            return self.t0_rate + (self.tmax_rate - self.t0_rate) * (1 - torch.cos(t * torch.pi)) / 2
        elif self.masking_schedule_type == "exponential":
            # Exponential interpolation for more aggressive early masking
            t = torch.linspace(0, 1, self.num_timesteps)
            return self.t0_rate + (self.tmax_rate - self.t0_rate) * (torch.exp(t) - 1) / (torch.exp(torch.tensor(1.0)) - 1)
        else:
            raise ValueError(f"Unknown masking schedule type: {self.masking_schedule_type}")
    
    def _compute_beta_schedule(self) -> torch.Tensor:
        """Compute beta schedule based on specified type."""
        if self.beta_schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.beta_schedule == "cosine":
            return self._cosine_beta_schedule()
        elif self.beta_schedule == "sigmoid":
            return self._sigmoid_beta_schedule()
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine beta schedule."""
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _sigmoid_beta_schedule(self) -> torch.Tensor:
        """Sigmoid beta schedule."""
        betas = torch.linspace(-6, 6, self.num_timesteps)
        return torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start
    
    def add_noise(self, 
                  original_tokens: torch.Tensor,
                  timesteps: torch.Tensor,
                  sequence_mask: torch.Tensor,
                  rng_seeds_per_sample: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add discrete noise to sequence tokens using DPLM-inspired approach.
        
        Args:
            original_tokens: Original token IDs (batch_size, seq_len)
            timesteps: Timestep for each sample in batch (batch_size,)
            sequence_mask: Boolean mask indicating sequence modality positions (batch_size, seq_len)
            
        Returns:
            noisy_tokens: Corrupted token IDs
            noise_info: Dictionary containing noise information for loss computation
        """
        batch_size, seq_len = original_tokens.shape
        device = original_tokens.device
        
        # Ensure timesteps are on correct device
        timesteps = timesteps.to(device)
        
        # Get noise schedule values for current timesteps
        # Ensure alphas_cumprod is on the same device as timesteps
        alphas_cumprod_device = self.alphas_cumprod.to(device)
        alphas_cumprod_t = alphas_cumprod_device[timesteps].view(batch_size, 1)
        
        # Initialize noisy tokens as original tokens
        noisy_tokens = original_tokens.clone()
        
        # Track noise information for loss computation
        noise_mask = torch.zeros_like(original_tokens, dtype=torch.bool)
        corruption_probs = torch.zeros_like(original_tokens, dtype=torch.float)
        
        # 🔍 DEBUG: Add debugging for noise addition
        debug_info = {
            'total_sequence_positions': 0,
            'total_corrupted_positions': 0,
            'corruption_rates': [],
            'sequence_positions_per_sample': []
        }
        
        # Apply noise only to sequence modality positions
        for b in range(batch_size):
            seq_positions = sequence_mask[b]
            if seq_positions.sum() == 0:
                continue
                
            # Get corruption probability for this timestep
            if self.use_cumulative_masking:
                # Use cumulative masking rates
                # Ensure masking_rates is on the same device as timesteps
                masking_rates_device = self.masking_rates.to(device)
                corruption_prob = masking_rates_device[timesteps[b]].item()
            elif self.use_explicit_masking:
                # Use explicit masking rates
                # Ensure masking_rates is on the same device as timesteps
                masking_rates_device = self.masking_rates.to(device)
                corruption_prob = masking_rates_device[timesteps[b]].item()
            else:
                # Use traditional beta-based corruption
                corruption_prob = 1.0 - alphas_cumprod_t[b].item()
            
            # Get sequence token positions
            seq_indices = torch.where(seq_positions)[0]
            
            # 🔍 DEBUG: Track sequence positions
            debug_info['total_sequence_positions'] += len(seq_indices)
            debug_info['sequence_positions_per_sample'].append(seq_indices.tolist())
            debug_info['corruption_rates'].append(corruption_prob)
            
            # For each sequence position, decide whether to corrupt based on probability
            if rng_seeds_per_sample is not None:
                gen = torch.Generator(device=device)
                gen.manual_seed(int(rng_seeds_per_sample[b].item()))
                corruption_decision = torch.rand(len(seq_indices), device=device, generator=gen) < corruption_prob
            else:
                corruption_decision = torch.rand(len(seq_indices), device=device) < corruption_prob
            positions_to_corrupt = seq_indices[corruption_decision]
            
            # 🔍 DEBUG: Track corruption
            debug_info['total_corrupted_positions'] += len(positions_to_corrupt)
            
            if len(positions_to_corrupt) > 0:
                # Corrupt selected positions with [MASK] tokens (DPLM-style)
                # No random nucleotide substitution - only mask tokens
                mask_token_id = self.vocabulary['special_tokens']['[MASK]']
                noisy_tokens[b, positions_to_corrupt] = mask_token_id
                
                # Mark corrupted positions
                noise_mask[b, positions_to_corrupt] = True
                corruption_probs[b, positions_to_corrupt] = corruption_prob
        
        noise_info = {
            'noise_mask': noise_mask,
            'corruption_probs': corruption_probs,
            'alphas_cumprod_t': alphas_cumprod_t,
            'timesteps': timesteps,
            'debug_info': debug_info  # Add debug info
        }
        
        # Add corruption rate to noise_info for logging
        if self.use_cumulative_masking or self.use_explicit_masking:
            # Calculate average corruption rate across the batch
            # Ensure masking_rates is on the same device as timesteps
            masking_rates_device = self.masking_rates.to(device)
            avg_corruption_rate = torch.mean(torch.stack([
                masking_rates_device[timesteps[b]] for b in range(batch_size)
            ])).item()
        else:
            # Calculate average corruption rate from alphas_cumprod
            avg_corruption_rate = torch.mean(1.0 - alphas_cumprod_t).item()
        
        noise_info['corruption_rate'] = avg_corruption_rate
        
        return noisy_tokens, noise_info
    
    def remove_noise(self, 
                    noisy_tokens: torch.Tensor,
                    timesteps: torch.Tensor,
                    model_predictions: torch.Tensor,
                    sequence_mask: torch.Tensor,
                    guidance_scale: float = 1.0) -> torch.Tensor:
        """
        Remove noise from tokens using model predictions with DPLM-inspired approach.
        
        Args:
            noisy_tokens: Current noisy token IDs
            timesteps: Current timesteps
            model_predictions: Model predictions for original tokens (batch_size, seq_len, vocab_size)
            sequence_mask: Boolean mask indicating sequence modality positions
            guidance_scale: Scale for classifier-free guidance (1.0 = no guidance)
            
        Returns:
            denoised_tokens: Denoised token IDs
        """
        batch_size, seq_len = noisy_tokens.shape
        device = noisy_tokens.device
        
        # Ensure timesteps are on correct device
        timesteps = timesteps.to(device)
        
        # Get noise schedule values
        # Ensure alphas_cumprod tensors are on the same device as timesteps
        alphas_cumprod_device = self.alphas_cumprod.to(device)
        alphas_cumprod_prev_device = self.alphas_cumprod_prev.to(device)
        alphas_cumprod_t = alphas_cumprod_device[timesteps].view(batch_size, 1)
        alphas_cumprod_prev_t = alphas_cumprod_prev_device[timesteps].view(batch_size, 1)
        
        # Initialize denoised tokens
        denoised_tokens = noisy_tokens.clone()
        
        # Apply denoising only to sequence modality positions
        for b in range(batch_size):
            seq_positions = sequence_mask[b]
            if seq_positions.sum() == 0:
                continue
            
            # Get sequence positions
            seq_indices = torch.where(seq_positions)[0]
            
            # Get model predictions for sequence positions
            seq_logits = model_predictions[b, seq_indices]  # (num_seq_tokens, vocab_size)
            
            # Apply classifier-free guidance if enabled
            if guidance_scale != 1.0:
                # This would require unconditional predictions as well
                # For now, we'll use the conditional predictions
                pass
            
            # Sample from the predicted distribution
            # Use temperature sampling for diversity
            temperature = 1.0
            if temperature != 1.0:
                seq_logits = seq_logits / temperature
            
            # Sample tokens from the predicted distribution
            seq_probs = F.softmax(seq_logits, dim=-1)
            sampled_tokens = torch.multinomial(seq_probs, num_samples=1).squeeze(-1)
            
            # Update denoised tokens
            denoised_tokens[b, seq_indices] = sampled_tokens
        
        return denoised_tokens


class ConditionalDataPreparator:
    """
    Prepares data for conditional diffusion training with modality dropout.
    """
    
    def __init__(self, 
                 vocabulary: Dict,
                 modality_dropout_rates: Dict[str, float] = None,
                 target_modality: str = "sequence",
                 conditioning_modalities: List[str] = None):
        """
        Initialize conditional data preparator.
        
        Args:
            vocabulary: Unified vocabulary dictionary
            modality_dropout_rates: Dictionary mapping modality names to dropout probabilities
                                  (e.g., {'ss': 0.3, 'consensus': 0.2, 'go_terms': 0.1})
                                  If None, uses default rates for all modalities
            target_modality: Primary modality to generate (sequence)
            conditioning_modalities: Modalities to condition on
        """
        self.vocabulary = vocabulary
        self.target_modality = target_modality
        self.conditioning_modalities = conditioning_modalities or ["ss", "consensus", "go_terms"]
        self.dropped_token_id = vocabulary['special_tokens']['[DROPPED]']
        
        # Set default dropout rates if not provided
        if modality_dropout_rates is None:
            self.modality_dropout_rates = {
                'ss': 0.3,
                'consensus': 0.3, 
                'go_terms': 0.3
            }
        else:
            self.modality_dropout_rates = modality_dropout_rates
            # Ensure all conditioning modalities have dropout rates
            for mod in self.conditioning_modalities:
                if mod not in self.modality_dropout_rates:
                    self.modality_dropout_rates[mod] = 0.3  # Default rate
        
        # Get modality separator token IDs
        self.modality_separators = {
            'sequence': vocabulary['special_tokens']['[SEQ_START]'],
            'ss': vocabulary['special_tokens']['[STRUCT_START]'],
            'consensus': vocabulary['special_tokens']['[CONS_START]'],
            'go_terms': vocabulary['special_tokens']['[GO_START]']
        }
        
        logger.info(f"Initialized conditional data preparator")
        logger.info(f"Target modality: {target_modality}")
        logger.info(f"Conditioning modalities: {conditioning_modalities}")
        logger.info(f"Modality dropout rates: {self.modality_dropout_rates}")
    
    def prepare_conditional_batch(self, 
                                batch: Dict[str, torch.Tensor],
                                training: bool = True,
                                dropout_seed_base: int | None = None,
                                sample_keys: list | None = None) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for conditional diffusion training.
        
        Args:
            batch: Input batch with input_ids, attention_mask, etc.
            training: Whether in training mode (applies dropout)
            
        Returns:
            prepared_batch: Batch with conditional information and modality dropout
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        modality_type_ids = batch['modality_type_ids']
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get special token IDs for protection
        special_tokens = self.vocabulary['special_tokens']
        sep_token_id = special_tokens['[SEP]']
        
        # Create sequence mask (positions that belong to sequence modality)
        # Only include actual RNA nucleotides, not special tokens
        sequence_mask = (modality_type_ids == 0)  # 0 = sequence modality
        
        # Filter out special tokens from sequence mask
        # Get sequence vocabulary range
        seq_vocab = self.vocabulary['modality_vocabs']['sequence']
        seq_offset = self.vocabulary['modality_offsets']['sequence']
        
        # Create mask for actual sequence tokens (A, U, G, C, N) only
        actual_seq_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b in range(batch_size):
            for s in range(seq_len):
                if sequence_mask[b, s]:
                    token_id = input_ids[b, s].item()
                    # Only include tokens that are actual RNA nucleotides
                    if seq_offset <= token_id < seq_offset + len(seq_vocab):
                        actual_seq_mask[b, s] = True
        
        # Use the filtered sequence mask
        sequence_mask = actual_seq_mask
        
        # Create individual modality masks (ONLY for content tokens, not special tokens)
        modality_masks = {}
        for mod_name, mod_id in [('sequence', 0), ('ss', 1), ('consensus', 2), ('go_terms', 3)]:
            # Get modality vocabulary range
            mod_vocab = self.vocabulary['modality_vocabs'][mod_name]
            mod_offset = self.vocabulary['modality_offsets'][mod_name]
            
            # Create mask for actual content tokens only (exclude special tokens)
            content_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for b in range(batch_size):
                for s in range(seq_len):
                    if modality_type_ids[b, s] == mod_id:
                        token_id = input_ids[b, s].item()
                        # Only include tokens that are actual content tokens (not special tokens)
                        if mod_offset <= token_id < mod_offset + len(mod_vocab):
                            content_mask[b, s] = True
            
            modality_masks[mod_name] = content_mask
        
        # Create conditioning mask (positions that belong to conditioning modalities)
        conditioning_mask = torch.zeros_like(modality_type_ids, dtype=torch.bool)
        for mod_id, mod_name in enumerate(['sequence', 'ss', 'consensus', 'go_terms']):
            if mod_name in self.conditioning_modalities:
                conditioning_mask |= (modality_type_ids == mod_id)
        
        # Apply modality dropout during training
        if training:
            # Create modified input_ids with dropped modalities
            modified_input_ids = input_ids.clone()
            
            # Track which modalities were dropped for each sample
            dropped_modalities = []
            
            # Track dropout statistics
            total_content_tokens = 0
            go_terms_bounded_by_sep = 0
            
            for b in range(batch_size):
                # Optional deterministic seeding per sample
                if dropout_seed_base is not None and sample_keys is not None and b < len(sample_keys):
                    torch.manual_seed(int(dropout_seed_base) ^ int(sample_keys[b]))
                # For each sample, randomly decide which conditioning modalities to drop
                sample_dropped_mods = []
                
                for mod_name in self.conditioning_modalities:
                    # Apply individual dropout rate for each conditioning modality
                    dropout_rate = self.modality_dropout_rates.get(mod_name, 0.3)
                    if torch.rand(1, device=device).item() < dropout_rate:
                        # Drop this modality for this sample
                        # Use content-only mask (excludes special tokens)
                        content_mask = modality_masks[mod_name][b]
                        
                        # Count content tokens being dropped
                        content_tokens_dropped = content_mask.sum().item()
                        total_content_tokens += content_tokens_dropped
                        
                        # For GO terms, ensure we don't drop beyond [SEP] token
                        if mod_name == 'go_terms':
                            # Find [SEP] token position for this sample
                            sep_positions = (input_ids[b] == sep_token_id).nonzero(as_tuple=True)[0]
                            if len(sep_positions) > 0:
                                sep_pos = sep_positions[0].item()
                                # Only drop GO terms up to [SEP] token
                                original_content_mask = content_mask.clone()
                                content_mask = content_mask & (torch.arange(seq_len, device=device) < sep_pos)
                                go_terms_bounded_by_sep += 1
                                
                                # Track if we had to bound the dropout
                                if original_content_mask.sum() != content_mask.sum():
                                    go_terms_bounded_by_sep += 1
                        
                        # Apply dropout only to content tokens (not special tokens)
                        modified_input_ids[b, content_mask] = self.dropped_token_id
                        sample_dropped_mods.append(mod_name)
                
                dropped_modalities.append(sample_dropped_mods)
            
            # Log protection statistics (only if significant)
            if total_content_tokens > 0 and go_terms_bounded_by_sep > 0:
                logger.debug(f"GO terms bounded by [SEP]: {go_terms_bounded_by_sep} samples")
            
            # Create conditioning information
            conditioning_info = {
                'has_conditioning': torch.tensor([len(dropped) < len(self.conditioning_modalities) for dropped in dropped_modalities], 
                                               dtype=torch.bool, device=device),
                'sequence_mask': sequence_mask,
                'conditioning_mask': conditioning_mask,
                'dropout_mask': torch.tensor([len(dropped) > 0 for dropped in dropped_modalities], 
                                           dtype=torch.bool, device=device),
                'dropped_modalities': dropped_modalities
            }
            
            # Modality dropout applied (logging handled in main training loop)
        else:
            # No dropout during evaluation
            modified_input_ids = input_ids
            conditioning_info = {
                'has_conditioning': torch.ones(batch_size, dtype=torch.bool, device=device),
                'sequence_mask': sequence_mask,
                'conditioning_mask': conditioning_mask,
                'dropout_mask': torch.zeros(batch_size, dtype=torch.bool, device=device),
                'dropped_modalities': [[] for _ in range(batch_size)]
            }
        
        # Create prepared batch
        prepared_batch = {
            'input_ids': modified_input_ids,
            'attention_mask': attention_mask,
            'modality_type_ids': modality_type_ids,
            'original_input_ids': batch.get('original_input_ids', input_ids),
            'family_ids': batch.get('family_ids', []),
            'conditioning_info': conditioning_info
        }
        
        return prepared_batch 


class DiscreteDiffusionLoss(nn.Module):
    """
    Discrete diffusion loss function for sequence generation using ELBO-based approach.
    Inspired by DPLM's implementation for protein diffusion.
    """
    
    def __init__(self, 
                 vocabulary: Dict,
                 sequence_loss_weight: float = 1.0,
                 loss_type: str = "elbo"):
        """
        Initialize discrete diffusion loss.
        
        Args:
            vocabulary: Unified vocabulary dictionary
            sequence_loss_weight: Weight for sequence generation loss
            loss_type: Type of loss ("elbo", "simple", "hybrid")
        """
        super().__init__()
        self.vocabulary = vocabulary
        self.sequence_loss_weight = sequence_loss_weight
        self.loss_type = loss_type
        self.padding_label_id = -100
        
        # Get sequence modality token range
        seq_offset = vocabulary['modality_offsets']['sequence']
        seq_vocab_size = len(vocabulary['modality_vocabs']['sequence'])
        self.sequence_token_range = (seq_offset, seq_offset + seq_vocab_size)
        
        logger.info(f"Initialized discrete diffusion loss (type: {loss_type})")
        logger.info(f"Sequence loss weight: {sequence_loss_weight}")
    
    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor,
                timesteps: torch.Tensor,
                sequence_mask: torch.Tensor,
                conditioning_mask: torch.Tensor,
                noise_info: Dict = None) -> Dict[str, torch.Tensor]:
        """
        Compute discrete diffusion loss using ELBO-based approach.
        
        Args:
            predictions: Model predictions (batch_size, seq_len, vocab_size)
            targets: Target token IDs (batch_size, seq_len)
            timesteps: Diffusion timesteps (batch_size,)
            sequence_mask: Boolean mask for sequence modality positions
            conditioning_mask: Boolean mask for conditioning modality positions
            noise_info: Dictionary containing noise information from add_noise
            
        Returns:
            loss_dict: Dictionary containing various loss components
        """
        batch_size, seq_len, vocab_size = predictions.shape
        device = predictions.device
        
        # Initialize loss components
        sequence_loss = torch.tensor(0.0, device=device)
        total_loss = torch.tensor(0.0, device=device)
        
        if self.loss_type == "elbo" and noise_info is not None:
            # ELBO-based loss inspired by DPLM
            sequence_loss = self._compute_elbo_loss(predictions, targets, timesteps, 
                                                   sequence_mask, noise_info)
        elif self.loss_type == "cross-entropy":
            # Standard cross-entropy loss (same as MLM)
            sequence_loss = self._compute_simple_loss(predictions, targets, 
                                                     sequence_mask)
        else:
            # Simple cross-entropy loss (fallback)
            sequence_loss = self._compute_simple_loss(predictions, targets, 
                                                     sequence_mask)
        
        # Total loss
        total_loss = sequence_loss * self.sequence_loss_weight
        
        # Compute additional metrics
        with torch.no_grad():
            # Sequence accuracy
            if sequence_mask.sum() > 0:
                seq_pred_ids = predictions[sequence_mask].argmax(dim=-1)
                seq_accuracy = (seq_pred_ids == targets[sequence_mask]).float().mean()
            else:
                seq_accuracy = torch.tensor(0.0, device=device)
            

        
        return {
            'total_loss': total_loss,
            'sequence_loss': sequence_loss,
            'sequence_accuracy': seq_accuracy
        }
    
    def _compute_elbo_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                          timesteps: torch.Tensor, sequence_mask: torch.Tensor,
                          noise_info: Dict) -> torch.Tensor:
        """
        Compute ELBO-based loss for discrete diffusion.
        Inspired by DPLM's implementation.
        """
        device = predictions.device
        
        # Get noise information
        noise_mask = noise_info['noise_mask']
        corruption_probs = noise_info['corruption_probs']
        alphas_cumprod_t = noise_info['alphas_cumprod_t']
        
        # Focus on sequence modality positions
        seq_noise_mask = noise_mask & sequence_mask
        seq_targets = targets[seq_noise_mask]
        seq_predictions = predictions[seq_noise_mask]
        
        if seq_noise_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # Compute cross-entropy loss for corrupted positions
        ce_loss = F.cross_entropy(seq_predictions, seq_targets, 
                                 ignore_index=self.padding_label_id,
                                 reduction='none')
        
        # Apply timestep-dependent weighting (similar to DPLM)
        # Higher timesteps get higher weight for corrupted positions
        # We need to expand alphas_cumprod_t to match the sequence mask shape
        batch_size = alphas_cumprod_t.shape[0]
        expanded_alphas = alphas_cumprod_t.expand(-1, seq_noise_mask.shape[1])  # (batch_size, seq_len)
        timestep_weights = expanded_alphas[seq_noise_mask]
        
        # Weight the loss by corruption probability and timestep
        weighted_loss = ce_loss * timestep_weights
        
        return weighted_loss.mean()
    
    def _compute_simple_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                            sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute simple cross-entropy loss for sequence generation (ONLY sequence tokens).
        """
        if sequence_mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        seq_predictions = predictions[sequence_mask]
        seq_targets = targets[sequence_mask]
        
        return F.cross_entropy(seq_predictions, seq_targets,
                              ignore_index=self.padding_label_id,
                              reduction='mean')


def create_diffusion_components(config: Dict, vocabulary: Dict) -> Tuple[DiscreteNoiseScheduler, ConditionalDataPreparator, DiscreteDiffusionLoss]:
    """
    Create all diffusion components from configuration.
    
    Args:
        config: Configuration dictionary
        vocabulary: Unified vocabulary dictionary
        
    Returns:
        Tuple of (noise_scheduler, data_preparator, loss_function)
    """
    diffusion_config = config['training']['discrete_diffusion']
    
    # Create noise scheduler
    masking_rates = diffusion_config.get('masking_rates', None)
    cumulative_masking = diffusion_config.get('cumulative_masking', None)
    
    # Use default beta values if not specified (for backward compatibility)
    beta_schedule = diffusion_config.get('beta_schedule', 'linear')
    beta_start = diffusion_config.get('beta_start', 0.0001)
    beta_end = diffusion_config.get('beta_end', 0.02)
    
    noise_scheduler = DiscreteNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        vocabulary=vocabulary,
        masking_rates=masking_rates,
        cumulative_masking=cumulative_masking
    )
    
    # Create data preparator
    conditional_config = diffusion_config['conditional_generation']
    data_preparator = ConditionalDataPreparator(
        vocabulary=vocabulary,
        modality_dropout_rates=conditional_config.get('modality_dropout_rates', None),
        target_modality=conditional_config['target_modality'],
        conditioning_modalities=conditional_config['conditioning_modalities']
    )
    
    # Create loss function
    loss_weights = diffusion_config['loss_weights']
    loss_function = DiscreteDiffusionLoss(
        vocabulary=vocabulary,
        sequence_loss_weight=loss_weights['sequence'],
        loss_type=diffusion_config.get('loss_type', 'elbo')
    )
    
    return noise_scheduler, data_preparator, loss_function 


def diffusion_collate_fn(batch, config, vocabulary, data_preparator=None, training=True):
    """
    Collate function for diffusion training. NO MLM masking - only applies modality dropout for conditional generation.
    Args:
        batch: List of samples (as in standard collate_fn)
        config: Configuration dictionary
        vocabulary: Unified vocabulary dictionary
        data_preparator: Optional, instance of ConditionalDataPreparator (if None, will be created)
        training: Whether in training mode (applies dropout)
    Returns:
        batch: Dictionary with input_ids, attention_mask, modality_type_ids, original_input_ids, family_ids, conditioning_info
    """
    # Filter out None samples (filtered due to length)
    valid_batch = [sample for sample in batch if sample is not None]
    
    # If no valid samples in batch, return empty batch
    if not valid_batch:
        return {
            'input_ids': torch.empty(0, dtype=torch.long),
            'attention_mask': torch.empty(0, dtype=torch.long),
            'labels': torch.empty(0, dtype=torch.long),
            'modality_type_ids': torch.empty(0, dtype=torch.long),
            'family_ids': []
        }
    
    # Get [PAD] token ID from special tokens
    pad_token_id = vocabulary['special_tokens']['[PAD]']
    padding_label_id = config['data']['padding_label_id']
    
    # Get max length in this batch
    max_len = max(len(sample['input_ids']) for sample in valid_batch)
    
    # Pad sequences with [PAD] token (NO MLM masking applied)
    input_ids = []
    attention_mask = []
    labels = []
    modality_type_ids = []
    
    # Precompute modality boundaries
    special = vocabulary['special_tokens']
    
    for sample in valid_batch:
        current_len = len(sample['input_ids'])
        padding_needed = max_len - current_len
        
        # Build modality_type_ids for this sample
        ids = sample['input_ids'].tolist()
        mod_ids = []
        mode = None
        for tid in ids:
            if tid == special['[SEQ_START]']:
                mode = 0
            elif tid == special['[STRUCT_START]']:
                mode = 1
            elif tid == special['[CONS_START]']:
                mode = 2
            elif tid == special['[GO_START]']:
                mode = 3
            mod_ids.append(mode if mode is not None else -1)
        
        # Pad modality_type_ids
        if padding_needed > 0:
            padding_tokens = torch.full((padding_needed,), pad_token_id, dtype=torch.long)
            padding_attention = torch.zeros(padding_needed, dtype=torch.long)
            padding_labels = torch.full((padding_needed,), padding_label_id, dtype=torch.long)
            padded_input_ids = torch.cat([sample['input_ids'], padding_tokens])
            padded_attention_mask = torch.cat([sample['attention_mask'], padding_attention])
            padded_labels = torch.cat([sample['labels'], padding_labels])
            padded_modality_type_ids = torch.tensor(mod_ids + [-1]*padding_needed, dtype=torch.long)
        else:
            padded_input_ids = sample['input_ids']
            padded_attention_mask = sample['attention_mask']
            padded_labels = sample['labels']
            padded_modality_type_ids = torch.tensor(mod_ids, dtype=torch.long)
        
        input_ids.append(padded_input_ids)
        attention_mask.append(padded_attention_mask)
        labels.append(padded_labels)
        modality_type_ids.append(padded_modality_type_ids)
    
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    labels = torch.stack(labels)
    modality_type_ids = torch.stack(modality_type_ids)
    
    # Save original input_ids (NO MLM masking applied)
    original_input_ids = input_ids.clone()
    
    # Collect family_ids (rfam_id) for each sample
    family_ids = [sample['rfam_id'] for sample in valid_batch]
    
    # Create base batch WITHOUT MLM masking
    base_batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'original_input_ids': original_input_ids,
        'modality_type_ids': modality_type_ids,
        'family_ids': family_ids
    }
    
    # Create or use provided data preparator
    if data_preparator is None:
        # Use default settings from config if available
        diffusion_cfg = config['training'].get('discrete_diffusion', {})
        cond_cfg = diffusion_cfg.get('conditional_generation', {})
        data_preparator = ConditionalDataPreparator(
            vocabulary=vocabulary,
            modality_dropout_rates=cond_cfg.get('modality_dropout_rates', None),
            target_modality=cond_cfg.get('target_modality', 'sequence'),
            conditioning_modalities=cond_cfg.get('conditioning_modalities', ['ss', 'consensus', 'go_terms'])
        )
    
    # Apply conditional data preparation (modality dropout, [DROPPED] tokens)
    diffusion_batch = data_preparator.prepare_conditional_batch(base_batch, training=training)
    return diffusion_batch 