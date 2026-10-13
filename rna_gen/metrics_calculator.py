import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MLMMetricsCalculator:
    """
    Metrics calculator specifically for Masked Language Modeling phase.
    Tracks standard MLM metrics: accuracy, loss, perplexity per modality.
    """
    
    def __init__(self, vocabulary: Dict, modality_offsets: Dict):
        self.vocabulary = vocabulary
        self.modality_offsets = modality_offsets
        self.padding_label_id = -100
        
        # Phase tracking
        self.phases = ['training', 'validation', 'evaluation']
        self.metrics = {}
        self.reset_all()
        
        # Get sequence vocabulary for nucleotide analysis
        self.seq_vocab = vocabulary['modality_vocabs']['sequence']
        self.seq_offset = vocabulary['modality_offsets']['sequence']
        
        logger.info("Initialized MLM metrics calculator")
    
    def reset_all(self):
        """Reset all metrics for all phases."""
        for phase in self.phases:
            self.reset_phase(phase)
    
    def reset_phase(self, phase: str):
        """Reset metrics for a specific phase."""
        self.metrics[phase] = {
            'modality_stats': {
                'sequence': {'total_tokens': 0, 'correct_tokens': 0, 'total_loss': 0.0, 'batches': 0},
                'secondary_structure': {'total_tokens': 0, 'correct_tokens': 0, 'total_loss': 0.0, 'batches': 0},
                'consensus': {'total_tokens': 0, 'correct_tokens': 0, 'total_loss': 0.0, 'batches': 0},
                'go_terms': {'total_tokens': 0, 'correct_tokens': 0, 'total_loss': 0.0, 'batches': 0}
            },
            'overall_stats': {'total_tokens': 0, 'correct_tokens': 0, 'total_loss': 0.0, 'batches': 0},
            'nucleotide_stats': {nuc: {'tp': 0, 'fp': 0, 'fn': 0} for nuc in ['A', 'U', 'G', 'C', 'N']},
            'training_stats': {
                'total_sequences': 0,
                'total_masked_tokens': 0,
                'unique_families': set(),
                'unique_sequences': set(),  # Track unique sequence content
                'total_families': 0,
                'sequence_lengths': [],
                'total_dropped_tokens': 0,
                'modality_dropout_combinations': defaultdict(int)
            },
            'token_prediction_counts': {
                'sequence': defaultdict(int),
                'secondary_structure': defaultdict(int),
                'consensus': defaultdict(int),
                'go_terms': defaultdict(int)
            }
        }
    
    def update_batch_metrics(self, 
                           predictions: torch.Tensor,
                           targets: torch.Tensor, 
                           modality_type_ids: torch.Tensor,
                           phase: str = 'training',
                           modality_losses: Optional[Dict[str, float]] = None,
                           overall_loss: Optional[float] = None,
                           family_ids: Optional[List] = None,
                           input_ids: Optional[torch.Tensor] = None,
                           mlm_dropout_info: Optional[Dict] = None,
                           original_input_ids: Optional[torch.Tensor] = None):
        """
        Update metrics from a single batch.
        
        Args:
            predictions: Model predictions (batch_size, seq_len, vocab_size)
            targets: Target labels (batch_size, seq_len) 
            modality_type_ids: Modality type IDs (batch_size, seq_len)
            phase: training/validation/evaluation
            modality_losses: Per-modality loss values
            overall_loss: Overall batch loss
            family_ids: Family IDs for diversity tracking
            input_ids: Input token IDs (for excluding [DROPPED] tokens)
            mlm_dropout_info: Information about modality dropout applied
        """
        batch_size, seq_len = targets.shape
        
        # Get predictions as token IDs
        pred_token_ids = predictions.argmax(dim=-1)  # (batch_size, seq_len)
        
        # Create mask for non-padding positions
        valid_mask = (targets != self.padding_label_id)
        
        # CRITICAL: Exclude [DROPPED] token positions from ALL metrics calculations
        if input_ids is not None:
            dropped_token_id = self.vocabulary['special_tokens']['[DROPPED]']
            valid_mask = valid_mask & (input_ids != dropped_token_id)
        
        # Update overall stats
        total_valid = valid_mask.sum().item()
        correct_overall = ((pred_token_ids == targets) & valid_mask).sum().item()
        
        self.metrics[phase]['overall_stats']['total_tokens'] += total_valid
        self.metrics[phase]['overall_stats']['correct_tokens'] += correct_overall
        self.metrics[phase]['overall_stats']['batches'] += 1
        if overall_loss is not None:
            self.metrics[phase]['overall_stats']['total_loss'] += overall_loss
        
        # Update per-modality stats and track token predictions
        modality_map = {0: 'sequence', 1: 'secondary_structure', 2: 'consensus', 3: 'go_terms'}
        
        for mod_id, mod_name in modality_map.items():
            # Create mask for this modality (and valid tokens)
            mod_mask = (modality_type_ids == mod_id) & valid_mask
            
            if mod_mask.sum() > 0:
                mod_total = mod_mask.sum().item()
                mod_correct = ((pred_token_ids == targets) & mod_mask).sum().item()
                
                self.metrics[phase]['modality_stats'][mod_name]['total_tokens'] += mod_total
                self.metrics[phase]['modality_stats'][mod_name]['correct_tokens'] += mod_correct
                
                # Add modality-specific loss if provided
                if modality_losses and mod_name in modality_losses:
                    self.metrics[phase]['modality_stats'][mod_name]['total_loss'] += modality_losses[mod_name]
                    self.metrics[phase]['modality_stats'][mod_name]['batches'] += 1
                
                # Track token prediction frequencies for this modality
                mod_preds = pred_token_ids[mod_mask].cpu().tolist()
                for pred_token_id in mod_preds:
                    self.metrics[phase]['token_prediction_counts'][mod_name][pred_token_id] += 1
        
        # Update nucleotide-level metrics for sequence modality
        self._update_nucleotide_metrics(pred_token_ids, targets, modality_type_ids, phase)
        
        # Update training stats
        self.metrics[phase]['training_stats']['total_sequences'] += batch_size
        if family_ids:
            self.metrics[phase]['training_stats']['unique_families'].update(family_ids)
            self.metrics[phase]['training_stats']['total_families'] += len(family_ids)
        
        # Count masked tokens
        mask_token_id = self.vocabulary['special_tokens']['[MASK]']
        masked_count = (predictions.argmax(dim=-1) == mask_token_id).sum().item()  # Count predicted masks
        self.metrics[phase]['training_stats']['total_masked_tokens'] += masked_count
        
        # Track sequence lengths using FULL sequence modality (exclude specials), and unique sequences as before
        for b in range(batch_size):
            # Compute content-only sequence mask from original inputs when available
            if original_input_ids is not None:
                tokens_row = original_input_ids[b]
            else:
                # Fallback to masked input_ids if original not provided
                tokens_row = input_ids[b] if input_ids is not None else targets[b]

            # Content tokens for sequence modality: modality==0 and token id within sequence vocab range
            seq_content_mask = (modality_type_ids[b] == 0)
            if self.seq_offset is not None and self.seq_vocab is not None:
                seq_content_mask = seq_content_mask & (tokens_row >= self.seq_offset) & (tokens_row < self.seq_offset + len(self.seq_vocab))

            if seq_content_mask.sum() > 0:
                seq_full_len = seq_content_mask.sum().item()
                self.metrics[phase]['training_stats']['sequence_lengths'].append(seq_full_len)

            # Track UNIQUE SEQUENCES based on the FULL sequence content (sequence modality only)
            # Prefer original_input_ids to avoid dependence on masking/dropout; fallback remains safe
            full_seq_tokens = tokens_row[seq_content_mask].cpu().tolist()
            if full_seq_tokens:
                full_seq_tuple = tuple(full_seq_tokens)
                self.metrics[phase]['training_stats']['unique_sequences'].add(full_seq_tuple)
        
        # Track modality dropout statistics (MLM-specific)
        if mlm_dropout_info is not None:
            # Count total dropped tokens
            self.metrics[phase]['training_stats']['total_dropped_tokens'] += mlm_dropout_info.get('total_content_tokens_dropped', 0)
            
            # Track modality dropout combinations
            dropped_modalities_list = mlm_dropout_info.get('dropped_modalities', [])
            for sample_dropped_mods in dropped_modalities_list:
                # Sort modalities to ensure consistent combination naming
                sorted_mods = sorted(sample_dropped_mods)
                if sorted_mods:
                    combo_name = '+'.join(sorted_mods)
                else:
                    combo_name = 'none'
                self.metrics[phase]['training_stats']['modality_dropout_combinations'][combo_name] += 1
    
    def _update_nucleotide_metrics(self, pred_token_ids: torch.Tensor, targets: torch.Tensor, 
                                 modality_type_ids: torch.Tensor, phase: str):
        """Update nucleotide-level confusion matrix for sequence tokens."""
        # Focus on sequence modality only
        seq_mask = (modality_type_ids == 0) & (targets != self.padding_label_id)
        
        if seq_mask.sum() == 0:
            return
        
        # Get sequence predictions and targets
        seq_preds = pred_token_ids[seq_mask]
        seq_targets = targets[seq_mask]
        
        # Convert token IDs to nucleotides
        for pred_id, target_id in zip(seq_preds.cpu().numpy(), seq_targets.cpu().numpy()):
            pred_nuc = self._token_id_to_nucleotide(pred_id)
            target_nuc = self._token_id_to_nucleotide(target_id)
            
            if target_nuc is not None:  # Valid sequence token
                if pred_nuc == target_nuc:
                    self.metrics[phase]['nucleotide_stats'][target_nuc]['tp'] += 1
                else:
                    self.metrics[phase]['nucleotide_stats'][target_nuc]['fn'] += 1
                    if pred_nuc is not None:
                        self.metrics[phase]['nucleotide_stats'][pred_nuc]['fp'] += 1
    
    def _token_id_to_nucleotide(self, token_id: int) -> Optional[str]:
        """Convert token ID to nucleotide (A, U, G, C, N) or None."""
        if self.seq_offset <= token_id < self.seq_offset + len(self.seq_vocab):
            local_id = token_id - self.seq_offset
            token = self.seq_vocab[local_id]
            if token in {'A', 'U', 'G', 'C', 'N'}:
                return token
        return None
    
    def calculate_metrics(self, phase: str = 'training') -> Dict[str, Any]:
        """Calculate final metrics for a phase."""
        stats = self.metrics[phase]
        results = {}
        
        # Overall metrics
        overall = stats['overall_stats']
        if overall['total_tokens'] > 0:
            results['overall'] = {
                'accuracy': overall['correct_tokens'] / overall['total_tokens'],
                'total_tokens': overall['total_tokens'],
                'loss': overall['total_loss'] / overall['batches'] if overall['batches'] > 0 else 0.0
            }
            results['overall']['perplexity'] = np.exp(results['overall']['loss'])
        
        # Per-modality metrics
        results['modalities'] = {}
        for mod_name, mod_stats in stats['modality_stats'].items():
            if mod_stats['total_tokens'] > 0:
                accuracy = mod_stats['correct_tokens'] / mod_stats['total_tokens']
                avg_loss = mod_stats['total_loss'] / mod_stats['batches'] if mod_stats['batches'] > 0 else 0.0
                
                results['modalities'][mod_name] = {
                    'accuracy': accuracy,
                    'loss': avg_loss,
                    'perplexity': np.exp(avg_loss) if avg_loss > 0 else 1.0,
                    'total_tokens': mod_stats['total_tokens']
                }
        
        # Nucleotide metrics
        results['nucleotides'] = {}
        for nuc, nuc_stats in stats['nucleotide_stats'].items():
            tp, fp, fn = nuc_stats['tp'], nuc_stats['fp'], nuc_stats['fn']
            if (tp + fp + fn) > 0:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                results['nucleotides'][nuc] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'support': tp + fn
                }
        
        # Training statistics
        train_stats = stats['training_stats']
        results['training_stats'] = {
            'total_sequences': train_stats['total_sequences'],
            'unique_sequences': len(train_stats['unique_sequences']),
            'unique_families': len(train_stats['unique_families']),
            'total_families': train_stats['total_families'],
            'total_masked_tokens': train_stats['total_masked_tokens'],
            'total_dropped_tokens': train_stats['total_dropped_tokens'],
            'avg_sequence_length': np.mean(train_stats['sequence_lengths']) if train_stats['sequence_lengths'] else 0.0,
            'std_sequence_length': np.std(train_stats['sequence_lengths']) if train_stats['sequence_lengths'] else 0.0,
            'min_sequence_length': np.min(train_stats['sequence_lengths']) if train_stats['sequence_lengths'] else 0,
            'max_sequence_length': np.max(train_stats['sequence_lengths']) if train_stats['sequence_lengths'] else 0
        }
        
        # Add modality dropout combinations as a separate key for wandb logging
        results['modality_dropout_combinations'] = dict(train_stats['modality_dropout_combinations'])
        
        # Top token prediction frequencies
        results['top_token_predictions'] = {}
        for modality_name in ['sequence', 'secondary_structure', 'consensus', 'go_terms']:
            token_counts = stats['token_prediction_counts'][modality_name]
            if token_counts:
                # Get top 4 most frequent tokens
                top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:4]
                results['top_token_predictions'][modality_name] = {
                    'token_ids': [token_id for token_id, _ in top_tokens],
                    'counts': [count for _, count in top_tokens]
                }
            else:
                results['top_token_predictions'][modality_name] = {
                    'token_ids': [],
                    'counts': []
                }
        
        return results
    
    def log_metrics(self, phase: str, step: int, results: Dict[str, Any]):
        """Log metrics in a structured format."""
        logger.info(f"\n{'='*60}")
        logger.info(f"MLM METRICS - {phase.upper()} - Step {step}")
        logger.info(f"{'='*60}")
        
        # Overall metrics
        if 'overall' in results:
            overall = results['overall']
            logger.info(f"OVERALL: Acc={overall['accuracy']:.4f}, Loss={overall['loss']:.4f}, "
                       f"Perp={overall['perplexity']:.2f}, Tokens={overall['total_tokens']}")
        
        # Per-modality metrics  
        if 'modalities' in results:
            for mod_name, metrics in results['modalities'].items():
                logger.info(f"{mod_name.upper()}: Acc={metrics['accuracy']:.4f}, "
                           f"Loss={metrics['loss']:.4f}, Perp={metrics['perplexity']:.2f}, "
                           f"Tokens={metrics['total_tokens']}")
        
        # Nucleotide metrics
        if 'nucleotides' in results:
            logger.info("NUCLEOTIDES:")
            for nuc in ['A', 'U', 'G', 'C', 'N']:
                if nuc in results['nucleotides']:
                    metrics = results['nucleotides'][nuc]
                    logger.info(f"  {nuc}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                               f"F1={metrics['f1_score']:.3f}, Support={metrics['support']}")
        
        # Training stats
        if 'training_stats' in results:
            stats = results['training_stats']
            logger.info(f"STATS: Sequences={stats['total_sequences']} (unique: {stats['unique_sequences']}), "
                       f"Families={stats['total_families']} (unique: {stats['unique_families']}), "
                       f"AvgLen={stats['avg_sequence_length']:.1f}±{stats['std_sequence_length']:.1f}, "
                       f"Range=[{stats['min_sequence_length']}-{stats['max_sequence_length']}]")


class DiffusionMetricsCalculator:
    """
    Metrics calculator specifically for Diffusion phase.
    Tracks generation quality, denoising accuracy, and conditioning effectiveness.
    """
    
    def __init__(self, vocabulary: Dict, modality_offsets: Dict):
        self.vocabulary = vocabulary
        self.modality_offsets = modality_offsets
        self.padding_label_id = -100
        
        # Get important token IDs
        self.mask_token_id = vocabulary['special_tokens']['[MASK]']
        self.dropped_token_id = vocabulary['special_tokens']['[DROPPED]']
        
        # Phase tracking
        self.phases = ['training', 'validation', 'evaluation']
        self.metrics = {}
        self.reset_all()
        
        # Get sequence vocabulary for nucleotide analysis
        self.seq_vocab = vocabulary['modality_vocabs']['sequence']
        self.seq_offset = vocabulary['modality_offsets']['sequence']
        
        logger.info("Initialized Diffusion metrics calculator")
    
    def reset_all(self):
        """Reset all metrics for all phases."""
        for phase in self.phases:
            self.reset_phase(phase)
    
    def reset_phase(self, phase: str):
        """Reset metrics for a specific phase."""
        self.metrics[phase] = {
            # Core generation metrics (only on corrupted sequence positions)
            'generation': {
                'total_corrupted_tokens': 0,
                'correct_reconstructions': 0,
                'total_loss': 0.0,
                'loss_count': 0
            },
            # Denoising metrics at different noise levels
            'denoising_by_timestep': defaultdict(lambda: {
                'total_tokens': 0, 'correct_tokens': 0, 'loss_sum': 0.0, 'count': 0
            }),
            # Conditioning effectiveness
            'conditioning': {
                'no_dropout': {'total_tokens': 0, 'correct_tokens': 0, 'loss_sum': 0.0, 'count': 0},
                'partial_dropout': {'total_tokens': 0, 'correct_tokens': 0, 'loss_sum': 0.0, 'count': 0},
                'full_dropout': {'total_tokens': 0, 'correct_tokens': 0, 'loss_sum': 0.0, 'count': 0}
            },
            # Nucleotide-level generation quality  
            'nucleotide_generation': {nuc: {'tp': 0, 'fp': 0, 'fn': 0} for nuc in ['A', 'U', 'G', 'C', 'N']},
            # Training statistics
            'training_stats': {
                'total_samples': 0,
                'total_timesteps_seen': [],
                'corruption_rates_seen': [],
                'per_sequence_corruption_rates': [],  # Track per-sequence corruption rates for bounds checking
                'unique_families': set(),
                'unique_sequences': set(),  # Track unique sequence content
                'total_families': 0,
                'sequence_lengths': [],  # Track sequence lengths
                'modality_dropout_stats': defaultdict(int),
                'modality_dropout_combinations': defaultdict(int)  # Track dropout combinations
            },
            'token_prediction_counts': {
                'sequence': defaultdict(int),
                'secondary_structure': defaultdict(int),
                'consensus': defaultdict(int),
                'go_terms': defaultdict(int)
            }
        }
    
    def update_batch_metrics(self,
                           predictions: torch.Tensor,
                           targets: torch.Tensor,
                           timesteps: torch.Tensor,
                           sequence_mask: torch.Tensor,
                           noise_info: Dict,
                           conditioning_info: Dict,
                           loss: float,  # Note: Unused - we compute per-sequence losses instead
                           phase: str = 'training',
                           family_ids: Optional[List] = None):
        """
        Update diffusion metrics from a single batch.
        
        Args:
            predictions: Model predictions (batch_size, seq_len, vocab_size)
            targets: Original sequence tokens (batch_size, seq_len)
            timesteps: Diffusion timesteps (batch_size,)
            sequence_mask: Mask for sequence modality positions (batch_size, seq_len)
            noise_info: Information about noise application
            conditioning_info: Information about conditioning/dropout
            loss: Batch loss value
            phase: training/validation/evaluation
            family_ids: Family IDs for diversity tracking
        """
        batch_size = predictions.shape[0]
        
        # Get predictions as token IDs
        pred_token_ids = predictions.argmax(dim=-1)  # (batch_size, seq_len)
        
        # Focus ONLY on corrupted sequence positions (this is what we're actually trying to generate)
        noise_mask = noise_info.get('noise_mask', torch.zeros_like(targets, dtype=torch.bool))
        
        # Combined mask: sequence positions that were actually corrupted  
        eval_mask = sequence_mask & noise_mask
        
        if eval_mask.sum() == 0:
            return  # No corrupted sequence tokens to evaluate
        
        # Update core generation metrics using per-sequence mean loss
        total_corrupted = eval_mask.sum().item()
        correct_reconstructions = ((pred_token_ids == targets) & eval_mask).sum().item()
        
        # Calculate per-sequence losses for overall generation metrics
        batch_seq_losses = []
        for b in range(batch_size):
            sample_mask = eval_mask[b]
            if sample_mask.sum() > 0:
                sample_predictions = predictions[b][sample_mask]
                sample_targets = targets[b][sample_mask]
                sample_ce_losses = torch.nn.functional.cross_entropy(sample_predictions, sample_targets, reduction='none')
                sample_seq_loss = sample_ce_losses.mean().item()
                batch_seq_losses.append(sample_seq_loss)
        
        # Average per-sequence loss across the batch
        if batch_seq_losses:
            batch_mean_seq_loss = sum(batch_seq_losses) / len(batch_seq_losses)
        else:
            batch_mean_seq_loss = 0.0
        
        self.metrics[phase]['generation']['total_corrupted_tokens'] += total_corrupted
        self.metrics[phase]['generation']['correct_reconstructions'] += correct_reconstructions
        self.metrics[phase]['generation']['total_loss'] += batch_mean_seq_loss
        self.metrics[phase]['generation']['loss_count'] += 1
        
        # Update metrics by timestep (binned for analysis) using per-sequence loss
        for b in range(batch_size):
            timestep = timesteps[b].item()
            timestep_bin = (timestep // 100) * 100  # Bin timesteps in groups of 100
            
            sample_eval_mask = eval_mask[b]
            if sample_eval_mask.sum() > 0:
                sample_total = sample_eval_mask.sum().item()
                sample_correct = ((pred_token_ids[b] == targets[b]) & sample_eval_mask).sum().item()
                
                # Calculate per-sequence loss for this sample
                sample_predictions = predictions[b][sample_eval_mask]
                sample_targets = targets[b][sample_eval_mask]
                sample_ce_losses = torch.nn.functional.cross_entropy(sample_predictions, sample_targets, reduction='none')
                sample_seq_loss = sample_ce_losses.mean().item()
                
                self.metrics[phase]['denoising_by_timestep'][timestep_bin]['total_tokens'] += sample_total
                self.metrics[phase]['denoising_by_timestep'][timestep_bin]['correct_tokens'] += sample_correct
                self.metrics[phase]['denoising_by_timestep'][timestep_bin]['loss_sum'] += sample_seq_loss
                self.metrics[phase]['denoising_by_timestep'][timestep_bin]['count'] += 1
        
        # DIFFUSION MODE: Only calculate metrics on MASKED SEQUENCE positions
        # Conditioning modalities are used for conditioning, not for evaluation
        
        # Calculate per-sequence mean loss over supervised tokens
        # L = (1/B) * Σ_b [ Σ_i (m_b,i * CE_b,i) / max(1, Σ_i m_b,i) ]
        # This removes dependence on how many tokens were masked in each sequence
        
        for b in range(batch_size):
            # Only evaluate on masked sequence positions (the generation task)
            masked_sequence_positions = noise_mask[b]  # These are the positions we're trying to predict
            
            if masked_sequence_positions.sum() == 0:
                continue  # No masked positions to evaluate
                
            # Calculate accuracy on masked sequence positions only
            sample_total = masked_sequence_positions.sum().item()
            sample_correct = ((pred_token_ids[b] == targets[b]) & masked_sequence_positions).sum().item()
            
            # Calculate per-sequence loss using the formula:
            # Per-sequence loss = Σ_i (m_b,i * CE_b,i) / max(1, Σ_i m_b,i)
            seq_predictions = predictions[b][masked_sequence_positions]
            seq_targets = targets[b][masked_sequence_positions]
            
            # Compute cross-entropy loss for each token (reduction='none')
            token_ce_losses = torch.nn.functional.cross_entropy(seq_predictions, seq_targets, reduction='none')
            
            # Per-sequence mean: sum of CE losses / number of masked tokens
            # This is equivalent to the formula since m_b,i = 1 for masked positions, 0 for others
            seq_loss = token_ce_losses.mean().item()
            
            # Debug: Check for infinite or NaN values
            if np.isinf(seq_loss) or np.isnan(seq_loss):
                print(f"WARNING: Infinite/NaN sequence loss detected: {seq_loss}")
                print(f"  seq_predictions shape: {seq_predictions.shape}")
                print(f"  seq_targets shape: {seq_targets.shape}")
                print(f"  masked_sequence_positions sum: {masked_sequence_positions.sum().item()}")
                print(f"  token_ce_losses: {token_ce_losses}")
                
                # Additional debugging
                print(f"  seq_targets min/max: {seq_targets.min().item()}/{seq_targets.max().item()}")
                print(f"  seq_predictions min/max: {seq_predictions.min().item()}/{seq_predictions.max().item()}")
                print(f"  seq_predictions has inf: {torch.isinf(seq_predictions).any().item()}")
                print(f"  seq_predictions has nan: {torch.isnan(seq_predictions).any().item()}")
                
                # Check if targets are within vocabulary range
                vocab_size = seq_predictions.shape[-1]
                invalid_targets = (seq_targets < 0) | (seq_targets >= vocab_size)
                if invalid_targets.any():
                    print(f"  INVALID TARGETS: {invalid_targets.sum().item()} targets outside vocab range [0, {vocab_size})")
                    print(f"  Invalid target values: {seq_targets[invalid_targets]}")
                
                seq_loss = 0.0  # Set to 0 to avoid infinite accumulation
            
            # Categorize by dropout level (for conditioning effectiveness analysis)
            dropped_modalities = conditioning_info.get('dropped_modalities', [])
            num_dropped = len(dropped_modalities[b]) if b < len(dropped_modalities) else 0
            if num_dropped == 0:
                category = 'no_dropout'
            elif num_dropped >= 3:  # All 3 conditioning modalities dropped
                category = 'full_dropout'
            else:
                category = 'partial_dropout'
            
            # Update sequence generation metrics using per-sequence loss
            # Now each sequence contributes equally regardless of number of masked tokens
            self.metrics[phase]['conditioning'][category]['total_tokens'] += sample_total
            self.metrics[phase]['conditioning'][category]['correct_tokens'] += sample_correct
            self.metrics[phase]['conditioning'][category]['loss_sum'] += seq_loss
            self.metrics[phase]['conditioning'][category]['count'] += 1
        
        # Update nucleotide-level generation metrics (only on corrupted positions)
        self._update_nucleotide_generation_metrics(pred_token_ids, targets, eval_mask, phase)
        
        # Track token prediction frequencies for sequence modality (only on corrupted positions)
        seq_preds = pred_token_ids[eval_mask].cpu().tolist()
        for pred_token_id in seq_preds:
            self.metrics[phase]['token_prediction_counts']['sequence'][pred_token_id] += 1
        
        # Update training statistics
        self.metrics[phase]['training_stats']['total_samples'] += batch_size
        self.metrics[phase]['training_stats']['total_timesteps_seen'].extend(timesteps.cpu().tolist())
        
        if 'corruption_rate' in noise_info:
            self.metrics[phase]['training_stats']['corruption_rates_seen'].append(noise_info['corruption_rate'])
        
        if family_ids:
            self.metrics[phase]['training_stats']['unique_families'].update(family_ids)
            self.metrics[phase]['training_stats']['total_families'] += len(family_ids)
        
        # Track unique sequences and sequence lengths (only sequence modality, use original targets not predictions)
        for b in range(batch_size):
            # Ensure sequence_mask is on the same device as targets
            seq_positions = sequence_mask[b].to(targets.device)
            if seq_positions.sum() > 0:
                # Extract original sequence tokens
                seq_tokens = targets[b][seq_positions].cpu().tolist()
                if seq_tokens:  # Only add non-empty sequences
                    seq_tuple = tuple(seq_tokens)
                    self.metrics[phase]['training_stats']['unique_sequences'].add(seq_tuple)
                    # Track sequence length
                    seq_len = len(seq_tokens)
                    self.metrics[phase]['training_stats']['sequence_lengths'].append(seq_len)
                
                # Track per-sequence corruption rate for bounds checking
                sample_seq_positions = seq_positions.sum().item()
                sample_corrupted_positions = noise_mask[b][seq_positions].sum().item()
                if sample_seq_positions > 0:
                    sample_corruption_rate = sample_corrupted_positions / sample_seq_positions
                    self.metrics[phase]['training_stats']['per_sequence_corruption_rates'].append(sample_corruption_rate)
        
        # Track modality dropout statistics
        for dropped_list in dropped_modalities:
            for modality in dropped_list:
                self.metrics[phase]['training_stats']['modality_dropout_stats'][modality] += 1
            
            # Track modality dropout combinations
            if not dropped_list:
                combo_key = "none"
            else:
                # Sort modalities for consistent naming
                combo_key = "_".join(sorted(dropped_list))
            self.metrics[phase]['training_stats']['modality_dropout_combinations'][combo_key] += 1
    
    def _update_nucleotide_generation_metrics(self, pred_token_ids: torch.Tensor, targets: torch.Tensor,
                                            eval_mask: torch.Tensor, phase: str):
        """Update nucleotide-level generation quality metrics."""
        if eval_mask.sum() == 0:
            return
        
        # Get predictions and targets for corrupted sequence positions only
        corrupted_preds = pred_token_ids[eval_mask]
        corrupted_targets = targets[eval_mask]
        
        # Convert to nucleotides and update confusion matrix
        for pred_id, target_id in zip(corrupted_preds.cpu().numpy(), corrupted_targets.cpu().numpy()):
            pred_nuc = self._token_id_to_nucleotide(pred_id)
            target_nuc = self._token_id_to_nucleotide(target_id)
            
            if target_nuc is not None:  # Valid sequence nucleotide
                if pred_nuc == target_nuc:
                    self.metrics[phase]['nucleotide_generation'][target_nuc]['tp'] += 1
                else:
                    self.metrics[phase]['nucleotide_generation'][target_nuc]['fn'] += 1
                    if pred_nuc is not None:
                        self.metrics[phase]['nucleotide_generation'][pred_nuc]['fp'] += 1
    
    def _token_id_to_nucleotide(self, token_id: int) -> Optional[str]:
        """Convert token ID to nucleotide (A, U, G, C, N) or None."""
        if self.seq_offset <= token_id < self.seq_offset + len(self.seq_vocab):
            local_id = token_id - self.seq_offset
            token = self.seq_vocab[local_id]
            if token in {'A', 'U', 'G', 'C', 'N'}:
                return token
        return None
    
    def calculate_metrics(self, phase: str = 'training') -> Dict[str, Any]:
        """Calculate final metrics for a phase."""
        stats = self.metrics[phase]
        results = {}
        
        # Core generation metrics
        gen = stats['generation']
        if gen['total_corrupted_tokens'] > 0:
            results['generation'] = {
                'accuracy': gen['correct_reconstructions'] / gen['total_corrupted_tokens'],
                'loss': gen['total_loss'] / gen['loss_count'] if gen['loss_count'] > 0 else 0.0,
                'total_corrupted_tokens': gen['total_corrupted_tokens'],
                'total_batches': gen['loss_count']
            }
            results['generation']['perplexity'] = np.exp(results['generation']['loss'])
        
        # Denoising by timestep
        results['denoising_by_timestep'] = {}
        for timestep_bin, timestep_stats in stats['denoising_by_timestep'].items():
            if timestep_stats['total_tokens'] > 0:
                accuracy = timestep_stats['correct_tokens'] / timestep_stats['total_tokens']
                avg_loss = timestep_stats['loss_sum'] / timestep_stats['count'] if timestep_stats['count'] > 0 else 0.0
                
                results['denoising_by_timestep'][timestep_bin] = {
                    'accuracy': accuracy,
                    'loss': avg_loss,
                    'perplexity': np.exp(avg_loss) if avg_loss > 0 else 1.0,
                    'total_tokens': timestep_stats['total_tokens']
                }
        
        # Conditioning effectiveness
        results['conditioning'] = {}
        for condition_type, cond_stats in stats['conditioning'].items():
            if cond_stats['total_tokens'] > 0:
                accuracy = cond_stats['correct_tokens'] / cond_stats['total_tokens']
                # Fix division by zero and handle infinite values
                if cond_stats['count'] > 0:
                    avg_loss = cond_stats['loss_sum'] / cond_stats['count']
                    # Handle infinite or NaN values
                    if np.isinf(avg_loss) or np.isnan(avg_loss):
                        avg_loss = 0.0
                else:
                    avg_loss = 0.0
                
                results['conditioning'][condition_type] = {
                    'accuracy': accuracy,
                    'loss': avg_loss,
                    'perplexity': np.exp(avg_loss) if avg_loss > 0 and not np.isinf(avg_loss) else 1.0,
                    'total_tokens': cond_stats['total_tokens']
                }
        
        # Nucleotide generation quality
        results['nucleotide_generation'] = {}
        for nuc, nuc_stats in stats['nucleotide_generation'].items():
            tp, fp, fn = nuc_stats['tp'], nuc_stats['fp'], nuc_stats['fn']
            if (tp + fp + fn) > 0:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                results['nucleotide_generation'][nuc] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'support': tp + fn
                }
        
        # Training statistics
        train_stats = stats['training_stats']
        results['training_stats'] = {
            'total_samples': train_stats['total_samples'],
            'unique_sequences': len(train_stats['unique_sequences']),
            'unique_families': len(train_stats['unique_families']),
            'total_families': train_stats['total_families'],
            'avg_timestep': np.mean(train_stats['total_timesteps_seen']) if train_stats['total_timesteps_seen'] else 0.0,
            'avg_corruption_rate': np.mean(train_stats['corruption_rates_seen']) if train_stats['corruption_rates_seen'] else 0.0,
            'min_corruption_rate': np.min(train_stats['per_sequence_corruption_rates']) if train_stats['per_sequence_corruption_rates'] else 0.0,
            'max_corruption_rate': np.max(train_stats['per_sequence_corruption_rates']) if train_stats['per_sequence_corruption_rates'] else 0.0,
            'avg_sequence_length': np.mean(train_stats['sequence_lengths']) if train_stats['sequence_lengths'] else 0.0,
            'std_sequence_length': np.std(train_stats['sequence_lengths']) if train_stats['sequence_lengths'] else 0.0,
            'min_sequence_length': np.min(train_stats['sequence_lengths']) if train_stats['sequence_lengths'] else 0,
            'max_sequence_length': np.max(train_stats['sequence_lengths']) if train_stats['sequence_lengths'] else 0,
            'modality_dropout_counts': dict(train_stats['modality_dropout_stats'])
        }
        
        # Add modality dropout combinations as a separate key for wandb logging
        results['modality_dropout_combinations'] = dict(train_stats['modality_dropout_combinations'])
        
        # Top token prediction frequencies (only sequence modality for diffusion)
        results['top_token_predictions'] = {}
        token_counts = stats['token_prediction_counts']['sequence']
        if token_counts:
            # Get top 4 most frequent tokens
            top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:4]
            results['top_token_predictions']['sequence'] = {
                'token_ids': [token_id for token_id, _ in top_tokens],
                'counts': [count for _, count in top_tokens]
            }
        else:
            results['top_token_predictions']['sequence'] = {
                'token_ids': [],
                'counts': []
            }
        
        return results
    
    def log_metrics(self, phase: str, step: int, results: Dict[str, Any]):
        """Log metrics in a structured format."""
        logger.info(f"\n{'='*60}")
        logger.info(f"DIFFUSION METRICS - {phase.upper()} - Step {step}")
        logger.info(f"{'='*60}")
        
        # Core generation metrics
        if 'generation' in results:
            gen = results['generation']
            logger.info(f"GENERATION: Acc={gen['accuracy']:.4f}, Loss={gen['loss']:.4f}, "
                       f"Perp={gen['perplexity']:.2f}, Tokens={gen['total_corrupted_tokens']}")
        
        # Conditioning effectiveness
        if 'conditioning' in results:
            logger.info("CONDITIONING:")
            for cond_type, metrics in results['conditioning'].items():
                logger.info(f"  {cond_type}: Acc={metrics['accuracy']:.4f}, "
                           f"Loss={metrics['loss']:.4f}, Perp={metrics['perplexity']:.2f}")
        
        # Sample timestep analysis (show a few bins)
        if 'denoising_by_timestep' in results:
            logger.info("DENOISING BY TIMESTEP (sample):")
            sorted_timesteps = sorted(results['denoising_by_timestep'].keys())
            for timestep_bin in sorted_timesteps[:3]:  # Show first 3 bins
                metrics = results['denoising_by_timestep'][timestep_bin]
                logger.info(f"  t={timestep_bin}-{timestep_bin+99}: Acc={metrics['accuracy']:.4f}, "
                           f"Perp={metrics['perplexity']:.2f}")
        
        # Nucleotide generation quality
        if 'nucleotide_generation' in results:
            logger.info("NUCLEOTIDE GENERATION:")
            for nuc in ['A', 'U', 'G', 'C', 'N']:
                if nuc in results['nucleotide_generation']:
                    metrics = results['nucleotide_generation'][nuc]
                    logger.info(f"  {nuc}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                               f"F1={metrics['f1_score']:.3f}, Support={metrics['support']}")
        
        # Training stats
        if 'training_stats' in results:
            stats = results['training_stats']
            logger.info(f"STATS: Samples={stats['total_samples']} (unique: {stats['unique_sequences']}), "
                       f"Families={stats['total_families']} (unique: {stats['unique_families']}), "
                       f"AvgTimestep={stats['avg_timestep']:.1f}, "
                       f"AvgCorruption={stats['avg_corruption_rate']:.3f}, "
                       f"CorruptionBounds=[{stats['min_corruption_rate']:.3f}-{stats['max_corruption_rate']:.3f}], "
                       f"AvgLen={stats['avg_sequence_length']:.1f}±{stats['std_sequence_length']:.1f}, "
                       f"Range=[{stats['min_sequence_length']}-{stats['max_sequence_length']}]") 