"""
Training Data Preparation for RNA Multimodal Model

This script handles:
1. Family-based train/val/test splitting
2. Efficient data loading with streaming
3. Batch construction with variable-length GO terms
4. Sequence length truncation and padding
"""

import h5py
import numpy as np
import yaml
import random
import os
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
import json
from io import StringIO
import functools


class RNAMultimodalDataset(Dataset):
    """
    Dataset for RNA multimodal data with on-the-fly loading.
    
    Benefits of on-the-fly loading:
    - Memory efficient: Only loads data when needed
    - Scalable: Works with datasets of any size
    - Fast startup: No need to preload entire dataset
    - Flexible: Easy to modify data processing on-the-fly
    """
    
    def __init__(self, h5_file_path: str, split: str, data_indices: List[int], 
                 config: Dict, vocabulary_path: str):
        """
        Initialize dataset.
        
        Args:
            h5_file_path: Path to HDF5 file
            split: 'train', 'val', or 'test'
            data_indices: List of data indices for this split
            config: Configuration dictionary
            vocabulary_path: Path to vocabulary JSON file
        """
        self.h5_file_path = h5_file_path
        self.split = split
        self.data_indices = data_indices
        self.config = config
        self.vocabulary_path = vocabulary_path
        
        # Load vocabulary
        self.vocab = self._load_vocabulary()
        
        print(f"Initialized {split} dataset with {len(self.data_indices)} samples")
        print(f"  - Data indices range: {min(self.data_indices)} to {max(self.data_indices)}")
    
    def _load_vocabulary(self) -> Dict:
        """Load unified vocabulary."""
        with open(self.vocabulary_path, 'r') as f:
            return json.load(f)
    
    def __len__(self) -> int:
        return len(self.data_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with on-the-fly loading and length filtering."""
        data_idx = self.data_indices[idx]
        
        try:
            # Load data on-the-fly from HDF5 file
            with h5py.File(self.h5_file_path, 'r') as f:
                # Load data
                sequence = f['sequence'][data_idx].decode('utf-8')
                structure = f['ss'][data_idx].decode('utf-8')
                consensus = f['consensus'][data_idx].decode('utf-8')
                go_terms = f['go_terms'][data_idx].decode('utf-8')
                rfam_id = f['rfam_id'][data_idx].decode('utf-8')
                
                # Process and tokenize (this will raise ValueError if sequence too long)
                processed_data = self._process_sample(sequence, structure, consensus, go_terms)
                processed_data['rfam_id'] = rfam_id
                
                return processed_data
        except ValueError as e:
            # Skip samples that exceed max_sequence_length
            print(f"Skipping sample {idx} (data_idx={data_idx}): {e}")
            # Return a placeholder that will be filtered out by the collate function
            return None
        except Exception as e:
            print(f"Error loading sample {idx} (data_idx={data_idx}): {e}")
            raise
    
    def _process_sample(self, sequence: str, structure: str, consensus: str, go_terms: str) -> Dict[str, torch.Tensor]:
        """Process a single sample with total length filtering (no truncation)."""
        max_position_embeddings = self.config['data']['max_position_embeddings']
        max_go_terms = self.config['data']['max_go_terms']
        
        # Process GO terms first to calculate total length
        if go_terms:
            go_list = [term.strip() for term in go_terms.split(',') if term.strip()]
            # Add GO: prefix if not present
            go_list = [f"GO:{term}" if not term.startswith('GO:') else term for term in go_list]
            # Truncate to max GO terms
            go_list = go_list[:max_go_terms]
        else:
            go_list = []
        
        # Calculate total concatenated length: 6 special tokens + 3×max_sequence_length + max_go_terms
        # Special tokens: [CLS], [SEQ_START], [STRUCT_START], [CONS_START], [GO_START], [SEP] = 6
        total_length = 6 + len(sequence) + len(structure) + len(consensus) + len(go_list)
        
        # Check if total length exceeds max_position_embeddings
        if total_length > max_position_embeddings:
            raise ValueError(f"Total concatenated length ({total_length}) exceeds max_position_embeddings ({max_position_embeddings}): "
                           f"sequence={len(sequence)}, structure={len(structure)}, consensus={len(consensus)}, go_terms={len(go_list)}")
        
        # Tokenize using the vocabulary (no truncation)
        tokenized = self._tokenize_sample(sequence, structure, consensus, go_list)
        
        return tokenized
    
    def _tokenize_sample(self, sequence: str, structure: str, consensus: str, go_terms: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a sample using the unified vocabulary."""
        tokens = []
        
        # Add CLS token
        tokens.append(self.vocab['special_tokens']['[CLS]'])
        
        # Add sequence
        tokens.append(self.vocab['special_tokens']['[SEQ_START]'])
        for char in sequence:
            if char in self.vocab['modality_vocabs']['sequence']:
                char_idx = self.vocab['modality_vocabs']['sequence'].index(char)
                tokens.append(self.vocab['modality_offsets']['sequence'] + char_idx)
            else:
                # Use RNA-UNK
                unk_idx = self.vocab['modality_vocabs']['sequence'].index('[RNA-UNK]')
                tokens.append(self.vocab['modality_offsets']['sequence'] + unk_idx)
        
        # Add structure
        tokens.append(self.vocab['special_tokens']['[STRUCT_START]'])
        for char in structure:
            if char in self.vocab['modality_vocabs']['ss']:
                char_idx = self.vocab['modality_vocabs']['ss'].index(char)
                tokens.append(self.vocab['modality_offsets']['ss'] + char_idx)
            else:
                # Use STRUC-UNK
                unk_idx = self.vocab['modality_vocabs']['ss'].index('[STRUC-UNK]')
                tokens.append(self.vocab['modality_offsets']['ss'] + unk_idx)
        
        # Add consensus
        tokens.append(self.vocab['special_tokens']['[CONS_START]'])
        for char in consensus:
            if char in self.vocab['modality_vocabs']['consensus']:
                char_idx = self.vocab['modality_vocabs']['consensus'].index(char)
                tokens.append(self.vocab['modality_offsets']['consensus'] + char_idx)
            else:
                # Use CONS-UNK
                unk_idx = self.vocab['modality_vocabs']['consensus'].index('[CONS-UNK]')
                tokens.append(self.vocab['modality_offsets']['consensus'] + unk_idx)
        
        # Add GO terms
        tokens.append(self.vocab['special_tokens']['[GO_START]'])
        for term in go_terms:
            if term in self.vocab['modality_vocabs']['go_terms']:
                term_idx = self.vocab['modality_vocabs']['go_terms'].index(term)
                tokens.append(self.vocab['modality_offsets']['go_terms'] + term_idx)
            else:
                # Use GO-UNK
                unk_idx = self.vocab['modality_vocabs']['go_terms'].index('[GO-UNK]')
                tokens.append(self.vocab['modality_offsets']['go_terms'] + unk_idx)
        
        # Add SEP token
        tokens.append(self.vocab['special_tokens']['[SEP]'])
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.ones(len(tokens), dtype=torch.long),
            'labels': torch.tensor(tokens, dtype=torch.long)  # For MLM, labels are same as input_ids
        }


class FamilyDiverseSampler(Sampler):
    """Sampler that ensures each batch contains samples from diverse families."""
    
    def __init__(self, dataset: RNAMultimodalDataset, batch_size: int, family_lookup_path: str, seed: int = 42):
        """
        Initialize sampler.
        
        Args:
            dataset: RNAMultimodalDataset instance
            batch_size: Batch size
            family_lookup_path: Path to family lookup JSON file
            seed: Random seed for reproducible sampling
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        
        # Load pre-computed family lookup
        print(f"Loading family lookup for {dataset.split} split...")
        with open(family_lookup_path, 'r') as f:
            self.family_lookup = json.load(f)
        
        # Map dataset indices to families using the lookup
        print(f"Mapping families for {dataset.split} split...")
        self.sample_to_family = {}
        
        # Create reverse mapping: data_idx -> family_id
        data_idx_to_family = {}
        for family_id, data_indices in self.family_lookup.items():
            for data_idx in data_indices:
                data_idx_to_family[data_idx] = family_id
        
        # Map dataset indices to families
        for dataset_idx, data_idx in enumerate(dataset.data_indices):
            if data_idx in data_idx_to_family:
                self.sample_to_family[dataset_idx] = data_idx_to_family[data_idx]
            else:
                print(f"Warning: data_idx {data_idx} not found in family lookup")
        
        # Group samples by family
        self.family_to_samples = defaultdict(list)
        for sample_idx, family_id in self.sample_to_family.items():
            self.family_to_samples[family_id].append(sample_idx)
        
        self.families = list(self.family_to_samples.keys())
        print(f"Found {len(self.families)} families in {dataset.split} split with {len(self.dataset)} total samples")
    
    def __iter__(self) -> Iterator[int]:
        """Generate individual indices with diverse family sampling."""
        # Set random seed for reproducible sampling
        random.seed(self.seed)
        
        # Create a copy of family samples for sampling
        family_samples_copy = {family: samples.copy() for family, samples in self.family_to_samples.items()}
        
        # Get all available families (those with samples)
        available_families = [f for f in self.families if family_samples_copy[f]]
        
        # Continue yielding batches until we run out of families
        while len(available_families) >= self.batch_size:
            # Randomly select batch_size unique families
            selected_families = random.sample(available_families, self.batch_size)
            
            # For each selected family, randomly select 1 sequence
            batch_indices = []
            for family in selected_families:
                # Randomly select 1 sequence from this family
                sample_idx = random.choice(family_samples_copy[family])
                batch_indices.append(sample_idx)
                
                # Remove the selected sample from this family
                family_samples_copy[family].remove(sample_idx)
                
                # If this family is now empty, remove it from available families
                if not family_samples_copy[family]:
                    available_families.remove(family)
            
            # Yield all indices for this batch
            for idx in batch_indices:
                yield idx
    
    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size


class DistributedFamilyDiverseSampler:
    """Distributed sampler that maintains family diversity across ranks."""
    
    def __init__(self, dataset: RNAMultimodalDataset, num_replicas: int, rank: int, 
                 batch_size: int, family_lookup_path: str, shuffle: bool = True, 
                 seed: int = 42, drop_last: bool = False):
        """
        Initialize distributed family diverse sampler.
        
        Args:
            dataset: RNAMultimodalDataset instance
            num_replicas: Number of processes participating in distributed training
            rank: Rank of the current process
            batch_size: Batch size
            family_lookup_path: Path to family lookup JSON file
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducible sampling
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        
        # Load family information
        print(f"Loading family lookup for {dataset.split} split (rank {rank})...")
        if family_lookup_path is not None:
            with open(family_lookup_path, 'r') as f:
                self.family_lookup = json.load(f)
        else:
            # For testing: lookup will be set externally
            self.family_lookup = None
        
        # Map dataset indices to families using the lookup
        print(f"Mapping families for {dataset.split} split (rank {rank})...")
        self.sample_to_family = {}
        
        # Only proceed with mapping if family_lookup is available
        if self.family_lookup is not None:
            # Create reverse mapping: data_idx -> family_id
            data_idx_to_family = {}
            for family_id, data_indices in self.family_lookup.items():
                for data_idx in data_indices:
                    data_idx_to_family[data_idx] = family_id
            
            # Map dataset indices to families
            for dataset_idx, data_idx in enumerate(dataset.data_indices):
                if data_idx in data_idx_to_family:
                    self.sample_to_family[dataset_idx] = data_idx_to_family[data_idx]
                else:
                    print(f"Warning: data_idx {data_idx} not found in family lookup")
            
            # Group samples by family
            self.family_to_samples = defaultdict(list)
            for sample_idx, family_id in self.sample_to_family.items():
                self.family_to_samples[family_id].append(sample_idx)
            
            self.families = list(self.family_to_samples.keys())
            print(f"Found {len(self.families)} families in {dataset.split} split with {len(self.dataset)} total samples (rank {rank})")
        else:
            # Initialize empty structures for testing
            self.family_to_samples = defaultdict(list)
            self.families = []
        
        # Calculate samples per rank
        self.num_samples = len(dataset)
        self.total_size = self.num_samples
        if self.drop_last and self.total_size % self.num_replicas != 0:
            self.total_size = ((self.total_size // self.num_replicas) * self.num_replicas)
        self.num_samples_per_rank = self.total_size // self.num_replicas
        self.start_idx = self.rank * self.num_samples_per_rank
        self.end_idx = self.start_idx + self.num_samples_per_rank
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices with family diversity for this rank."""
        # Set random seed for reproducible sampling (include rank for different seeds per rank)
        random.seed(self.seed + self.rank)
        
        # APPROACH: Each rank sees ALL families, but samples different sequences from each family
        # This ensures maximum family diversity across all ranks while maintaining load balance
        
        all_families = list(self.family_to_samples.keys())
        if self.shuffle:
            # Shuffle families deterministically based on seed
            family_rng = random.Random(self.seed)
            family_rng.shuffle(all_families)
        
        # For each family, determine which samples this rank should process
        # Ensure each rank sees every family, even if some families have fewer samples than ranks
        rank_indices = []
        family_groups = defaultdict(list)
        
        for family in all_families:
            family_samples = self.family_to_samples[family]
            
            if len(family_samples) >= self.num_replicas:
                # Family has enough samples: use round-robin assignment
                rank_family_samples = [sample for i, sample in enumerate(family_samples) 
                                     if i % self.num_replicas == self.rank]
            else:
                # Family has fewer samples than ranks: ensure each rank gets at least one sample
                # Use modulo to cycle through available samples
                sample_idx = self.rank % len(family_samples)
                rank_family_samples = [family_samples[sample_idx]]
            
            # Every rank gets samples from every family
            rank_indices.extend(rank_family_samples)
            family_groups[family].extend(rank_family_samples)
        
        # Shuffle indices within this rank if requested
        if self.shuffle:
            random.shuffle(rank_indices)
        
        # Create diverse batches within this rank's data
        diverse_indices = []
        available_families = [f for f in family_groups.keys() if family_groups[f]]
        
        while len(available_families) >= 1:  # At least 1 family available
            # Select families for this batch (up to batch_size)
            batch_size = min(len(available_families), self.batch_size)
            selected_families = random.sample(available_families, batch_size)
            
            # For each family, take one sample
            for family in selected_families:
                if family_groups[family]:
                    sample_idx = family_groups[family].pop()
                    diverse_indices.append(sample_idx)
                    
                    # Remove family if no more samples
                    if not family_groups[family]:
                        available_families.remove(family)
        
        return iter(diverse_indices)
    
    def __len__(self) -> int:
        # Calculate actual number of samples this rank will process
        # Each rank sees ALL families, with at least one sample per family
        all_families = list(self.family_to_samples.keys())
        total_rank_samples = 0
        
        for family in all_families:
            family_samples = self.family_to_samples[family]
            
            if len(family_samples) >= self.num_replicas:
                # Family has enough samples: use round-robin assignment
                rank_family_samples = len([sample for i, sample in enumerate(family_samples) 
                                         if i % self.num_replicas == self.rank])
            else:
                # Family has fewer samples than ranks: each rank gets exactly one sample
                rank_family_samples = 1
            
            total_rank_samples += rank_family_samples
        
        return total_rank_samples


class SimpleFamilyDiverseSampler:
    """Simple sampler that ensures each batch contains samples from diverse families.
    
    For each batch:
    1. Randomly sample batch_size unique families (without replacement)
    2. For each selected family, randomly select 1 sequence (with replacement)
    
    This design ensures:
    - Equal probability for all families regardless of size
    - Continuous sequence diversity across epochs
    - No artificial exhaustion or stopping conditions
    """
    
    def __init__(self, dataset: RNAMultimodalDataset, batch_size: int, family_lookup_path: str, 
                 shuffle: bool = True, seed: int = 42):
        """
        Initialize simple family diverse sampler.
        
        Args:
            dataset: RNAMultimodalDataset instance
            batch_size: Batch size
            family_lookup_path: Path to family lookup JSON file
            shuffle: Whether to shuffle (kept for compatibility)
            seed: Random seed for reproducible sampling
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # Load family information
        print(f"Loading family lookup for {dataset.split} split...")
        if family_lookup_path is not None:
            with open(family_lookup_path, 'r') as f:
                self.family_lookup = json.load(f)
        else:
            # For testing: lookup will be set externally
            self.family_lookup = None
        
        # Map dataset indices to families using the lookup
        print(f"Mapping families for {dataset.split} split...")
        self.sample_to_family = {}
        
        # Only proceed with mapping if family_lookup is available
        if self.family_lookup is not None:
            # Create reverse mapping: data_idx -> family_id
            data_idx_to_family = {}
            for family_id, data_indices in self.family_lookup.items():
                for data_idx in data_indices:
                    data_idx_to_family[data_idx] = family_id
            
            # Map dataset indices to families
            for dataset_idx, data_idx in enumerate(dataset.data_indices):
                if data_idx in data_idx_to_family:
                    self.sample_to_family[dataset_idx] = data_idx_to_family[data_idx]
                else:
                    print(f"Warning: data_idx {data_idx} not found in family lookup")
            
            # Group samples by family
            self.family_to_samples = defaultdict(list)
            for sample_idx, family_id in self.sample_to_family.items():
                self.family_to_samples[family_id].append(sample_idx)
            
            self.families = list(self.family_to_samples.keys())
            print(f"Found {len(self.families)} families in {dataset.split} split with {len(self.dataset)} total samples")
            
            # Adjust batch size if we have fewer families than requested
            if len(self.families) < self.batch_size:
                print(f"Warning: Only {len(self.families)} families available for batch size {self.batch_size}")
                print(f"         Will sample with replacement to maintain batch size")
                self.actual_batch_size = len(self.families)  # Use available families
            else:
                self.actual_batch_size = self.batch_size
        else:
            # Initialize empty structures for testing
            self.family_to_samples = defaultdict(list)
            self.families = []
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling in distributed training."""
        self.epoch = epoch
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices with family diversity."""
        # Set random seed including epoch for different sampling each epoch
        epoch_seed = self.seed + (self.epoch * 100003)  # Large prime to avoid collisions
        random.seed(epoch_seed)
        
        # Infinite generator - keep producing batches
        while True:
            # Step 1: Sample families for this batch
            if len(self.families) >= self.batch_size:
                # Normal case: sample batch_size unique families without replacement
                selected_families = random.sample(self.families, self.batch_size)
            else:
                # Small dataset case: sample with replacement to fill batch
                selected_families = random.choices(self.families, k=self.batch_size)
            
            # Step 2: For each family, randomly select 1 sequence (with replacement)
            batch_indices = []
            for family in selected_families:
                family_samples = self.family_to_samples[family]
                sequence_idx = random.choice(family_samples)
                batch_indices.append(sequence_idx)
            
            # Yield all indices for this batch
            for idx in batch_indices:
                yield idx
    
    def __len__(self) -> int:
        """Return a reasonable estimate of samples per epoch.
        
        Since this is an infinite sampler, we return a value that represents
        approximately one pass through all families with the current batch size.
        """
        # Estimate: number of batches to see each family at least once
        if len(self.families) >= self.batch_size:
            batches_per_epoch = len(self.families) 
        else:
            # For small datasets, estimate based on unique family coverage
            batches_per_epoch = max(len(self.families), 10)  # At least 10 batches
        return batches_per_epoch * self.batch_size


class DistributedSimpleFamilyDiverseSampler:
    """Distributed version of SimpleFamilyDiverseSampler for multi-GPU training.
    
    Each rank runs an independent SimpleFamilyDiverseSampler with rank-specific seeding
    to ensure different sampling patterns across ranks while maintaining reproducibility.
    """
    
    def __init__(self, dataset: RNAMultimodalDataset, num_replicas: int, rank: int,
                 batch_size: int, family_lookup_path: str, shuffle: bool = True, 
                 seed: int = 42, drop_last: bool = False):
        """
        Initialize distributed simple family diverse sampler.
        
        Args:
            dataset: RNAMultimodalDataset instance
            num_replicas: Number of processes participating in distributed training
            rank: Rank of the current process
            batch_size: Batch size
            family_lookup_path: Path to family lookup JSON file
            shuffle: Whether to shuffle (kept for compatibility)
            seed: Random seed for reproducible sampling
            drop_last: Whether to drop the last incomplete batch (unused for infinite sampler)
        """
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        
        # Create rank-specific seed to ensure different sampling across ranks
        rank_seed = seed + rank * 97
        
        # Create the underlying simple sampler with rank-specific seed
        self.simple_sampler = SimpleFamilyDiverseSampler(
            dataset=dataset,
            batch_size=batch_size,
            family_lookup_path=family_lookup_path,
            shuffle=shuffle,
            seed=rank_seed
        )
        
        print(f"Initialized DistributedSimpleFamilyDiverseSampler for rank {rank}/{num_replicas}")
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling in distributed training."""
        self.simple_sampler.set_epoch(epoch)
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices with family diversity for this rank."""
        return iter(self.simple_sampler)
    
    def __len__(self) -> int:
        """Return the length of the underlying simple sampler."""
        return len(self.simple_sampler)


def _process_split_chunk(chunk_data):
    """
    Process a chunk of family data for multiprocessing during split creation.
    
    Args:
        chunk_data: Tuple of (start_idx, end_idx, rfam_ids_chunk, family_sets)
        
    Returns:
        Dictionary with train/val/test indices for this chunk
    """
    start_idx, end_idx, rfam_ids_chunk, family_sets = chunk_data
    train_families, val_families, test_families = family_sets
    
    chunk_results = {
        'train_indices': [],
        'val_indices': [],
        'test_indices': []
    }
    
    for local_idx, family_id in enumerate(rfam_ids_chunk):
        global_idx = start_idx + local_idx
        
        if family_id in train_families:
            chunk_results['train_indices'].append(global_idx)
        elif family_id in val_families:
            chunk_results['val_indices'].append(global_idx)
        elif family_id in test_families:
            chunk_results['test_indices'].append(global_idx)
    
    return chunk_results


def create_family_splits(h5_file_path: str, config: Dict, save_splits: bool = True, splits_dir: str = None, seed: int = 42, num_workers: int = None) -> Tuple[List[int], List[int], List[int]]:
    """
    Create train/val/test splits based on Rfam families.
    
    Args:
        h5_file_path: Path to HDF5 file
        config: Configuration dictionary
        save_splits: Whether to save splits to disk
        splits_dir: Directory to save splits
        seed: Random seed for reproducible splits
        num_workers: Number of worker processes (default: CPU count)
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    print("Creating family-based splits...")
    print(f"Using random seed: {seed}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(os.cpu_count(), 8)  # Cap at 8 to avoid too many processes
    
    print(f"Using {num_workers} worker processes")
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Load all rfam_ids first
    print("Loading family data...")
    with h5py.File(h5_file_path, 'r') as f:
        rfam_ids = f['rfam_id'][:]
        rfam_ids = [rid.decode('utf-8') for rid in rfam_ids]
    
    total_sequences = len(rfam_ids)
    print(f"Processing {total_sequences} sequences...")
    
    # Get unique families
    unique_families = list(set(rfam_ids))
    print(f"Found {len(unique_families)} unique Rfam families")
    
    # Shuffle families
    random.shuffle(unique_families)
    
    # Split families
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    
    n_families = len(unique_families)
    n_train = int(n_families * train_split)
    n_val = int(n_families * val_split)
    
    train_families = set(unique_families[:n_train])
    val_families = set(unique_families[n_train:n_train + n_val])
    test_families = set(unique_families[n_train + n_val:])
    
    print(f"Split families: Train={len(train_families)}, Val={len(val_families)}, Test={len(test_families)}")
    
    # Split data into chunks for multiprocessing
    chunk_size = max(1, total_sequences // num_workers)
    chunks = []
    
    for i in range(0, total_sequences, chunk_size):
        end_idx = min(i + chunk_size, total_sequences)
        chunk_data = (i, end_idx, rfam_ids[i:end_idx], (train_families, val_families, test_families))
        chunks.append(chunk_data)
    
    print(f"Split data into {len(chunks)} chunks")
    
    # Process chunks with multiprocessing
    train_indices = []
    val_indices = []
    test_indices = []
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use tqdm for progress bar
        results = list(tqdm(
            pool.imap(_process_split_chunk, chunks),
            total=len(chunks),
            desc="Processing split chunks",
            unit="chunk"
        ))
    
    # Merge results from all chunks
    print("Merging results from all chunks...")
    for chunk_result in results:
        train_indices.extend(chunk_result['train_indices'])
        val_indices.extend(chunk_result['val_indices'])
        test_indices.extend(chunk_result['test_indices'])
    
    print(f"Split samples: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # Save splits if requested
    if save_splits:
        # Determine splits directory - use same directory as HDF5 file if not specified
        if splits_dir is None:
            splits_dir = os.path.dirname(h5_file_path)
        
        Path(splits_dir).mkdir(parents=True, exist_ok=True)
        
        splits_data = {
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'train_families': list(train_families),
            'val_families': list(val_families),
            'test_families': list(test_families),
            'config': config,
            'seed': seed
        }
        
        splits_path = Path(splits_dir) / "family_splits.json"
        with open(splits_path, 'w') as f:
            json.dump(splits_data, f, indent=2)
        
        print(f"Saved splits to {splits_path}")
    
    return train_indices, val_indices, test_indices


def _process_family_chunk(chunk_data):
    """
    Process a chunk of family data for multiprocessing.
    
    Args:
        chunk_data: Tuple of (start_idx, end_idx, rfam_ids_chunk)
        
    Returns:
        Dictionary mapping family_id -> list of sequence indices for this chunk
    """
    start_idx, end_idx, rfam_ids_chunk = chunk_data
    family_lookup = defaultdict(list)
    
    for local_idx, family_id in enumerate(rfam_ids_chunk):
        global_idx = start_idx + local_idx
        family_lookup[family_id].append(global_idx)
    
    return dict(family_lookup)


def create_family_lookup(h5_file_path: str, save_lookup: bool = True, num_workers: int = None) -> Dict[str, List[int]]:
    """
    Create a lookup mapping from Rfam families to sequence indices in the HDF5 file.
    
    Args:
        h5_file_path: Path to HDF5 file
        save_lookup: Whether to save lookup to disk
        num_workers: Number of worker processes (default: CPU count)
        
    Returns:
        Dictionary mapping family_id -> list of sequence indices
    """
    print("Creating family lookup mapping...")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(os.cpu_count(), 8)  # Cap at 8 to avoid too many processes
    
    print(f"Using {num_workers} worker processes")
    
    # Load all rfam_ids first
    with h5py.File(h5_file_path, 'r') as f:
        rfam_ids = f['rfam_id'][:]
        rfam_ids = [rid.decode('utf-8') for rid in rfam_ids]
    
    total_sequences = len(rfam_ids)
    print(f"Processing {total_sequences} sequences...")
    
    # Split data into chunks for multiprocessing
    chunk_size = max(1, total_sequences // num_workers)
    chunks = []
    
    for i in range(0, total_sequences, chunk_size):
        end_idx = min(i + chunk_size, total_sequences)
        chunk_data = (i, end_idx, rfam_ids[i:end_idx])
        chunks.append(chunk_data)
    
    print(f"Split data into {len(chunks)} chunks")
    
    # Process chunks with multiprocessing
    family_lookup = defaultdict(list)
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use tqdm for progress bar
        results = list(tqdm(
            pool.imap(_process_family_chunk, chunks),
            total=len(chunks),
            desc="Processing family chunks",
            unit="chunk"
        ))
    
    # Merge results from all chunks
    print("Merging results from all chunks...")
    for chunk_result in results:
        for family_id, indices in chunk_result.items():
            family_lookup[family_id].extend(indices)
    
    # Convert defaultdict to regular dict
    family_lookup = dict(family_lookup)
    
    print(f"Created lookup for {len(family_lookup)} families")
    print(f"Total sequences: {sum(len(indices) for indices in family_lookup.values())}")
    
    # Save lookup file
    if save_lookup:
        h5_dir = os.path.dirname(h5_file_path)
        lookup_path = os.path.join(h5_dir, 'family_lookup.json')
        
        with open(lookup_path, 'w') as f:
            json.dump(family_lookup, f, indent=2)
        
        print(f"Saved family lookup to: {lookup_path}")
    
    return family_lookup


def load_family_lookup(h5_file_path: str) -> Dict[str, List[int]]:
    """
    Load family lookup from disk.
    
    Args:
        h5_file_path: Path to HDF5 file (used to determine lookup file location)
        
    Returns:
        Dictionary mapping family_id -> list of sequence indices
    """
    h5_dir = os.path.dirname(h5_file_path)
    lookup_path = os.path.join(h5_dir, 'family_lookup.json')
    
    if not os.path.exists(lookup_path):
        print(f"Family lookup not found at {lookup_path}, creating...")
        return create_family_lookup(h5_file_path, save_lookup=True)
    
    with open(lookup_path, 'r') as f:
        family_lookup = json.load(f)
    
    print(f"Loaded family lookup with {len(family_lookup)} families")
    return family_lookup


def load_family_splits(splits_path: str) -> Tuple[List[int], List[int], List[int]]:
    """
    Load pre-computed family splits.
    
    Args:
        splits_path: Path to splits JSON file
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    with open(splits_path, 'r') as f:
        splits_data = json.load(f)
    
    return splits_data['train_indices'], splits_data['val_indices'], splits_data['test_indices']


def create_data_loaders(config_path: str, h5_file_path: str, vocabulary_path: str, family_lookup_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training.
    
    Args:
        config_path: Path to config file
        h5_file_path: Path to HDF5 file
        vocabulary_path: Path to vocabulary file
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print("Creating data loaders...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get file paths from config
    family_splits_file = config['data_files']['family_splits_file']
    family_lookup_file = config['data_files']['family_lookup_file']
    print(f"Using family splits file: {family_splits_file}")
    print(f"Using family lookup file: {family_lookup_file}")
    
    # Check if splits file already exists
    if os.path.exists(family_splits_file):
        print(f"Found existing splits file: {family_splits_file}")
        print("Loading existing splits...")
        train_indices, val_indices, test_indices = load_family_splits(family_splits_file)
        print(f"✅ Loaded existing splits: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    else:
        print(f"No existing splits file found. Creating new splits...")
        splits_dir = os.path.dirname(family_splits_file)
        train_indices, val_indices, test_indices = create_family_splits(h5_file_path, config, save_splits=True, splits_dir=splits_dir)
    
    # Check if using all data for training (production mode)
    use_all_data_for_training = config['data'].get('use_all_data_for_training', False)
    if use_all_data_for_training:
        # Combine all indices for training, but keep original val/test for evaluation
        original_train_indices = train_indices.copy()
        train_indices = train_indices + val_indices + test_indices
        print(f"🚀 Production mode: Using ALL data for training ({len(train_indices)} samples)")
        print(f"   Original train: {len(original_train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        print(f"   Note: Validation and test sets remain unchanged for consistent evaluation")
    
    # Create datasets with on-the-fly loading
    print("Creating datasets with on-the-fly loading...")
    train_dataset = RNAMultimodalDataset(h5_file_path, 'train', train_indices, config, vocabulary_path)
    val_dataset = RNAMultimodalDataset(h5_file_path, 'val', val_indices, config, vocabulary_path)
    test_dataset = RNAMultimodalDataset(h5_file_path, 'test', test_indices, config, vocabulary_path)
    print(f"✅ Created datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Get data loading parameters from config
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    pin_memory = config['data']['pin_memory']
    
    print(f"Data loading parameters: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")
    
    # Get random seed from config
    random_seed = config.get('random_seed', 42)
    if random_seed == 'random':
        import time
        random_seed = int(time.time() * 1000) % (2**32)
        print(f"Using random seed: {random_seed}")
    else:
        random_seed = int(random_seed)
        print(f"Using seed from config: {random_seed}")
    
    # Create samplers with family diversity
    print("Creating samplers...")
    train_sampler = FamilyDiverseSampler(train_dataset, batch_size, family_lookup_file, seed=random_seed)
    val_sampler = FamilyDiverseSampler(val_dataset, batch_size, family_lookup_file, seed=random_seed)  # Use diverse sampling for validation
    test_sampler = FamilyDiverseSampler(test_dataset, batch_size, family_lookup_file, seed=random_seed)  # Use diverse sampling for testing
    print("✅ Created samplers")
    
    # Load vocabulary for collate function
    print("Loading vocabulary...")
    with open(vocabulary_path, 'r') as f:
        vocabulary = json.load(f)
    print(f"✅ Loaded vocabulary with {len(vocabulary.get('global_to_local', {}))} tokens")

    # Use functools.partial to bind config and vocabulary to collate_fn
    collate_train = functools.partial(collate_fn, config=config, vocabulary=vocabulary, split='train')
    collate_val = functools.partial(collate_fn, config=config, vocabulary=vocabulary, split='val')
    collate_test = functools.partial(collate_fn, config=config, vocabulary=vocabulary, split='test')

    # Get dataloader config from optimizations section
    dataloader_config = config.get('optimizations', {}).get('dataloader', {})
    persistent_workers = dataloader_config.get('persistent_workers', False)
    prefetch_factor = dataloader_config.get('prefetch_factor', 2)
    drop_last = dataloader_config.get('drop_last', True)
    timeout = dataloader_config.get('timeout', 0)
    
    print(f"DataLoader config: persistent_workers={persistent_workers}, prefetch_factor={prefetch_factor}, drop_last={drop_last}, timeout={timeout}")
    
    # Create data loaders with config-driven optimizations
    print("Creating data loaders with on-the-fly loading...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_train,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        timeout=timeout
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_val,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        timeout=timeout
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_test,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        timeout=timeout
    )

    print("✅ Created all data loaders")
    return train_loader, val_loader, test_loader


# Global variables for collate function to avoid pickle issues
_GLOBAL_CONFIG = None
_GLOBAL_VOCABULARY = None

def set_global_collate_vars(config, vocabulary):
    """Set global variables for collate function."""
    global _GLOBAL_CONFIG, _GLOBAL_VOCABULARY
    _GLOBAL_CONFIG = config
    _GLOBAL_VOCABULARY = vocabulary

def apply_mlm_modality_dropout(input_ids, modality_type_ids, config, vocabulary, mask_seed_base=None, sample_keys=None):
    """
    Apply modality dropout for MLM phase - randomly drop entire modalities by replacing with [DROPPED].
    This happens BEFORE span masking.
    
    Args:
        input_ids: (batch, seq_len) torch.LongTensor
        modality_type_ids: (batch, seq_len) torch.LongTensor - modality IDs for each position
        config: config dict (must contain ['training']['MLM_masking']['modality_dropout_rates'])
        vocabulary: loaded vocab dict (must contain 'special_tokens')
        mask_seed_base: Optional seed for deterministic dropout (for stale validation)
        sample_keys: Optional list of sample keys for deterministic dropout (for stale validation)
    Returns:
        input_ids_with_dropout, dropout_info (torch.LongTensor, Dict)
    """
    # Get modality dropout configuration
    mlm_config = config['training']['MLM_masking']
    modality_dropout_rates = mlm_config.get('modality_dropout_rates', {})
    
    # If no modality dropout configured, return unchanged
    if not modality_dropout_rates:
        return input_ids.clone(), {'dropped_modalities': [], 'total_content_tokens_dropped': 0}
    
    dropped_token_id = vocabulary['special_tokens']['[DROPPED]']
    pad_token_id = vocabulary['special_tokens']['[PAD]']
    special_token_ids = set(vocabulary['special_tokens'].values())
    
    input_ids = input_ids.clone()
    batch_size, seq_len = input_ids.shape
    
    # Track dropout statistics
    batch_dropped_modalities = []
    total_content_tokens_dropped = 0
    
    # Modality mapping
    modality_map = {0: 'sequence', 1: 'ss', 2: 'consensus', 3: 'go_terms'}
    conditioning_modalities = ['ss', 'consensus', 'go_terms']  # sequence is never dropped in MLM
    
    # Set up deterministic RNG if seed and sample keys are provided
    if mask_seed_base is not None and sample_keys is not None:
        rng = random.Random(mask_seed_base)
    
    for b in range(batch_size):
        sample_dropped_mods = []
        
        # Optional deterministic seeding per sample
        if mask_seed_base is not None and sample_keys is not None and b < len(sample_keys):
            torch.manual_seed(int(mask_seed_base) ^ int(sample_keys[b]))
        
        for mod_name in conditioning_modalities:
            if mod_name not in modality_dropout_rates:
                continue
                
            dropout_rate = modality_dropout_rates[mod_name]
            
            # Decide whether to drop this modality for this sample
            if mask_seed_base is not None and sample_keys is not None:
                # Deterministic dropout
                sample_key = sample_keys[b] if b < len(sample_keys) else b
                rng.seed(mask_seed_base + sample_key + hash(mod_name))
                should_drop = rng.random() < dropout_rate
            else:
                # Random dropout
                should_drop = torch.rand(1).item() < dropout_rate
            
            if should_drop:
                # Find modality ID
                mod_id = None
                for mid, mname in modality_map.items():
                    if mname == mod_name:
                        mod_id = mid
                        break
                
                if mod_id is not None:
                    # Create mask for this modality's content tokens (exclude special tokens)
                    modality_mask = (modality_type_ids[b] == mod_id)
                    content_mask = modality_mask & (input_ids[b] != pad_token_id)
                    
                    # Further exclude all special tokens
                    for special_id in special_token_ids:
                        content_mask = content_mask & (input_ids[b] != special_id)
                    
                    # Apply dropout to content tokens only
                    content_positions = content_mask.nonzero(as_tuple=True)[0]
                    if len(content_positions) > 0:
                        input_ids[b, content_positions] = dropped_token_id
                        sample_dropped_mods.append(mod_name)
                        total_content_tokens_dropped += len(content_positions)
                        # Clear intermediate tensors to prevent memory buildup
                        del content_positions
        
        batch_dropped_modalities.append(sample_dropped_mods)
    
    dropout_info = {
        'dropped_modalities': batch_dropped_modalities,
        'total_content_tokens_dropped': total_content_tokens_dropped
    }
    
    return input_ids, dropout_info


def apply_mlm_masking(input_ids, config, vocabulary, epoch=0, total_epochs=1, mask_seed_base=None, sample_keys=None):
    """
    Apply BERT/ModernBERT-style MLM masking to a batch of input_ids.
    Supports both random masking and span-based masking.
    
    NOTE: This function should be called AFTER apply_mlm_modality_dropout.
    [DROPPED] tokens are excluded from masking candidates.
    
    Args:
        input_ids: (batch, seq_len) torch.LongTensor
        config: config dict (must contain ['training']['MLM_masking']['rate'])
        vocabulary: loaded vocab dict (must contain 'special_tokens')
        epoch: current epoch (ignored for static)
        total_epochs: total epochs (ignored for static)
        mask_seed_base: Optional seed for deterministic masking (for stale validation)
        sample_keys: Optional list of sample keys for deterministic masking (for stale validation)
    Returns:
        masked_input_ids, labels (torch.LongTensor)
    """
    mask_token_id = vocabulary['special_tokens']['[MASK]']
    pad_token_id = vocabulary['special_tokens']['[PAD]']
    dropped_token_id = vocabulary['special_tokens']['[DROPPED]']
    special_token_ids = set(vocabulary['special_tokens'].values())
    masking_rate = float(config['training']['MLM_masking']['rate'])
    
    # Get span masking configuration
    span_config = config['training']['MLM_masking'].get('span_masking', {})
    use_span_masking = span_config.get('enabled', False)
    
    input_ids = input_ids.clone()
    labels = input_ids.clone()
    batch_size, seq_len = input_ids.shape
    
    # Set up deterministic RNG if seed and sample keys are provided
    if mask_seed_base is not None and sample_keys is not None:
        # Create deterministic RNG for this batch
        rng = random.Random(mask_seed_base)
    
    for i in range(batch_size):
        # Get candidate indices (EXCLUDE ALL special tokens, padding, AND dropped tokens)
        candidate_idxs = [j for j in range(seq_len)
                          if (input_ids[i, j].item() not in special_token_ids and 
                              input_ids[i, j].item() != pad_token_id and 
                              input_ids[i, j].item() != dropped_token_id)]
        
        if not candidate_idxs:
            continue
            
        num_to_mask = max(1, int(round(len(candidate_idxs) * masking_rate)))
        
        if use_span_masking:
            # Span-based masking
            if mask_seed_base is not None and sample_keys is not None:
                # Deterministic span masking
                mask_idxs = _generate_span_masks_safe_deterministic(
                    candidate_idxs, num_to_mask, span_config, input_ids[i], vocabulary, 
                    rng, sample_keys[i] if i < len(sample_keys) else i
                )
            else:
                # Random span masking
                mask_idxs = _generate_span_masks_safe(
                    candidate_idxs, num_to_mask, span_config, input_ids[i], vocabulary
                )
        else:
            # Random masking
            if mask_seed_base is not None and sample_keys is not None:
                # Deterministic random masking
                sample_key = sample_keys[i] if i < len(sample_keys) else i
                rng.seed(mask_seed_base + sample_key)
                mask_idxs = rng.sample(candidate_idxs, num_to_mask) if len(candidate_idxs) >= num_to_mask else candidate_idxs
            else:
                # Original random masking
                mask_idxs = random.sample(candidate_idxs, num_to_mask) if len(candidate_idxs) >= num_to_mask else candidate_idxs
        
        # Apply BERT-style masking strategy
        for idx in mask_idxs:
            if mask_seed_base is not None and sample_keys is not None:
                # Deterministic masking strategy
                sample_key = sample_keys[i] if i < len(sample_keys) else i
                rng.seed(mask_seed_base + sample_key + idx)  # Include position for uniqueness
                prob = rng.random()
            else:
                prob = random.random()
                
            if prob < 0.8:
                input_ids[i, idx] = mask_token_id
            elif prob < 0.9:
                # Replace with random token (not special)
                all_ids = [tid for tid in range(len(vocabulary['global_to_local'])) if tid not in special_token_ids]
                if mask_seed_base is not None and sample_keys is not None:
                    input_ids[i, idx] = rng.choice(all_ids)
                else:
                    input_ids[i, idx] = random.choice(all_ids)
            # else: leave unchanged
        
        # Set labels: masked positions get original token, others get padding_label_id
        for j in range(seq_len):
            if j not in mask_idxs or input_ids[i, j].item() == pad_token_id:
                labels[i, j] = config['data']['padding_label_id']
    
    return input_ids, labels


def _generate_span_masks_safe(candidate_idxs, num_to_mask, span_config, sequence_tokens, vocabulary):
    """
    Generate span-based masks using geometric distribution for span lengths.
    GUARANTEES that no special tokens are ever masked.
    
    Args:
        candidate_idxs: List of candidate indices (already filtered to exclude special tokens)
        num_to_mask: Total number of tokens to mask
        span_config: Span masking configuration
        sequence_tokens: Full sequence tokens for validation
        vocabulary: Vocabulary dictionary
        
    Returns:
        List of indices to mask
    """
    min_span_length = span_config.get('min_span_length', 2)
    max_span_length = span_config.get('max_span_length', 8)
    geometric_p = span_config.get('geometric_p', 0.3)
    
    mask_idxs = []
    remaining_candidates = candidate_idxs.copy()
    
    while len(mask_idxs) < num_to_mask and remaining_candidates:
        # Sample span length from geometric distribution
        span_length = _sample_geometric_span_length(geometric_p, min_span_length, max_span_length)
        
        # Find valid span positions
        valid_spans = _find_valid_spans_safe(
            remaining_candidates, span_length, sequence_tokens, vocabulary
        )
        
        if not valid_spans:
            # If no valid spans found, fall back to single token masking
            if remaining_candidates:
                mask_idxs.append(remaining_candidates.pop(0))
            continue
        
        # Randomly select a span
        selected_span = random.choice(valid_spans)
        
        # Add span indices to mask list
        for idx in selected_span:
            if idx in remaining_candidates:
                mask_idxs.append(idx)
                remaining_candidates.remove(idx)
    
    # If we still need more masks, add individual tokens
    while len(mask_idxs) < num_to_mask and remaining_candidates:
        mask_idxs.append(remaining_candidates.pop(0))
    
    return mask_idxs


def _generate_span_masks_safe_deterministic(candidate_idxs, num_to_mask, span_config, sequence_tokens, vocabulary, rng, sample_key):
    """
    Generate span-based masks using geometric distribution for span lengths.
    Deterministic version for stale validation.
    GUARANTEES that no special tokens are ever masked.
    
    Args:
        candidate_idxs: List of candidate indices (already filtered to exclude special tokens)
        num_to_mask: Total number of tokens to mask
        span_config: Span masking configuration
        sequence_tokens: Full sequence tokens for validation
        vocabulary: Vocabulary dictionary
        rng: Deterministic random number generator
        sample_key: Key for this sample for deterministic seeding
        
    Returns:
        List of indices to mask
    """
    min_span_length = span_config.get('min_span_length', 2)
    max_span_length = span_config.get('max_span_length', 8)
    geometric_p = span_config.get('geometric_p', 0.3)
    
    mask_idxs = []
    remaining_candidates = candidate_idxs.copy()
    
    # Seed the RNG for this sample
    rng.seed(sample_key)
    
    while len(mask_idxs) < num_to_mask and remaining_candidates:
        # Sample span length from geometric distribution (deterministic)
        span_length = _sample_geometric_span_length_deterministic(geometric_p, min_span_length, max_span_length, rng)
        
        # Find valid span positions
        valid_spans = _find_valid_spans_safe(
            remaining_candidates, span_length, sequence_tokens, vocabulary
        )
        
        if not valid_spans:
            # If no valid spans found, fall back to single token masking
            if remaining_candidates:
                mask_idxs.append(remaining_candidates.pop(0))
            continue
        
        # Deterministically select a span
        selected_span = rng.choice(valid_spans)
        
        # Add span indices to mask list
        for idx in selected_span:
            if idx in remaining_candidates:
                mask_idxs.append(idx)
                remaining_candidates.remove(idx)
    
    # If we still need more masks, add individual tokens
    while len(mask_idxs) < num_to_mask and remaining_candidates:
        mask_idxs.append(remaining_candidates.pop(0))
    
    return mask_idxs


def _sample_geometric_span_length_deterministic(p, min_length, max_length, rng):
    """
    Sample span length from truncated geometric distribution (deterministic version).
    
    Args:
        p: Geometric distribution parameter (0 < p < 1)
        min_length: Minimum span length
        max_length: Maximum span length
        rng: Deterministic random number generator
        
    Returns:
        Sampled span length
    """
    # Sample from geometric distribution using deterministic RNG
    # Use inverse transform method for geometric distribution
    u = rng.random()
    span_length = min_length + int(np.log(1 - u) / np.log(1 - p)) - 1
    
    # Truncate to max_length
    return min(span_length, max_length)


def _sample_geometric_span_length(p, min_length, max_length):
    """
    Sample span length from truncated geometric distribution.
    
    Args:
        p: Geometric distribution parameter (0 < p < 1)
        min_length: Minimum span length
        max_length: Maximum span length
        
    Returns:
        Sampled span length
    """
    # Sample from geometric distribution
    span_length = min_length + np.random.geometric(p) - 1
    
    # Truncate to max_length
    return min(span_length, max_length)


def _find_valid_spans_safe(candidate_idxs, span_length, sequence_tokens, vocabulary):
    """
    Find all valid spans of given length that are contiguous in the sequence.
    GUARANTEES that no special tokens are included.
    
    Args:
        candidate_idxs: List of candidate indices (already filtered to exclude special tokens)
        span_length: Length of spans to find
        sequence_tokens: Full sequence tokens for validation
        vocabulary: Vocabulary dictionary
        
    Returns:
        List of valid spans (each span is a list of indices)
    """
    if len(candidate_idxs) < span_length:
        return []
    
    valid_spans = []
    candidate_set = set(candidate_idxs)
    special_token_ids = set(vocabulary['special_tokens'].values())
    
    # Sort candidates for efficient span finding
    sorted_candidates = sorted(candidate_idxs)
    
    for i in range(len(sorted_candidates) - span_length + 1):
        # Check if we can form a span starting at this position
        potential_span = sorted_candidates[i:i + span_length]
        
        # Check if all indices in span are candidates
        if not all(idx in candidate_set for idx in potential_span):
            continue
        
        # Check if span is contiguous in the original sequence
        if potential_span[-1] - potential_span[0] + 1 != span_length:
            continue
        
        # CRITICAL: Double-check that no special tokens are in the span
        # This is the safety check to ensure we never mask special tokens
        span_contains_special = False
        for idx in potential_span:
            if sequence_tokens[idx].item() in special_token_ids:
                span_contains_special = True
                break
        
        if span_contains_special:
            continue
        
        # Additional check: ensure all positions between start and end are actually candidates
        # This prevents gaps in the span
        for pos in range(potential_span[0], potential_span[-1] + 1):
            if pos not in candidate_set:
                # This creates a gap, so this span is invalid
                break
        else:
            # No gaps found and no special tokens, this is a valid span
            valid_spans.append(potential_span)
    
    return valid_spans


def collate_fn(batch: List[Dict[str, torch.Tensor]], config: Dict = None, vocabulary: Dict = None, epoch: int = 0, total_epochs: int = 1, split: str = 'train') -> Dict[str, torch.Tensor]:
    # Use global variables if not provided
    if config is None:
        config = _GLOBAL_CONFIG
    if vocabulary is None:
        vocabulary = _GLOBAL_VOCABULARY
    """
    Collate function for variable-length sequences with modality-specific padding.
    Now also returns modality_type_ids: a tensor of shape (batch, seq_len) with values:
      0 = sequence, 1 = ss, 2 = consensus, 3 = go_terms
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
    if vocabulary is None:
        pad_token_id = 3  # Default to 3 if not loaded
    else:
        pad_token_id = vocabulary['special_tokens']['[PAD]']
    # Get padding label ID
    if config is None:
        padding_label_id = -100
    else:
        padding_label_id = config['data']['padding_label_id']
    # Determine target batch length - use natural maximum like diffusion mode
    sample_lengths = [len(sample['input_ids']) for sample in valid_batch]
    max_len = max(sample_lengths)
    target_len = max_len
    
    # Apply max_position_embeddings limit (but don't truncate unnecessarily)
    if config is not None:
        max_position_embeddings = config['data']['max_position_embeddings']
        if target_len > max_position_embeddings:
            target_len = max_position_embeddings
    # Pad sequences with [PAD] token
    input_ids = []
    attention_mask = []
    labels = []
    modality_type_ids = []
    # Precompute modality boundaries
    special = vocabulary['special_tokens']
    for sample in valid_batch:
        current_len = len(sample['input_ids'])
        # Truncate if current length exceeds target_len
        if current_len > target_len:
            ids_list = sample['input_ids'][:target_len]
            attn_list = sample['attention_mask'][:target_len]
            labels_list = sample['labels'][:target_len]
        else:
            ids_list = sample['input_ids']
            attn_list = sample['attention_mask']
            labels_list = sample['labels']
        padding_needed = target_len - len(ids_list)
        # Build modality_type_ids for this sample
        ids = ids_list.tolist()
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
            padded_input_ids = torch.cat([ids_list, padding_tokens])
            padded_attention_mask = torch.cat([attn_list, padding_attention])
            padded_labels = torch.cat([labels_list, padding_labels])
            padded_modality_type_ids = torch.tensor(mod_ids + [-1]*padding_needed, dtype=torch.long)
        else:
            padded_input_ids = ids_list
            padded_attention_mask = attn_list
            padded_labels = labels_list
            padded_modality_type_ids = torch.tensor(mod_ids, dtype=torch.long)
        input_ids.append(padded_input_ids)
        attention_mask.append(padded_attention_mask)
        labels.append(padded_labels)
        modality_type_ids.append(padded_modality_type_ids)
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    labels = torch.stack(labels)
    modality_type_ids = torch.stack(modality_type_ids)
    # --- Save original input_ids before any modifications ---
    original_input_ids = input_ids.clone()
    
    # --- Step 1: Apply MLM modality dropout (if configured) ---
    # Stale validation: deterministic dropout and masking using resolved seed and stable sample keys
    if config.get('training', {}).get('stale_validation', False) and split == 'val':
        seed_base = int(config.get('_resolved_random_seed', 1337))
        sample_keys = [int(sample.get('data_idx', i)) for i, sample in enumerate(valid_batch)]
        
        # Apply modality dropout first
        input_ids_with_dropout, dropout_info = apply_mlm_modality_dropout(
            input_ids, modality_type_ids, config, vocabulary,
            mask_seed_base=seed_base, sample_keys=sample_keys
        )
        
        # Then apply MLM masking to the dropout-modified input_ids
        masked_input_ids, masked_labels = apply_mlm_masking(
            input_ids_with_dropout, config, vocabulary, epoch=epoch, total_epochs=total_epochs,
            mask_seed_base=seed_base, sample_keys=sample_keys
        )
    else:
        # Apply modality dropout first
        input_ids_with_dropout, dropout_info = apply_mlm_modality_dropout(
            input_ids, modality_type_ids, config, vocabulary
        )
        
        # Then apply MLM masking to the dropout-modified input_ids
        masked_input_ids, masked_labels = apply_mlm_masking(input_ids_with_dropout, config, vocabulary, epoch=epoch, total_epochs=total_epochs)
    # Collect family_ids (rfam_id) for each sample
    family_ids = [sample['rfam_id'] for sample in valid_batch]
    return {
        'input_ids': masked_input_ids,
        'attention_mask': attention_mask,
        'labels': masked_labels,
        'original_input_ids': original_input_ids,
        'modality_type_ids': modality_type_ids,
        'family_ids': family_ids,
        'data_indices': torch.tensor([int(sample.get('data_idx', i)) for i, sample in enumerate(valid_batch)], dtype=torch.long),
        'mlm_dropout_info': dropout_info
    }


def _get_modality_boundaries(token_ids: torch.Tensor, vocabulary: Dict) -> List[Tuple[str, int, int]]:
    """
    Get modality boundaries in a token sequence.
    
    Args:
        token_ids: Token IDs tensor
        vocabulary: Vocabulary dictionary
        
    Returns:
        List of (modality, start_pos, end_pos) tuples
    """
    boundaries = []
    current_modality = None
    start_pos = 0
    
    for i, token_id in enumerate(token_ids):
        token, modality = _get_token_and_modality(token_id, vocabulary)
        
        # Check for modality separator tokens
        if token in ['[SEQ_START]', '[STRUCT_START]', '[CONS_START]', '[GO_START]']:
            # End previous modality
            if current_modality and i > start_pos:
                boundaries.append((current_modality, start_pos, i))
            
            # Start new modality
            if token == '[SEQ_START]':
                current_modality = 'sequence'
            elif token == '[STRUCT_START]':
                current_modality = 'ss'
            elif token == '[CONS_START]':
                current_modality = 'consensus'
            elif token == '[GO_START]':
                current_modality = 'go_terms'
            
            start_pos = i + 1
    
    # Add final modality
    if current_modality and len(token_ids) > start_pos:
        boundaries.append((current_modality, start_pos, len(token_ids)))
    
    return boundaries


def _get_token_and_modality(token_id: int, vocabulary: Dict) -> Tuple[str, str]:
    """
    Get token string and modality for a token ID.
    
    Args:
        token_id: Token ID
        vocabulary: Vocabulary dictionary
        
    Returns:
        Tuple of (token, modality)
    """
    if vocabulary is None:
        return f"token_{token_id}", "unknown"
    
    # Check special tokens first
    for token, special_id in vocabulary['special_tokens'].items():
        if special_id == token_id:
            return token, 'special'
    
    # Check modality vocabularies
    for modality, tokens in vocabulary['modality_vocabs'].items():
        offset = vocabulary['modality_offsets'][modality]
        if offset <= token_id < offset + len(tokens):
            local_id = token_id - offset
            return tokens[local_id], modality
    
    return f"unknown_{token_id}", "unknown"


def _get_max_modality_length(modality: str, max_len: int, vocabulary: Dict) -> int:
    """
    Get maximum length for a specific modality.
    
    Args:
        modality: Modality name
        max_len: Overall max length
        vocabulary: Vocabulary dictionary
        
    Returns:
        Maximum length for this modality
    """
    # For now, use a simple heuristic
    if modality == 'sequence':
        return max_len // 4  # Roughly 1/4 of total length
    elif modality == 'ss':
        return max_len // 4
    elif modality == 'consensus':
        return max_len // 4
    elif modality == 'go_terms':
        return max_len // 4
    else:
        return max_len // 4


def main():
    """Test the data loading pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RNA Multimodal Training Data Preparation")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--h5-file", type=str, default=None,
                       help="Path to HDF5 data file (overrides config)")
    parser.add_argument("--vocabulary", type=str, default=None,
                       help="Path to vocabulary JSON file (overrides config)")
    parser.add_argument("--test-splits", action="store_true",
                       help="Test family-based splitting only")
    parser.add_argument("--test-loading", action="store_true",
                       help="Test data loading pipeline")
    parser.add_argument("--load-splits", type=str, default=None,
                       help="Path to existing splits JSON file")
    parser.add_argument("--save-splits", action="store_true", default=True,
                       help="Save splits to disk")
    parser.add_argument("--create-lookup", action="store_true",
                       help="Create family lookup file")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of worker processes for family lookup creation (default: CPU count)")
    parser.add_argument("--seed", type=str, default=None,
                       help="Random seed (overrides config file, use 'random' for random seed)")
    
    args = parser.parse_args()
    
    print(f"Using config: {args.config}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get file paths from config (with command line override)
    h5_file = args.h5_file if args.h5_file is not None else config['data_files']['h5_file']
    vocabulary_file = args.vocabulary if args.vocabulary is not None else config['data_files']['vocabulary_file']
    family_splits_file = config['data_files']['family_splits_file']
    family_lookup_file = config['data_files']['family_lookup_file']
    
    print(f"Using HDF5 file: {h5_file}")
    print(f"Using vocabulary: {vocabulary_file}")
    print(f"Using family splits file: {family_splits_file}")
    print(f"Using family lookup file: {family_lookup_file}")
    
    print(f"Loaded config: {config}")
    
    # Determine seed: command line argument overrides config file
    if args.seed is not None:
        seed = args.seed
    else:
        seed = config.get('random_seed', 42)
    
    # Handle 'random' seed
    if seed == 'random':
        import time
        seed = int(time.time() * 1000) % (2**32)  # Use current time as seed
        print(f"Using random seed: {seed}")
    else:
        seed = int(seed)  # Convert to int
        print(f"Using seed from config: {seed}")
    
    if args.create_lookup:
        print("\nCreating family lookup file...")
        family_lookup = create_family_lookup(h5_file, save_lookup=True, num_workers=args.num_workers)
        print(f"✅ Family lookup creation completed!")
        print(f"   Found {len(family_lookup)} families")
        print(f"   Total sequences: {sum(len(indices) for indices in family_lookup.values())}")
    
    if args.test_splits:
        print("\nTesting family-based splits...")
        if args.load_splits:
            print(f"Loading existing splits from {args.load_splits}")
            train_indices, val_indices, test_indices = load_family_splits(args.load_splits)
        else:
            # Check if splits file already exists
            if os.path.exists(family_splits_file):
                print(f"Found existing splits file: {family_splits_file}")
                print("Loading existing splits...")
                train_indices, val_indices, test_indices = load_family_splits(family_splits_file)
                print(f"✅ Loaded existing splits: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
            else:
                print(f"No existing splits file found. Creating new splits...")
                splits_dir = os.path.dirname(family_splits_file)
                train_indices, val_indices, test_indices = create_family_splits(h5_file, config, save_splits=args.save_splits, seed=seed, num_workers=args.num_workers, splits_dir=splits_dir)
        print(f"✅ Split test completed!")
    
    if args.test_loading:
        print("\nTesting data loading pipeline...")
        # Load vocabulary for debugging
        with open(vocabulary_file, 'r') as f:
            vocabulary = json.load(f)
        print(f"✅ Loaded vocabulary with {len(vocabulary.get('global_to_local', {}))} tokens")
        
        # Get [PAD] token ID from special tokens
        pad_token_id = vocabulary['special_tokens']['[PAD]']
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(args.config, h5_file, vocabulary_file, family_lookup_file)
        
        # Test a few batches
        print("\nTesting train loader...")
        print("Loading batches from train loader...")
        # Create a detailed sample for inspection
        log_lines = []
        log_lines.append("\n" + "="*80)
        log_lines.append("DETAILED SAMPLE INSPECTION")
        log_lines.append("="*80)
        
        # Get the first batch and first sample
        first_batch = next(iter(train_loader))
        sample_tokens = first_batch['input_ids'][0].tolist()
        
        # Get the original raw data for this sample
        with h5py.File(h5_file, 'r') as f:
            # Get the first sample from the dataset
            first_sample_idx = 0  # This will be the first sample in the batch
            sequence = f['sequence'][first_sample_idx].decode('utf-8')
            structure = f['ss'][first_sample_idx].decode('utf-8')
            consensus = f['consensus'][first_sample_idx].decode('utf-8')
            go_terms = f['go_terms'][first_sample_idx].decode('utf-8')
            rfam_id = f['rfam_id'][first_sample_idx].decode('utf-8')
        
        log_lines.append(f"\n📋 SAMPLE DETAILS:")
        log_lines.append(f"  Rfam ID: {rfam_id}")
        log_lines.append(f"  Sequence length: {len(sequence)}")
        log_lines.append(f"  Structure length: {len(structure)}")
        log_lines.append(f"  Consensus length: {len(consensus)}")
        log_lines.append(f"  GO terms: {go_terms}")
        
        log_lines.append(f"\n🔤 RAW INPUT (first 100 chars each):")
        log_lines.append(f"  Sequence:  {sequence[:100]}")
        log_lines.append(f"  Structure: {structure[:100]}")
        log_lines.append(f"  Consensus: {consensus[:100]}")
        
        log_lines.append(f"\n🎯 TOKENIZED SEQUENCE:")
        log_lines.append(f"  Total tokens: {len(sample_tokens)}")
        log_lines.append(f"  Token IDs: {sample_tokens}")
        
        log_lines.append(f"\n📖 READABLE TOKEN SEQUENCE:")
        modality_boundaries = _get_modality_boundaries(first_batch['input_ids'][0], vocabulary)
        
        # Create readable representation
        readable_tokens = []
        for i, token_id in enumerate(sample_tokens):
            token_name, modality = _get_token_and_modality(token_id, vocabulary)
            readable_tokens.append(f"{token_name}")
        
        # Show the complete readable sequence
        log_lines.append(f"  Complete sequence:")
        log_lines.append(f"    {' '.join(readable_tokens)}")
        
        log_lines.append(f"\n🏷️  MODALITY BREAKDOWN:")
        for modality, start_pos, end_pos in modality_boundaries:
            modality_tokens = sample_tokens[start_pos:end_pos]
            modality_readable = [readable_tokens[i] for i in range(start_pos, end_pos)]
            log_lines.append(f"  {modality.upper()}:")
            log_lines.append(f"    Position: {start_pos}-{end_pos} ({end_pos-start_pos} tokens)")
            log_lines.append(f"    Tokens: {' '.join(modality_readable[:20])}{'...' if len(modality_readable) > 20 else ''}")
        
        log_lines.append(f"\n🔍 TOKEN-BY-TOKEN ANALYSIS (all tokens):")
        for i, token_id in enumerate(sample_tokens):
            token_name, modality = _get_token_and_modality(token_id, vocabulary)
            log_lines.append(f"    {i:2d}: {token_id:3d} -> {token_name:12s} ({modality})")
        log_lines.append("="*80)
        
        # Print to terminal
        for line in log_lines:
            print(line)
        
        # Write to file
        out_dir = os.path.join(os.path.dirname(h5_file), "sample_tokenization_log.txt")
        with open(out_dir, 'w') as f:
            for line in log_lines:
                f.write(line + '\n')
        print(f"\n✅ Detailed sample inspection written to: {out_dir}\n")
        # Continue with regular batch testing
        for i, batch in enumerate(train_loader):
            print(f"\nBatch {i}:")
            print(f"  Input shape: {batch['input_ids'].shape}")
            print(f"  Attention mask shape: {batch['attention_mask'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            
            if i >= 2:  # Test 3 batches
                break
        
        print("\nTesting validation loader...")
        print("Loading batches from validation loader...")
        for i, batch in enumerate(val_loader):
            print(f"Val Batch {i}:")
            print(f"  Input shape: {batch['input_ids'].shape}")
            print(f"  Attention mask shape: {batch['attention_mask'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            if i >= 1:  # Test 2 validation batches
                break
        
        print("\n✅ Data loading pipeline test completed!")
    
    if not args.test_splits and not args.test_loading and not args.create_lookup:
        print("Please specify --create-lookup, --test-splits, or --test-loading to run tests")
        print("\nUsage examples:")
        print("  # Create family lookup file (uses all CPU cores)")
        print("  python data_processing/training_data_preparation.py --create-lookup --config configs/macbook_training.yaml")
        print("  # Create family lookup file with specific number of workers")
        print("  python data_processing/training_data_preparation.py --create-lookup --config configs/macbook_training.yaml --num-workers 4")
        print("  # Use seed from config file")
        print("  python data_processing/training_data_preparation.py --test-splits --config configs/macbook_training.yaml")
        print("  # Use random seed")
        print("  python data_processing/training_data_preparation.py --test-splits --config configs/macbook_training.yaml --seed random")
        print("  # Override config seed")
        print("  python data_processing/training_data_preparation.py --test-splits --config configs/macbook_training.yaml --seed 123")


if __name__ == "__main__":
    main() 