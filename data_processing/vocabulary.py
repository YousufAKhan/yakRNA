"""
vocabulary.py

A comprehensive vocabulary analysis script for RNA multimodal data.
Analyzes token distributions across all modalities in HDF5 files:
- sequence: RNA sequences (A, U, G, C, N)
- ss: Secondary structure strings
- consensus: Consensus sequences
- go_terms: GO term identifiers

Features:
- Token frequency analysis per modality
- Cross-modality token overlap analysis
- Vocabulary size calculations
- Token length distributions
- Visualization capabilities
- Export functionality for downstream use
- Unified vocabulary system with modal separators
- Offset-based token mapping
- Multimodal tokenization pipeline

Usage:
    python data_processing/vocabulary.py --input training_data/processed/bert_ready_data.h5
    python data_processing/vocabulary.py --input training_data/processed/bert_ready_data.h5 --analyze-all
    python data_processing/vocabulary.py --input training_data/processed/bert_ready_data.h5 --export-vocab
    python data_processing/vocabulary.py --input training_data/processed/bert_ready_data.h5 --build-unified-vocab
"""

import argparse
import h5py
import numpy as np
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Any, Optional, Union
from pathlib import Path
import re

# matplotlib and seaborn are imported lazily inside generate_visualizations:
# building a vocabulary should not require a plotting stack to be installed.


class UnifiedVocabulary:
    """
    Unified vocabulary for multimodal RNA BERT/diffusion models.
    Special tokens include [CLS], [SEP], [MASK], [PAD], [DROPPED], and modality separators.
    [DROPPED] is used to explicitly mark dropped modalities for conditional generation/diffusion.
    """
    
    def __init__(self, vocab_analysis_path: str = None):
        """
        Initialize unified vocabulary from analysis results or create default structure.
        
        Args:
            vocab_analysis_path: Path to vocabulary analysis JSON file
        """
        self.modalities = ['sequence', 'ss', 'consensus', 'go_terms']
        self.modality_order = ['sequence', 'ss', 'consensus', 'go_terms']  # Fixed order
        
        # Special tokens (global IDs 0-8)
        self.special_tokens = {
            '[CLS]': 0, '[SEP]': 1, '[MASK]': 2, '[PAD]': 3, '[DROPPED]': 4,
            '[SEQ_START]': 5, '[STRUCT_START]': 6, '[CONS_START]': 7, '[GO_START]': 8
        }
        # [DROPPED] is used to explicitly mark dropped modalities for conditional generation/diffusion.
        
        # Initialize modality vocabularies and offsets
        self.modality_vocabs = {}
        self.modality_offsets = {}
        self.global_to_local = {}
        self.local_to_global = {}
        
        if vocab_analysis_path:
            self.load_from_analysis(vocab_analysis_path)
        else:
            self.create_default_vocabulary()
    
    def load_from_analysis(self, vocab_analysis_path: str) -> None:
        """Load vocabulary from the unified vocabulary JSON (single, modern format)."""
        with open(vocab_analysis_path, 'r') as f:
            analysis = json.load(f)
        # Expect only the new format
        if 'modality_vocabs' in analysis and 'modality_offsets' in analysis and 'special_tokens' in analysis:
            self.modality_vocabs = analysis['modality_vocabs']
            self.modality_offsets = analysis['modality_offsets']
            self.special_tokens = analysis['special_tokens']
        else:
            raise ValueError('UnifiedVocabulary expects a JSON with modality_vocabs, modality_offsets, and special_tokens.')
        self._build_offset_mappings()
    
    def create_default_vocabulary(self) -> None:
        """Create default vocabulary structure based on known RNA modalities."""
        # Sequence tokens: A, C, G, U, N, [RNA-UNK], [SEQ-GAP]
        self.modality_vocabs['sequence'] = ['A', 'C', 'G', 'U', 'N', '[RNA-UNK]', '[SEQ-GAP]']
        # Structure tokens: ( ) . < > [ ] { } : ~ - _ , [STRUC-UNK], [STRUC-GAP]
        self.modality_vocabs['ss'] = ['(', ')', '.', '<', '>', '[', ']', '{', '}', ':', '~', '-', '_', ',', '[STRUC-UNK]', '[STRUC-GAP]']
        # Consensus tokens: A, C, G, U, a, c, g, u, ., ~, [CONS-UNK], [CONS-GAP]
        self.modality_vocabs['consensus'] = ['A', 'C', 'G', 'U', 'a', 'c', 'g', 'u', '.', '~', '[CONS-UNK]', '[CONS-GAP]']
        # GO terms: [GO-UNK], [GO-GAP] (no [GO-PAD])
        self.modality_vocabs['go_terms'] = ['[GO-UNK]', '[GO-GAP]']

        # Compute offsets after special tokens
        offset = len(self.special_tokens)
        for modality in self.modality_order:
            self.modality_offsets[modality] = offset
            offset += len(self.modality_vocabs[modality])

        # Build global_to_local and local_to_global
        self.global_to_local = {}
        self.local_to_global = {}
        # Special tokens
        for token, idx in self.special_tokens.items():
            self.global_to_local[idx] = (token, 'special')
            self.local_to_global[(token, 'special')] = idx
        # Modalities
        for modality in self.modality_order:
            for i, token in enumerate(self.modality_vocabs[modality]):
                global_id = self.modality_offsets[modality] + i
                self.global_to_local[global_id] = (token, modality)
                self.local_to_global[(modality, token)] = global_id

    def _build_offset_mappings(self) -> None:
        """Build offset mappings and bidirectional token mappings."""
        # First add special tokens to global mapping
        for token, global_id in self.special_tokens.items():
            self.global_to_local[global_id] = (token, 'special')
            self.local_to_global[('special', token)] = global_id
        
        current_offset = len(self.special_tokens)  # Start after special tokens
        
        for modality in self.modality_order:
            self.modality_offsets[modality] = current_offset
            tokens = self.modality_vocabs[modality]
            
            # Build local to global mapping
            for local_id, token in enumerate(tokens):
                global_id = current_offset + local_id
                self.local_to_global[(modality, token)] = global_id  # Only this order
                self.local_to_global[(modality, local_id)] = global_id
                self.global_to_local[global_id] = (token, modality)
            
            current_offset += len(tokens)
        
        self.vocab_size = current_offset
    
    def get_global_token_id(self, token: str, modality: str) -> int:
        """
        Convert local token + modality to global token ID.
        
        Args:
            token: Local token string
            modality: Modality name
            
        Returns:
            Global token ID
        """
        # Check special tokens first
        if token in self.special_tokens:
            return self.special_tokens[token]
        
        # Handle unknown tokens based on modality
        if modality == 'sequence':
            if token not in self.modality_vocabs[modality]:
                token = '[RNA-UNK]'  # Use RNA-specific UNK
        elif modality == 'consensus':
            if token not in self.modality_vocabs[modality]:
                token = '[CONS-UNK]'  # Use consensus-specific UNK
        elif modality == 'ss':
            if token not in self.modality_vocabs[modality]:
                token = '[STRUC-UNK]'  # Use structure-specific UNK
        elif modality == 'go_terms':
            if token not in self.modality_vocabs[modality]:
                token = '[GO-UNK]'  # Use GO-specific UNK
        
        # Check modality-specific mapping
        if (modality, token) in self.local_to_global:
            return self.local_to_global[(modality, token)]
        
        # Final fallback: raise clear error
        raise ValueError(f"Token '{token}' (modality '{modality}') could not be mapped to a global ID. Check vocabulary and input data.")
    
    def get_local_token_id(self, global_token_id: int) -> Tuple[str, str]:
        """
        Convert global token ID to local token and modality.
        
        Args:
            global_token_id: Global token ID
            
        Returns:
            Tuple of (token, modality)
        """
        if global_token_id in self.global_to_local:
            return self.global_to_local[global_token_id]
        
        # Return UNK for unknown global IDs
        return ('[UNK]', 'special')
    
    def get_modality_for_token(self, global_token_id: int) -> str:
        """
        Determine which modality a global token ID belongs to.
        
        Args:
            global_token_id: Global token ID
            
        Returns:
            Modality name
        """
        if global_token_id in self.global_to_local:
            return self.global_to_local[global_token_id][1]
        
        return 'special'
    
    def get_modality_range(self, modality: str) -> Tuple[int, int]:
        """
        Get the global token ID range for a modality.
        
        Args:
            modality: Modality name
            
        Returns:
            Tuple of (start_id, end_id)
        """
        if modality not in self.modality_offsets:
            return (0, 0)
        
        start_id = self.modality_offsets[modality]
        end_id = start_id + len(self.modality_vocabs[modality])
        return (start_id, end_id)
    
    def add_go_terms(self, go_terms: List[str]) -> None:
        """
        Add GO terms to the vocabulary and rebuild mappings.
        
        Args:
            go_terms: List of GO term strings (without GO: prefix)
        """
        # Add new GO terms to the vocabulary with GO: prefix
        for term in go_terms:
            go_term_with_prefix = f"GO:{term}" if not term.startswith('GO:') else term
            if go_term_with_prefix not in self.modality_vocabs['go_terms']:
                self.modality_vocabs['go_terms'].append(go_term_with_prefix)
        
        # Rebuild mappings
        self._build_offset_mappings()


class ModalTokenizer:
    """Multimodal tokenizer with modal separators and offset-based mapping."""
    
    def __init__(self, unified_vocab: UnifiedVocabulary):
        """
        Initialize modal tokenizer.
        
        Args:
            unified_vocab: Unified vocabulary instance
        """
        self.vocab = unified_vocab
        self.modality_order = self.vocab.modality_order
    
    def encode_multimodal(self, sequence: str = None, structure: str = None, 
                         consensus: str = None, go_terms: str = None) -> List[int]:
        """
        Tokenize multimodal data with modal separators.
        
        Args:
            sequence: RNA sequence
            structure: Secondary structure
            consensus: Consensus sequence
            go_terms: GO terms (comma-separated)
            
        Returns:
            List of global token IDs
        """
        # Validate that at least sequence is provided
        if not sequence:
            raise ValueError("At least RNA sequence must be provided")
        
        tokens = []
        
        # Add CLS token
        tokens.append(self.vocab.special_tokens['[CLS]'])
        
        # Process each modality in order
        if sequence:
            tokens.append(self.vocab.special_tokens['[SEQ_START]'])
            tokens.extend(self._tokenize_sequence(sequence, 'sequence'))
        
        if structure:
            tokens.append(self.vocab.special_tokens['[STRUCT_START]'])
            tokens.extend(self._tokenize_sequence(structure, 'ss'))
        
        if consensus:
            tokens.append(self.vocab.special_tokens['[CONS_START]'])
            tokens.extend(self._tokenize_sequence(consensus, 'consensus'))
        
        if go_terms:
            tokens.append(self.vocab.special_tokens['[GO_START]'])
            tokens.extend(self._tokenize_go_terms(go_terms))
        
        # Add SEP token
        tokens.append(self.vocab.special_tokens['[SEP]'])
        
        return tokens
    
    def _tokenize_sequence(self, sequence: str, modality: str) -> List[int]:
        """
        Tokenize a sequence for a specific modality.
        
        Args:
            sequence: Input sequence
            modality: Modality name
            
        Returns:
            List of global token IDs
        """
        tokens = []
        for char in sequence:
            token_id = self.vocab.get_global_token_id(char, modality)
            tokens.append(token_id)
        
        return tokens
    
    def _tokenize_go_terms(self, go_terms: str) -> List[int]:
        """
        Tokenize GO terms.
        
        Args:
            go_terms: Comma-separated GO terms (with or without GO: prefix)
            
        Returns:
            List of global token IDs
        """
        tokens = []
        if go_terms:
            term_list = [term.strip() for term in go_terms.split(',') if term.strip()]
            for term in term_list:
                # Add GO: prefix if not present
                go_term_with_prefix = f"GO:{term}" if not term.startswith('GO:') else term
                token_id = self.vocab.get_global_token_id(go_term_with_prefix, 'go_terms')
                tokens.append(token_id)
        
        return tokens
    
    def decode_multimodal(self, token_ids: List[int]) -> Dict[str, str]:
        """
        Decode global token IDs back to original modalities.
        
        Args:
            token_ids: List of global token IDs
            
        Returns:
            Dictionary with modality keys and decoded strings
        """
        result = {
            'sequence': '',
            'structure': '',
            'consensus': '',
            'go_terms': ''
        }
        
        current_modality = None
        current_tokens = []
        
        for token_id in token_ids:
            token, modality = self.vocab.get_local_token_id(token_id)
            
            # Handle special tokens
            if token in ['[CLS]', '[SEP]']:
                continue
            elif token in ['[SEQ_START]', '[STRUCT_START]', '[CONS_START]', '[GO_START]']:
                # Save previous modality tokens
                if current_modality and current_tokens:
                    result[current_modality] = ''.join(current_tokens)
                
                # Set new modality
                if token == '[SEQ_START]':
                    current_modality = 'sequence'
                elif token == '[STRUCT_START]':
                    current_modality = 'structure'
                elif token == '[CONS_START]':
                    current_modality = 'consensus'
                elif token == '[GO_START]':
                    current_modality = 'go_terms'
                
                current_tokens = []
            else:
                # Add token to current modality
                if current_modality:
                    current_tokens.append(token)
        
        # Save final modality tokens
        if current_modality and current_tokens:
            result[current_modality] = ''.join(current_tokens)
        
        return result


class ModalContextTracker:
    """Track and validate modal context during tokenization and generation."""
    
    def __init__(self, unified_vocab: UnifiedVocabulary):
        """
        Initialize context tracker.
        
        Args:
            unified_vocab: Unified vocabulary instance
        """
        self.vocab = unified_vocab
        self.modality_separators = {
            '[SEQ_START]': 'sequence',
            '[STRUCT_START]': 'structure', 
            '[CONS_START]': 'consensus',
            '[GO_START]': 'go_terms'
        }
    
    def detect_modality_boundaries(self, token_ids: List[int]) -> List[Tuple[int, str]]:
        """
        Find where modality switches occur.
        
        Args:
            token_ids: List of global token IDs
            
        Returns:
            List of (position, modality) tuples
        """
        boundaries = []
        
        for i, token_id in enumerate(token_ids):
            token, modality = self.vocab.get_local_token_id(token_id)
            if token in self.modality_separators:
                boundaries.append((i, self.modality_separators[token]))
        
        return boundaries
    
    def validate_modal_sequence(self, token_ids: List[int]) -> Tuple[bool, List[str]]:
        """
        Validate proper modal separator usage.
        
        Args:
            token_ids: List of global token IDs
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for CLS and SEP tokens
        if token_ids:
            first_token, _ = self.vocab.get_local_token_id(token_ids[0])
            last_token, _ = self.vocab.get_local_token_id(token_ids[-1])
            
            if first_token != '[CLS]':
                errors.append("Sequence must start with [CLS] token")
            
            if last_token != '[SEP]':
                errors.append("Sequence must end with [SEP] token")
        else:
            errors.append("Empty token sequence")
        
        # Check modality order
        boundaries = self.detect_modality_boundaries(token_ids)
        expected_order = ['sequence', 'structure', 'consensus', 'go_terms']
        actual_order = [modality for _, modality in boundaries]
        
        # Validate that at least sequence is present
        if 'sequence' not in actual_order:
            errors.append("Sequence modality is required")
        
        # Check order of modalities
        for i, (pos, modality) in enumerate(boundaries):
            if modality in expected_order:
                expected_pos = expected_order.index(modality)
                if i != expected_pos:
                    errors.append(f"Modality {modality} appears in wrong position")
        
        return len(errors) == 0, errors
    
    def get_current_modality(self, token_ids: List[int], position: int) -> Optional[str]:
        """
        Get the current modality at a given position.
        
        Args:
            token_ids: List of global token IDs
            position: Position in the sequence
            
        Returns:
            Current modality name or None
        """
        boundaries = self.detect_modality_boundaries(token_ids)
        
        # Find the last boundary before the position
        current_modality = None
        for pos, modality in boundaries:
            if pos <= position:
                current_modality = modality
            else:
                break
        
        return current_modality


class VocabularyAnalyzer:
    """Comprehensive vocabulary analyzer for RNA multimodal data."""
    
    def __init__(self, h5_file_path: str):
        """
        Initialize the analyzer with an HDF5 file.
        
        Args:
            h5_file_path: Path to the HDF5 file containing multimodal RNA data
        """
        self.h5_file_path = h5_file_path
        self.modalities = ['sequence', 'ss', 'consensus', 'go_terms']
        self.data = {}
        self.vocabularies = {}
        self.token_stats = {}
        
    def load_data(self) -> None:
        """Load data from HDF5 file into memory."""
        print(f"Loading data from {self.h5_file_path}...")
        with h5py.File(self.h5_file_path, 'r') as f:
            for modality in self.modalities:
                if modality in f:
                    # Convert bytes to strings if necessary
                    data = f[modality][:]
                    if data.dtype.kind == 'S':  # bytes
                        data = [item.decode('utf-8') if isinstance(item, bytes) else item for item in data]
                    self.data[modality] = data
                    print(f"  Loaded {len(data):,} entries for {modality}")
                else:
                    print(f"  Warning: {modality} not found in HDF5 file")
    
    def analyze_modality_tokens(self, modality: str) -> Dict[str, Any]:
        """
        Analyze tokens for a specific modality.
        
        Args:
            modality: Name of the modality to analyze
            
        Returns:
            Dictionary containing token analysis results
        """
        if modality not in self.data:
            return {}
            
        data = self.data[modality]
        
        # Token extraction based on modality type
        if modality == 'go_terms':
            # GO terms are comma-separated
            tokens = []
            for entry in data:
                if entry and entry != '':  # Skip empty entries
                    # Ensure entry is a string
                    if isinstance(entry, bytes):
                        entry = entry.decode('utf-8')
                    tokens.extend([token.strip() for token in entry.split(',') if token.strip()])
        else:
            # For sequence, ss, consensus - treat each character as a token
            tokens = []
            for entry in data:
                if entry and entry != '':  # Skip empty entries
                    # Ensure entry is a string
                    if isinstance(entry, bytes):
                        entry = entry.decode('utf-8')
                    tokens.extend(list(entry))
        
        # Calculate statistics
        token_counter = Counter(tokens)
        unique_tokens = set(tokens)
        
        # Length analysis for sequences
        lengths = []
        if modality in ['sequence', 'ss', 'consensus']:
            lengths = [len(entry) for entry in data if entry and entry != '']
        
        stats = {
            'total_tokens': len(tokens),
            'unique_tokens': len(unique_tokens),
            'token_frequencies': dict(token_counter.most_common()),
            'length_stats': {
                'min': min(lengths) if lengths else None,
                'max': max(lengths) if lengths else None,
                'mean': np.mean(lengths) if lengths else None,
                'median': np.median(lengths) if lengths else None,
                'std': np.std(lengths) if lengths else None
            } if lengths else None,
            'sample_tokens': list(unique_tokens)[:20],  # First 20 unique tokens
            'most_common_tokens': token_counter.most_common(20)
        }
        
        self.vocabularies[modality] = unique_tokens
        self.token_stats[modality] = stats
        
        return stats
    
    def analyze_all_modalities(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze tokens for all modalities.
        
        Returns:
            Dictionary containing analysis results for all modalities
        """
        print("Analyzing tokens across all modalities...")
        results = {}
        
        for modality in self.modalities:
            print(f"  Analyzing {modality}...")
            results[modality] = self.analyze_modality_tokens(modality)
        
        return results
    
    def analyze_cross_modality_overlap(self) -> Dict[str, Any]:
        """
        Analyze token overlap between different modalities.
        
        Returns:
            Dictionary containing cross-modality overlap analysis
        """
        print("Analyzing cross-modality token overlap...")
        
        overlap_matrix = {}
        overlap_details = {}
        
        for i, modality1 in enumerate(self.modalities):
            if modality1 not in self.vocabularies:
                continue
                
            overlap_matrix[modality1] = {}
            overlap_details[modality1] = {}
            
            for j, modality2 in enumerate(self.modalities):
                if modality2 not in self.vocabularies:
                    continue
                    
                if i == j:
                    overlap_matrix[modality1][modality2] = 1.0
                    overlap_details[modality1][modality2] = {
                        'overlap_tokens': list(self.vocabularies[modality1]),
                        'overlap_count': len(self.vocabularies[modality1])
                    }
                else:
                    vocab1 = self.vocabularies[modality1]
                    vocab2 = self.vocabularies[modality2]
                    
                    overlap = vocab1.intersection(vocab2)
                    union = vocab1.union(vocab2)
                    
                    jaccard_similarity = len(overlap) / len(union) if union else 0
                    overlap_matrix[modality1][modality2] = jaccard_similarity
                    
                    overlap_details[modality1][modality2] = {
                        'overlap_tokens': list(overlap),
                        'overlap_count': len(overlap),
                        'jaccard_similarity': jaccard_similarity
                    }
        
        return {
            'overlap_matrix': overlap_matrix,
            'overlap_details': overlap_details
        }
    
    def create_unified_vocabulary(self, include_special_tokens: bool = True) -> Dict[str, Any]:
        """
        Create a unified vocabulary from all modalities.
        
        Args:
            include_special_tokens: Whether to include special tokens like [PAD], [UNK], etc.
            
        Returns:
            Dictionary containing unified vocabulary information
        """
        print("Creating unified vocabulary...")
        
        # Collect all unique tokens
        all_tokens = set()
        modality_token_mapping = {}
        
        for modality in self.modalities:
            if modality in self.vocabularies:
                modality_tokens = self.vocabularies[modality]
                all_tokens.update(modality_tokens)
                modality_token_mapping[modality] = list(modality_tokens)
        
        # Create token to index mapping
        token_to_idx = {}
        idx_to_token = {}
        
        if include_special_tokens:
            special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
            for i, token in enumerate(special_tokens):
                token_to_idx[token] = i
                idx_to_token[i] = token
        
        # Add all other tokens
        start_idx = len(token_to_idx)
        for i, token in enumerate(sorted(all_tokens)):
            token_to_idx[token] = start_idx + i
            idx_to_token[start_idx + i] = token
        
        unified_vocab = {
            'token_to_idx': token_to_idx,
            'idx_to_token': idx_to_token,
            'vocab_size': len(token_to_idx),
            'modality_token_mapping': modality_token_mapping,
            'special_tokens': special_tokens if include_special_tokens else [],
            'modality_sizes': {
                modality: len(tokens) for modality, tokens in modality_token_mapping.items()
            }
        }
        
        return unified_vocab
    
    def build_unified_vocabulary_system(self) -> UnifiedVocabulary:
        """
        Build the complete unified vocabulary system with offset-based mapping.
        
        Returns:
            UnifiedVocabulary instance
        """
        print("Building unified vocabulary system...")
        
        # Create unified vocabulary
        unified_vocab = UnifiedVocabulary()
        
        # Add GO terms from the data
        if 'go_terms' in self.vocabularies:
            go_terms = list(self.vocabularies['go_terms'])
            print(f"Adding {len(go_terms)} GO terms to vocabulary...")
            unified_vocab.add_go_terms(go_terms)
        
        print(f"Unified vocabulary size: {unified_vocab.vocab_size}")
        print("Modality ranges:")
        for modality in unified_vocab.modality_order:
            start, end = unified_vocab.get_modality_range(modality)
            print(f"  {modality}: {start}-{end-1} ({end-start} tokens)")
        
        return unified_vocab
    
    def generate_visualizations(self, output_dir: str = "vocabulary_analysis") -> None:
        """
        Generate visualizations for vocabulary analysis.
        
        Args:
            output_dir: Directory to save visualization files
        """
        print(f"Generating visualizations in {output_dir}...")
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError as exc:
            raise ImportError(
                "--visualize needs matplotlib and seaborn: pip install matplotlib seaborn"
            ) from exc
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Token frequency distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, modality in enumerate(self.modalities):
            if modality in self.token_stats:
                stats = self.token_stats[modality]
                tokens, counts = zip(*stats['most_common_tokens'][:10])
                
                axes[i].bar(range(len(tokens)), counts)
                axes[i].set_title(f'{modality.upper()} - Top 10 Tokens')
                axes[i].set_xlabel('Tokens')
                axes[i].set_ylabel('Frequency')
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add token labels
                axes[i].set_xticks(range(len(tokens)))
                axes[i].set_xticklabels(tokens, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/token_frequencies.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Vocabulary size comparison
        vocab_sizes = []
        modality_names = []
        
        for modality in self.modalities:
            if modality in self.vocabularies:
                vocab_sizes.append(len(self.vocabularies[modality]))
                modality_names.append(modality.upper())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(modality_names, vocab_sizes, color=sns.color_palette("husl", len(modality_names)))
        plt.title('Vocabulary Size by Modality')
        plt.ylabel('Number of Unique Tokens')
        
        # Add value labels on bars
        for bar, size in zip(bars, vocab_sizes):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(vocab_sizes),
                    str(size), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/vocabulary_sizes.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Length distributions for sequence modalities
        sequence_modalities = ['sequence', 'ss', 'consensus']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, modality in enumerate(sequence_modalities):
            if modality in self.token_stats and self.token_stats[modality]['length_stats']:
                # Get actual lengths from data
                lengths = [len(entry) for entry in self.data[modality] if entry and entry != '']
                
                axes[i].hist(lengths, bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{modality.upper()} Length Distribution')
                axes[i].set_xlabel('Length')
                axes[i].set_ylabel('Frequency')
                
                # Add statistics text
                stats = self.token_stats[modality]['length_stats']
                stats_text = f"Mean: {stats['mean']:.1f}\nMedian: {stats['median']:.1f}\nStd: {stats['std']:.1f}"
                axes[i].text(0.7, 0.9, stats_text, transform=axes[i].transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/length_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
    
    def export_vocabulary(self, output_path: str, include_special_tokens: bool = True) -> None:
        """
        Export vocabulary analysis results to JSON file.
        
        Args:
            output_path: Path to save the vocabulary JSON file
            include_special_tokens: Whether to include special tokens in unified vocabulary
        """
        print(f"Exporting vocabulary to {output_path}...")
        
        # Create unified vocabulary
        unified_vocab = self.create_unified_vocabulary(include_special_tokens)
        
        # Prepare export data
        export_data = {
            'metadata': {
                'source_file': self.h5_file_path,
                'modalities': self.modalities,
                'total_entries': len(self.data.get(self.modalities[0], [])) if self.data else 0
            },
            'modality_analyses': self.token_stats,
            'unified_vocabulary': unified_vocab,
            'cross_modality_overlap': self.analyze_cross_modality_overlap()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Vocabulary exported to {output_path}")
        print(f"Unified vocabulary size: {unified_vocab['vocab_size']}")
    
    def print_summary(self) -> None:
        """Print a summary of the vocabulary analysis."""
        print("\n" + "="*60)
        print("VOCABULARY ANALYSIS SUMMARY")
        print("="*60)
        
        for modality in self.modalities:
            if modality in self.token_stats:
                stats = self.token_stats[modality]
                print(f"\n{modality.upper()}:")
                print(f"  Total tokens: {stats['total_tokens']:,}")
                print(f"  Unique tokens: {stats['unique_tokens']:,}")
                print(f"  Sample tokens: {stats['sample_tokens'][:10]}")
                
                if stats['length_stats']:
                    length_stats = stats['length_stats']
                    print(f"  Length stats - Mean: {length_stats['mean']:.1f}, "
                          f"Median: {length_stats['median']:.1f}, "
                          f"Std: {length_stats['std']:.1f}")
        
        # Cross-modality overlap
        overlap_analysis = self.analyze_cross_modality_overlap()
        print(f"\nCROSS-MODALITY OVERLAP:")
        for modality1 in self.modalities:
            if modality1 in overlap_analysis['overlap_matrix']:
                for modality2 in self.modalities:
                    if modality2 in overlap_analysis['overlap_matrix'][modality1]:
                        similarity = overlap_analysis['overlap_matrix'][modality1][modality2]
                        print(f"  {modality1} ↔ {modality2}: {similarity:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze vocabulary for RNA multimodal data')
    parser.add_argument('--input', '-i', required=True, 
                       help='Path to HDF5 file (e.g., bert_ready_data.h5)')
    parser.add_argument('--analyze-all', '-a', action='store_true',
                       help='Perform comprehensive analysis of all modalities')
    parser.add_argument('--export-vocab', '-e', action='store_true',
                       help='Export vocabulary analysis to JSON file')
    parser.add_argument('--build-unified-vocab', '-b', action='store_true',
                       help='Build unified vocabulary system with offset-based mapping')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory for visualizations and exports (default: same directory as HDF5 file)')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--no-special-tokens', action='store_true',
                       help='Exclude special tokens from unified vocabulary')
    
    args = parser.parse_args()
    
    # Set default output directory to the same directory as the HDF5 file
    if args.output_dir is None:
        args.output_dir = str(Path(args.input).parent / "vocabulary_analysis")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = VocabularyAnalyzer(args.input)
    
    # Load data
    analyzer.load_data()
    
    if args.analyze_all:
        # Perform comprehensive analysis
        analyzer.analyze_all_modalities()
        analyzer.print_summary()
        
        if args.visualize:
            analyzer.generate_visualizations(args.output_dir)
        
        if args.export_vocab:
            output_path = f"{args.output_dir}/vocabulary_analysis.json"
            analyzer.export_vocabulary(output_path, not args.no_special_tokens)
    
    if args.build_unified_vocab:
        # First run analysis to get all tokens
        analyzer.analyze_all_modalities()
        
        # Build unified vocabulary system
        unified_vocab = analyzer.build_unified_vocabulary_system()
        
        # Save unified vocabulary
        vocab_path = f"{args.output_dir}/unified_vocabulary.json"
        vocab_data = {
            'vocab_size': unified_vocab.vocab_size,
            'special_tokens': unified_vocab.special_tokens,
            'modality_offsets': unified_vocab.modality_offsets,
            'modality_vocabs': unified_vocab.modality_vocabs,
            'global_to_local': {str(k): v for k, v in unified_vocab.global_to_local.items()},
            'local_to_global': {str(k): v for k, v in unified_vocab.local_to_global.items()}
        }
        
        with open(vocab_path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        print(f"Unified vocabulary saved to {vocab_path}")
        
        # Test tokenization
        print("\nTesting tokenization...")
        tokenizer = ModalTokenizer(unified_vocab)
        context_tracker = ModalContextTracker(unified_vocab)
        
        # Test with sample data
        test_sequence = "AUGCAUGC"
        test_structure = "((....))"
        test_consensus = "AUGCAUGC"
        test_go_terms = "GO:0003735,GO:0005840"
        
        print(f"Testing with:")
        print(f"  Sequence: {test_sequence}")
        print(f"  Structure: {test_structure}")
        print(f"  Consensus: {test_consensus}")
        print(f"  GO Terms: {test_go_terms}")
        
        token_ids = tokenizer.encode_multimodal(
            sequence=test_sequence,
            structure=test_structure,
            consensus=test_consensus,
            go_terms=test_go_terms
        )
        
        print(f"\nEncoded tokens: {token_ids}")
        print(f"Token sequence: {[unified_vocab.get_local_token_id(tid)[0] for tid in token_ids]}")
        print(f"Special token check:")
        print(f"  Token 0: {unified_vocab.get_local_token_id(0)}")
        print(f"  Token 6: {unified_vocab.get_local_token_id(6)}")
        print(f"  Token 1: {unified_vocab.get_local_token_id(1)}")
        if '[DROPPED]' in unified_vocab.special_tokens:
            print(f"[DROPPED] token ID: {unified_vocab.special_tokens['[DROPPED]']}")
        
        decoded = tokenizer.decode_multimodal(token_ids)
        print(f"\nDecoded: {decoded}")
        
        is_valid, errors = context_tracker.validate_modal_sequence(token_ids)
        print(f"\nValid sequence: {is_valid}")
        if errors:
            print(f"Errors: {errors}")
        else:
            print("✅ Tokenization test passed!")
    
    if not args.analyze_all and not args.build_unified_vocab:
        # Quick summary
        analyzer.analyze_all_modalities()
        analyzer.print_summary()


if __name__ == '__main__':
    main() 