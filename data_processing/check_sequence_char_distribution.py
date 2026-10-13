"""
check_sequence_char_distribution.py

This script compares the character distributions in the 'sequence' fields of two JSON files:
    1. initial_sequence_extraction.json (raw extracted data)
    2. bert_ready_data.json (cleaned, BERT-ready data)

It uses multiprocessing and a progress bar (tqdm) to efficiently count the occurrences of each character in all sequences in both files.

The results are written to training_data/processed/check_analysis.txt, showing the counts of each unique character in both files.

Usage:
    micromamba run -n multimodal_rna python data_processing/check_sequence_char_distribution.py

Purpose:
    - To verify that gap characters (e.g., '.', '-') and non-AUGCN IUPAC codes were removed or converted to 'N' during cleaning.
    - To explain the reduction in file size after cleaning (due to gap removal).
    - To confirm that the cleaned file contains only the expected tokens: 'A', 'U', 'G', 'C', 'N'.

Interpretation:
    - The initial file should have many gap and ambiguous characters; the cleaned file should only have 'A', 'U', 'G', 'C', 'N'.
    - A large reduction in gap characters explains a large reduction in file size.
    - The script is useful for validating the integrity of the preprocessing pipeline.
"""

import json
import multiprocessing as mp
from collections import Counter
from tqdm import tqdm
import os

def count_entry_chars(entry):
    from collections import Counter
    return Counter(entry['sequence'])

def count_chars_in_sequences(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(count_entry_chars, data, chunksize=1000), total=len(data), desc=f"Counting chars in {os.path.basename(json_path)}"))
    total_counter = Counter()
    for c in results:
        total_counter.update(c)
    return total_counter

def main():
    initial_path = 'training_data/processed/initial_sequence_extraction.json'
    bert_ready_path = 'training_data/processed/bert_ready_data.json'
    initial_counts = count_chars_in_sequences(initial_path)
    bert_ready_counts = count_chars_in_sequences(bert_ready_path)
    with open('training_data/processed/check_analysis.txt', 'w') as f:
        f.write('Character counts in initial_sequence_extraction.json:\n')
        for char, count in initial_counts.most_common():
            f.write(f'{repr(char)}: {count}\n')
        f.write('\nCharacter counts in bert_ready_data.json:\n')
        for char, count in bert_ready_counts.most_common():
            f.write(f'{repr(char)}: {count}\n')
    print('Analysis written to training_data/processed/check_analysis.txt')

if __name__ == '__main__':
    main() 