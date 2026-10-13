import argparse
import os
import json
import re
from typing import Dict, List, Any
from tqdm import tqdm
import multiprocessing as mp
import h5py
import numpy as np


def ensure_parent_dir(path: str) -> None:
    """Create the directory for `path` if it is not there yet."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def extract_go_mappings(rfam2go_path: str, output_path: str) -> None:
    """
    Extract GO mappings from rfam2go.txt and save as JSON.
    Args:
        rfam2go_path: Path to rfam2go.txt file.
        output_path: Path to save the GO mappings JSON.
    """
    go_map = {}
    # Rfam metadata lines can contain non-UTF-8 author names; latin-1 never
    # fails and the fields this parses are ASCII.
    with open(rfam2go_path, 'r', encoding='latin-1') as f:
        for line in f:
            if line.startswith('!') or not line.strip():
                continue
            # Example: Rfam:RF00001 5S_rRNA > GO:structural constituent of ribosome ; GO:0003735
            match = re.match(r'Rfam:(RF\d{5})\s+[^>]+>.*GO:([0-9]+)', line)
            if match:
                rfam_id, go_id = match.groups()
                go_map.setdefault(rfam_id, set()).add(go_id)
    # Convert sets to sorted lists
    go_map = {k: sorted(list(v)) for k, v in go_map.items()}
    ensure_parent_dir(output_path)
    with open(output_path, 'w') as f:
        json.dump(go_map, f, indent=2)
    print(f"GO mappings saved to {output_path}")


def parse_stockholm_file(sto_path: str) -> Dict[str, Any]:
    """
    Parse a Stockholm file and extract sequence, SS, and consensus.
    Returns a dict with keys: 'sequences', 'ss_cons', 'consensus'.
    """
    sequences = {}
    ss_cons = None
    consensus = None
    # latin-1, not utf-8: Rfam seed alignments carry non-UTF-8 bytes in their
    # #=GF author lines, and a UnicodeDecodeError inside a worker process takes
    # the whole pool down. Alignment rows themselves are ASCII.
    with open(sto_path, 'r', encoding='latin-1') as f:
        for line in f:
            if line.startswith('#=GC SS_cons'):
                ss_cons = line.strip().split()[-1]
            elif line.startswith('#=GC RF'):
                consensus = line.strip().split()[-1]
            elif line.startswith('#') or line.startswith('//') or not line.strip():
                continue
            else:
                parts = line.strip().split()
                if len(parts) == 2:
                    seq_id, seq = parts
                    sequences[seq_id] = seq
    return {'sequences': sequences, 'ss_cons': ss_cons, 'consensus': consensus}


def parse_stockholm_file_with_go(args):
    sto_file, rfam_id, go_terms = args
    try:
        parsed = parse_stockholm_file(sto_file)
    except Exception as exc:  # noqa: BLE001 - keep the pool alive, report the family
        print(f"WARNING: skipping {rfam_id}: {type(exc).__name__}: {exc}")
        return []
    entries = []
    for seq_id, seq in parsed['sequences'].items():
        entry = {
            'rfam_id': rfam_id,
            'sequence_id': seq_id,
            'sequence': seq,
            'ss': parsed['ss_cons'],
            'consensus': parsed['consensus'],
            'go_terms': go_terms
        }
        entries.append(entry)
    return entries


def extract_stockholm_data(stockholm_dir: str, go_mapping_path: str, output_path: str) -> None:
    """
    Extract sequence, SS, consensus, and GO terms from all .sto files using multiprocessing and tqdm progress bar.
    Args:
        stockholm_dir: Directory containing Stockholm .sto files.
        go_mapping_path: Path to GO mappings JSON.
        output_path: Path to save the extracted data JSON.
    """
    with open(go_mapping_path, 'r') as f:
        go_map = json.load(f)
    tasks = []
    for rfam_folder in os.listdir(stockholm_dir):
        folder_path = os.path.join(stockholm_dir, rfam_folder)
        if not os.path.isdir(folder_path):
            continue
        sto_file = os.path.join(folder_path, f"{rfam_folder}.sto")
        if not os.path.isfile(sto_file):
            continue
        rfam_id = rfam_folder
        go_terms = go_map.get(rfam_id, [])
        tasks.append((sto_file, rfam_id, go_terms))
    all_entries = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for entries in tqdm(pool.imap_unordered(parse_stockholm_file_with_go, tasks), total=len(tasks), desc="Processing .sto files"):
            all_entries.extend(entries)
    ensure_parent_dir(output_path)
    with open(output_path, 'w') as f:
        json.dump(all_entries, f, indent=2)
    print(f"Extracted {len(all_entries)} entries to {output_path}")


def clean_entry(entry):
    seq = entry['sequence']
    ss = entry['ss']
    consensus = entry['consensus']
    
    # Only filter sequence data - remove gaps and standardize to A,U,G,C,N
    keep_indices = [i for i, c in enumerate(seq) if c.isalpha() or c.upper() in {'A', 'U', 'G', 'C', 'N'}]
    new_seq = ''.join([
        c.upper() if c.upper() in {'A', 'U', 'G', 'C', 'N'} else 'N' if c.isalpha() else ''
        for i, c in enumerate(seq) if i in keep_indices
    ])
    
    # For SS and consensus: only do position-based alignment, preserve original characters
    new_ss = ''.join([ss[i] for i in keep_indices]) if ss else None
    new_consensus = ''.join([consensus[i] for i in keep_indices]) if consensus else None
    
    cleaned_entry = {
        'rfam_id': entry['rfam_id'],
        'sequence_id': entry['sequence_id'],
        'sequence': new_seq,
        'ss': new_ss,
        'consensus': new_consensus,
        'go_terms': entry['go_terms']
    }
    return cleaned_entry


def clean_and_format_data(input_json_path: str, output_path: str) -> None:
    """
    Clean and format data for BERT-style processing using multiprocessing and tqdm progress bar.
    Args:
        input_json_path: Path to input JSON file.
        output_path: Path to save cleaned/formatted data.
    """
    with open(input_json_path, 'r') as f:
        entries = json.load(f)
    cleaned_entries = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for cleaned in tqdm(pool.imap_unordered(clean_entry, entries, chunksize=1000), total=len(entries), desc="Cleaning entries"):
            cleaned_entries.append(cleaned)
    ensure_parent_dir(output_path)
    with open(output_path, 'w') as f:
        json.dump(cleaned_entries, f, indent=2)
    print(f"Cleaned {len(cleaned_entries)} entries to {output_path}")


def json_to_hdf5(input_json_path: str, output_hdf5_path: str) -> None:
    """
    Convert a JSON file (list of dicts) to an HDF5 file for efficient storage and access.
    Each field (sequence, ss, consensus, go_terms, etc.) is stored as a dataset.
    Args:
        input_json_path: Path to the input JSON file.
        output_hdf5_path: Path to the output HDF5 file.
    """
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    ensure_parent_dir(output_hdf5_path)
    n = len(data)
    # Collect all keys
    keys = list(data[0].keys())
    # Prepare data columns
    columns = {k: [] for k in keys}
    for entry in data:
        for k in keys:
            columns[k].append(entry[k])
    # Write to HDF5
    with h5py.File(output_hdf5_path, 'w') as h5f:
        for k, values in columns.items():
            if isinstance(values[0], list):
                # For go_terms or other lists, store as variable-length sequences
                dt = h5py.special_dtype(vlen=str)
                h5f.create_dataset(k, data=np.array([','.join(map(str, v)) for v in values], dtype=object), dtype=dt)
            else:
                dt = h5py.string_dtype(encoding='utf-8') if isinstance(values[0], str) else None
                h5f.create_dataset(k, data=np.array(values, dtype=object), dtype=dt)
    print(f"Converted {input_json_path} to {output_hdf5_path}")


def main():
    parser = argparse.ArgumentParser(description="Rfam Data Preprocessing Pipeline")
    parser.add_argument('--stage', choices=['go_mapping', 'stockholm', 'clean', 'hdf5', 'all'], required=True,
                        help="Which stage of the pipeline to run.")
    parser.add_argument('--rfam2go_path', default='training_data/multi_modality_Rfam/rfam2go.txt',
                        help="Path to rfam2go.txt file.")
    parser.add_argument('--go_mapping_output', default='training_data/processed/GO_mappings.json',
                        help="Output path for GO mappings JSON.")
    parser.add_argument('--stockholm_dir', default='training_data/multi_modality_Rfam/stockholm',
                        help="Directory containing Stockholm .sto files.")
    parser.add_argument('--go_mapping_path', default='training_data/processed/GO_mappings.json',
                        help="Path to GO mappings JSON.")
    parser.add_argument('--stockholm_output', default='training_data/processed/initial_sequence_extraction.json',
                        help="Output path for extracted sequence data.")
    parser.add_argument('--input_json', default='training_data/processed/initial_sequence_extraction.json',
                        help="Input JSON for cleaning/formatting.")
    parser.add_argument('--clean_output', default='training_data/processed/bert_ready_data.json',
                        help="Output path for cleaned/formatted data.")
    parser.add_argument('--hdf5_output', default='training_data/processed/bert_ready_data.h5',
                        help="Output path for HDF5 file.")
    parser.add_argument('--remove-temp-files', action='store_true',
                        help="If set, remove all intermediate files except the .h5 file after processing.")
    args = parser.parse_args()

    if args.stage == 'go_mapping':
        extract_go_mappings(args.rfam2go_path, args.go_mapping_output)
    elif args.stage == 'stockholm':
        extract_stockholm_data(args.stockholm_dir, args.go_mapping_path, args.stockholm_output)
    elif args.stage == 'clean':
        clean_and_format_data(args.input_json, args.clean_output)
    elif args.stage == 'hdf5':
        json_to_hdf5(args.clean_output, args.hdf5_output)
    elif args.stage == 'all':
        # Step 1: GO mapping
        if os.path.exists(args.go_mapping_output):
            print(f"{args.go_mapping_output} already exists, proceeding to next step")
        else:
            extract_go_mappings(args.rfam2go_path, args.go_mapping_output)
        # Step 2: Stockholm extraction
        if os.path.exists(args.stockholm_output):
            print(f"{args.stockholm_output} already exists, proceeding to next step")
        else:
            extract_stockholm_data(args.stockholm_dir, args.go_mapping_path, args.stockholm_output)
        # Step 3: Cleaning
        if os.path.exists(args.clean_output):
            print(f"{args.clean_output} already exists, proceeding to next step")
        else:
            clean_and_format_data(args.stockholm_output, args.clean_output)
        # Step 4: HDF5 conversion
        if os.path.exists(args.hdf5_output):
            print(f"{args.hdf5_output} already exists, pipeline complete")
        else:
            json_to_hdf5(args.clean_output, args.hdf5_output)
        # Remove temp files if requested
        if args.remove_temp_files:
            to_remove = [args.go_mapping_output, args.stockholm_output, args.clean_output]
            for f in to_remove:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"Removed intermediate file: {f}")

if __name__ == "__main__":
    main() 