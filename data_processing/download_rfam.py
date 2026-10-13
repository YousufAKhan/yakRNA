"""
download_rfam.py

Fetches the Rfam release that yakRNA Design is trained on and lays it out the
way data_processing/rfam_data_preprocessing.py expects:

    training_data/multi_modality_Rfam/
    |-- rfam2go.txt                      GO term -> Rfam family mapping
    `-- stockholm/
        |-- RF00001/RF00001.sto          one Stockholm alignment per family
        |-- RF00002/RF00002.sto
        `-- ...

There are three ways to get there, in increasing order of effort:

  prebuilt         Download semRfam, the already-tokenized dataset (the exact
                   HDF5 the released model was trained on), from the Hub.
                   Nothing to parse or align: one ~2.2GB download and training
                   can start. This is the only option that reproduces the
                   published training set byte for byte, because Rfam's
                   full-region membership changes between releases.

The other two build the dataset locally from Rfam, and differ by three orders
of magnitude:

  seed  (default)  The curated seed alignments from Rfam.seed.gz.
                   Rfam 15.0: 4,178 families / ~93k sequences. One download of
                   ~5MB, finishes in under a minute, no external tools. Use it
                   to get the pipeline running or to train a small model.

  full             Every full-region sequence, realigned to its family
                   covariance model with Infernal's cmalign. Rfam 15.0:
                   ~4,000 families / ~4M sequences. This is what the released
                   110M checkpoint was trained on. It needs Infernal on PATH,
                   tens of GB of disk and, for the handful of families with
                   hundreds of thousands of members, CPU-days.

Usage:
    # fastest path: download the prebuilt semRfam dataset
    python data_processing/download_rfam.py --stage prebuilt

    # build locally instead: GO mappings + seed alignments
    python data_processing/download_rfam.py --stage seed-all

    # published path: GO mappings + CMs + per-family FASTA + cmalign
    python data_processing/download_rfam.py --stage full-all --jobs 8 --cmalign-cpu 4

    # try the full path on three families first
    python data_processing/download_rfam.py --stage full-all --families RF00001,RF00005,RF00162

    # individual stages (same flags, run in this order)
    python data_processing/download_rfam.py --stage rfam2go
    python data_processing/download_rfam.py --stage cm
    python data_processing/download_rfam.py --stage fasta
    python data_processing/download_rfam.py --stage align

After seed-all/full-all, continue with the tokenization pipeline described in
TRAINING.md (the prebuilt stage needs none of it):
    python data_processing/rfam_data_preprocessing.py --stage all
"""

import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else None

FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/Rfam"
# The prebuilt, already-tokenized dataset: semRfam. Override with --hf-repo.
DEFAULT_HF_REPO = "MasterYster/semRfam"
# (candidate names in the Hub repo, local name under --processed-dir, required).
# The local names are the ones the shipped configs point at and the ones the
# local build produces, so the two routes are interchangeable. The vocabulary
# already ships in this repository, so it is only refreshed if the Hub has it.
PREBUILT_FILES = [
    (('semRfam.h5', 'bert_ready_data.h5'), 'bert_ready_data.h5', True),
    (('family_lookup.json', 'semRfam_family_lookup.json'), 'family_lookup.json', True),
    (('family_splits.json', 'semRfam_family_splits.json'), 'family_splits.json', False),
    (('vocabulary_analysis/unified_vocabulary.json', 'unified_vocabulary.json'),
     'vocabulary_analysis/unified_vocabulary.json', False),
]
USER_AGENT = "yakRNA-design/1.0 (+https://github.com/YousufAKhan/yakRNA)"
# cmalign must emit one line per sequence: the Stockholm parser in
# rfam_data_preprocessing.py reads a single #=GC SS_cons / #=GC RF line per
# family, so interleaved (default) output would silently keep only the last
# block of every sequence.
CMALIGN_OUTFORMAT = "Pfam"


# ---------------------------------------------------------------- downloading

def download(url, dest, retries=3, quiet=False):
    """Stream `url` to `dest` atomically. Returns False if the file was already there."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        if not quiet:
            print(f"  exists, skipping: {dest}")
        return False

    os.makedirs(os.path.dirname(os.path.abspath(dest)) or '.', exist_ok=True)
    tmp = f"{dest}.part"
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response, open(tmp, 'wb') as out:
                shutil.copyfileobj(response, out, length=1 << 20)
            os.replace(tmp, dest)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last_error = exc
            if os.path.exists(tmp):
                os.remove(tmp)
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                raise FileNotFoundError(f"not found on the Rfam FTP: {url}") from exc
            if attempt < retries:
                wait = 2 ** attempt
                print(f"  {type(exc).__name__} on {url} (attempt {attempt}/{retries}), retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def gunzip(src, dest, keep_archive=True):
    """Decompress `src` to `dest`."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  exists, skipping: {dest}")
        return
    print(f"  decompressing {src} -> {dest}")
    tmp = f"{dest}.part"
    with gzip.open(src, 'rb') as fin, open(tmp, 'wb') as fout:
        shutil.copyfileobj(fin, fout, length=1 << 20)
    os.replace(tmp, dest)
    if not keep_archive:
        os.remove(src)


# ------------------------------------------------------------------- stage: GO

def stage_rfam2go(args):
    """rfam2go: the GO terms attached to each Rfam family."""
    print(f"[rfam2go] Rfam {args.release}")
    dest = os.path.join(args.out_dir, 'rfam2go.txt')
    download(f"{FTP_BASE}/{args.release}/rfam2go/rfam2go", dest)
    with open(dest, encoding='utf-8', errors='replace') as handle:
        families = {m.group(1) for m in (re.match(r'Rfam:(RF\d{5})', line) for line in handle) if m}
    print(f"[rfam2go] {dest}: GO terms for {len(families)} families")


# ----------------------------------------------------------------- stage: seed

def split_seed_alignments(seed_path, stockholm_dir, only=None):
    """Split the concatenated Rfam.seed into stockholm/<RF>/<RF>.sto files."""
    os.makedirs(stockholm_dir, exist_ok=True)
    written, skipped, sequences = 0, 0, 0
    block, accession = [], None

    # Rfam.seed carries a few non-UTF8 author names; latin-1 never fails and
    # leaves the alignment columns (pure ASCII) untouched.
    with open(seed_path, encoding='latin-1') as handle:
        for line in handle:
            block.append(line)
            if line.startswith('#=GF AC'):
                accession = line.split()[-1].strip()
            elif line.startswith('//'):
                if accession and (only is None or accession in only):
                    family_dir = os.path.join(stockholm_dir, accession)
                    os.makedirs(family_dir, exist_ok=True)
                    with open(os.path.join(family_dir, f"{accession}.sto"), 'w', encoding='latin-1') as out:
                        out.writelines(block)
                    written += 1
                    sequences += sum(
                        1 for entry in block
                        if entry.strip() and not entry.startswith('#') and not entry.startswith('//')
                    )
                else:
                    skipped += 1
                block, accession = [], None

    return written, skipped, sequences


def stage_seed(args):
    """Seed alignments: one small download, no Infernal needed."""
    print(f"[seed] Rfam {args.release} seed alignments")
    archive = os.path.join(args.out_dir, 'Rfam.seed.gz')
    seed_path = os.path.join(args.out_dir, 'Rfam.seed')
    download(f"{FTP_BASE}/{args.release}/Rfam.seed.gz", archive)
    gunzip(archive, seed_path, keep_archive=args.keep_archives)

    claim_stockholm_dir(args, 'seed')
    only = set(resolve_family_filter(args)) or None
    written, skipped, sequences = split_seed_alignments(seed_path, stockholm_dir(args), only)
    print(f"[seed] wrote {written} family alignments ({sequences} sequences) to {stockholm_dir(args)}")
    if skipped:
        print(f"[seed] skipped {skipped} families not in --families/--families-file")


# ------------------------------------------------------- stages: CM, FASTA, align

def stage_cm(args):
    """Covariance models, indexed so cmfetch can pull one family at a time."""
    print(f"[cm] Rfam {args.release} covariance models")
    archive = os.path.join(args.out_dir, 'Rfam.cm.gz')
    cm_path = os.path.join(args.out_dir, 'Rfam.cm')
    download(f"{FTP_BASE}/{args.release}/Rfam.cm.gz", archive)
    gunzip(archive, cm_path, keep_archive=args.keep_archives)

    require_infernal(['cmfetch'])
    if os.path.exists(cm_path + '.ssi'):
        print(f"  index exists: {cm_path}.ssi")
    else:
        print("  indexing with cmfetch --index (a minute or two)")
        subprocess.run(['cmfetch', '--index', cm_path], check=True)
    print(f"[cm] {cm_path} ready ({len(families_from_cm(cm_path))} families)")


def stage_fasta(args):
    """Per-family full-region sequences (unaligned)."""
    families = resolve_family_filter(args) or families_from_cm(cm_path(args))
    families = apply_max(args, families)
    fasta_dir = os.path.join(args.out_dir, 'fasta_files')
    os.makedirs(fasta_dir, exist_ok=True)
    print(f"[fasta] downloading {len(families)} family FASTA files to {fasta_dir}")

    failures = []

    def fetch(family):
        archive = os.path.join(fasta_dir, f"{family}.fa.gz")
        fasta = os.path.join(fasta_dir, f"{family}.fa")
        if os.path.exists(fasta) and os.path.getsize(fasta) > 0:
            return family, None
        try:
            download(f"{FTP_BASE}/{args.release}/fasta_files/{family}.fa.gz", archive, quiet=True)
            gunzip_quiet(archive, fasta, keep_archive=args.keep_archives)
            return family, None
        except Exception as exc:  # noqa: BLE001 - one bad family must not stop the rest
            return family, str(exc)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(fetch, family) for family in families]
        for future in tqdm(as_completed(futures), total=len(futures), desc="FASTA"):
            family, error = future.result()
            if error:
                failures.append((family, error))

    print(f"[fasta] done: {len(families) - len(failures)} ok, {len(failures)} failed")
    report_failures(args, 'fasta', failures)


def stage_align(args):
    """cmalign each family's sequences against its covariance model."""
    require_infernal(['cmalign', 'cmfetch'])
    fasta_dir = os.path.join(args.out_dir, 'fasta_files')
    families = resolve_family_filter(args) or sorted(
        name[:-3] for name in os.listdir(fasta_dir) if name.endswith('.fa')
    )
    families = apply_max(args, families)
    if not families:
        sys.exit(f"[align] no FASTA files in {fasta_dir} - run --stage fasta first")
    claim_stockholm_dir(args, 'full')

    # Smallest families first, so problems surface in seconds rather than hours.
    families.sort(key=lambda fam: file_size(os.path.join(fasta_dir, f"{fam}.fa")))
    print(f"[align] aligning {len(families)} families "
          f"({args.jobs} in parallel, cmalign --cpu {args.cmalign_cpu})")

    failures = []

    def align(family):
        fasta = os.path.join(fasta_dir, f"{family}.fa")
        out_dir = os.path.join(stockholm_dir(args), family)
        out_sto = os.path.join(out_dir, f"{family}.sto")
        if os.path.exists(out_sto) and os.path.getsize(out_sto) > 0:
            return family, None
        if not os.path.exists(fasta):
            return family, "missing FASTA"
        os.makedirs(out_dir, exist_ok=True)
        tmp_cm = tempfile.NamedTemporaryFile(prefix=f"{family}_", suffix='.cm', delete=False)
        tmp_cm.close()
        tmp_sto = f"{out_sto}.part"
        try:
            with open(tmp_cm.name, 'wb') as handle:
                subprocess.run(['cmfetch', cm_path(args), family],
                               stdout=handle, stderr=subprocess.PIPE, check=True)
            subprocess.run(
                ['cmalign', '--outformat', CMALIGN_OUTFORMAT, '--cpu', str(args.cmalign_cpu),
                 '--mxsize', str(args.mxsize), '-o', tmp_sto, tmp_cm.name, fasta],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True,
            )
            os.replace(tmp_sto, out_sto)
            return family, None
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or b'').decode('utf-8', 'replace').strip().splitlines()
            return family, (detail[-1] if detail else f"exit {exc.returncode}")
        except Exception as exc:  # noqa: BLE001
            return family, str(exc)
        finally:
            for path in (tmp_cm.name, tmp_sto):
                if os.path.exists(path):
                    os.remove(path)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(align, family) for family in families]
        for future in tqdm(as_completed(futures), total=len(futures), desc="cmalign"):
            family, error = future.result()
            if error:
                failures.append((family, error))

    print(f"[align] done: {len(families) - len(failures)} ok, {len(failures)} failed")
    report_failures(args, 'align', failures)


# ------------------------------------------------------------- stage: prebuilt

def stage_prebuilt(args):
    """Download the prebuilt semRfam dataset from the Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import (EntryNotFoundError, GatedRepoError,
                                           RepositoryNotFoundError)
    except ImportError:
        sys.exit("The prebuilt dataset needs huggingface_hub:  pip install huggingface_hub")

    print(f"[prebuilt] semRfam from {args.hf_repo} ({args.hf_repo_type}) -> {args.processed_dir}")
    os.makedirs(args.processed_dir, exist_ok=True)
    downloaded = 0

    for candidates, local_name, required in PREBUILT_FILES:
        destination = os.path.join(args.processed_dir, local_name)
        if os.path.exists(destination) and os.path.getsize(destination) > 0:
            print(f"  exists, skipping: {destination}")
            continue

        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        for candidate in candidates:
            try:
                if candidate != candidates[0]:
                    print(f"  trying alternate name {candidate}")
                else:
                    print(f"  downloading {candidate} -> {destination}")
                cached = hf_hub_download(
                    repo_id=args.hf_repo,
                    repo_type=args.hf_repo_type,
                    filename=candidate,
                )
                # Copy out of the Hub cache under the name the configs expect.
                shutil.copyfile(cached, destination)
                downloaded += 1
                break
            except (RepositoryNotFoundError, GatedRepoError) as exc:
                sys.exit(
                    f"\nCannot read {args.hf_repo_type} repo '{args.hf_repo}': {type(exc).__name__}.\n"
                    "Either it has not been published yet, it is private (run `hf auth login`),\n"
                    "or the name is different - pass --hf-repo <owner>/<name>.\n\n"
                    "To build the dataset from Rfam instead:\n"
                    "  python data_processing/download_rfam.py --stage seed-all   # minutes\n"
                    "  python data_processing/download_rfam.py --stage full-all   # the published set\n"
                )
            except EntryNotFoundError:
                continue
        else:
            if required:
                sys.exit(
                    f"\nNone of {list(candidates)} is in {args.hf_repo}; cannot continue.\n"
                    "Check the repo contents, or build the dataset locally with --stage seed-all."
                )
            print(f"  not in the repo (optional): {local_name}")

    print(f"[prebuilt] {downloaded} file(s) downloaded")
    summarize_processed(args)


def summarize_processed(args):
    """Report whether the processed directory is ready for training."""
    print(f"[summary] processed dataset in {args.processed_dir}:")
    ready = True
    for _candidates, local_name, required in PREBUILT_FILES:
        path = os.path.join(args.processed_dir, local_name)
        if os.path.exists(path):
            print(f"  {local_name}: {file_size(path) / 1e6:.1f} MB")
        else:
            note = 'MISSING' if required else 'absent (created on the first training run)'
            print(f"  {local_name}: {note}")
            ready = ready and not required
    if ready:
        print("[summary] ready - start training with:")
        print("  python rna_gen/rna_gen_training.py --config configs/train_single_gpu.yaml")


# -------------------------------------------------------------------- helpers

def stockholm_dir(args):
    return os.path.join(args.out_dir, 'stockholm')


def marker_path(args):
    return os.path.join(stockholm_dir(args), '.yakrna_dataset')


def read_marker(args):
    """Which dataset ('seed' or 'full') the stockholm/ directory currently holds."""
    try:
        with open(marker_path(args), encoding='utf-8') as handle:
            return handle.read().strip().split()[0]
    except (OSError, IndexError):
        return None


def claim_stockholm_dir(args, kind):
    """Guard against mixing seed and full alignments in one directory.

    Both datasets use stockholm/<RF>/<RF>.sto so that the next pipeline stage
    needs no extra flags, which means a seed run followed by a full run would
    otherwise leave a silent mixture (the full stage skips families that already
    have a .sto).
    """
    os.makedirs(stockholm_dir(args), exist_ok=True)
    existing = read_marker(args)
    if existing and existing != kind:
        if not args.overwrite:
            sys.exit(
                f"{stockholm_dir(args)} already holds the '{existing}' dataset.\n"
                f"Mixing it with '{kind}' alignments would produce an inconsistent training set.\n"
                f"Re-run with --overwrite to replace it, or use --out-dir to build "
                f"the '{kind}' dataset somewhere else."
            )
        print(f"  --overwrite: removing the existing '{existing}' alignments")
        for name in os.listdir(stockholm_dir(args)):
            path = os.path.join(stockholm_dir(args), name)
            if os.path.isdir(path):
                shutil.rmtree(path)
    with open(marker_path(args), 'w', encoding='utf-8') as handle:
        handle.write(f"{kind}\trfam-{args.release}\n")


def cm_path(args):
    path = os.path.join(args.out_dir, 'Rfam.cm')
    if not os.path.exists(path):
        sys.exit(f"{path} not found - run --stage cm first")
    return path


def file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def gunzip_quiet(src, dest, keep_archive=True):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    tmp = f"{dest}.part"
    with gzip.open(src, 'rb') as fin, open(tmp, 'wb') as fout:
        shutil.copyfileobj(fin, fout, length=1 << 20)
    os.replace(tmp, dest)
    if not keep_archive:
        os.remove(src)


def families_from_cm(path):
    """Every family accession in a concatenated Rfam.cm."""
    accessions = []
    with open(path, encoding='latin-1') as handle:
        for line in handle:
            if line.startswith('ACC'):
                parts = line.split()
                if len(parts) >= 2 and re.fullmatch(r'RF\d{5}', parts[1]):
                    accessions.append(parts[1])
    return sorted(set(accessions))


def resolve_family_filter(args):
    """Explicit family list from --families / --families-file, else []."""
    families = []
    if args.families:
        families += [token.strip().upper() for token in args.families.split(',') if token.strip()]
    if args.families_file:
        with open(args.families_file, encoding='utf-8') as handle:
            families += [line.strip().upper() for line in handle if line.strip() and not line.startswith('#')]
    bad = [family for family in families if not re.fullmatch(r'RF\d{5}', family)]
    if bad:
        sys.exit(f"not Rfam accessions: {', '.join(bad)}")
    return sorted(set(families))


def apply_max(args, families):
    if args.max_families and len(families) > args.max_families:
        print(f"  limiting to the first {args.max_families} of {len(families)} families")
        return families[:args.max_families]
    return list(families)


def report_failures(args, stage, failures):
    if not failures:
        return
    path = os.path.join(args.out_dir, f"{stage}_failures.txt")
    with open(path, 'w', encoding='utf-8') as handle:
        for family, error in sorted(failures):
            handle.write(f"{family}\t{error}\n")
    print(f"  {len(failures)} failures written to {path}")
    for family, error in sorted(failures)[:5]:
        print(f"    {family}: {error}")


def require_infernal(tools):
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        sys.exit(
            f"Infernal tool(s) not found on PATH: {', '.join(missing)}\n"
            "Install with:  conda install -c bioconda infernal\n"
            "(or build from http://eddylab.org/infernal/). The 'seed' stage needs no Infernal."
        )


def summarize(args):
    """What is on disk right now."""
    directory = stockholm_dir(args)
    if not os.path.isdir(directory):
        print(f"[summary] {directory} does not exist yet")
        return
    families = [name for name in sorted(os.listdir(directory))
                if re.fullmatch(r'RF\d{5}', name)
                and os.path.isfile(os.path.join(directory, name, f"{name}.sto"))]
    total_bytes = sum(file_size(os.path.join(directory, fam, f"{fam}.sto")) for fam in families)
    kind = read_marker(args) or 'unknown'
    print(f"[summary] {len(families)} family alignments in {directory} "
          f"({total_bytes / 1e9:.2f} GB, dataset: {kind})")
    go_path = os.path.join(args.out_dir, 'rfam2go.txt')
    print(f"[summary] rfam2go.txt: {'present' if os.path.exists(go_path) else 'MISSING'}")
    if families:
        print("[summary] next step:  python data_processing/rfam_data_preprocessing.py --stage all")


def main():
    parser = argparse.ArgumentParser(
        description="Download Rfam and build per-family Stockholm alignments for yakRNA training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages: seed-all (quick) or full-all (the published dataset). See TRAINING.md.",
    )
    parser.add_argument('--stage', required=True,
                        choices=['prebuilt', 'seed-all', 'full-all', 'rfam2go', 'seed', 'cm',
                                 'fasta', 'align', 'summary'],
                        help="'prebuilt' = download the prebuilt semRfam dataset from the Hub; "
                             "'seed-all' = rfam2go + seed; 'full-all' = rfam2go + cm + fasta + align")
    parser.add_argument('--release', default='15.0',
                        help="Rfam release (default: 15.0, the release the released model was trained on)")
    parser.add_argument('--out-dir', default='training_data/multi_modality_Rfam',
                        help='Where raw Rfam downloads and alignments go (default: %(default)s)')
    parser.add_argument('--processed-dir', default='training_data/processed',
                        help='Where the tokenized dataset lives; --stage prebuilt writes here, and it '
                             'is the directory the shipped configs point at (default: %(default)s)')
    parser.add_argument('--hf-repo', default=DEFAULT_HF_REPO,
                        help='Hugging Face repo holding the prebuilt semRfam dataset (default: %(default)s)')
    parser.add_argument('--hf-repo-type', default='dataset', choices=['dataset', 'model'],
                        help='Repo type for --hf-repo (default: %(default)s)')
    parser.add_argument('--families', default=None,
                        help='Comma-separated accessions to restrict to, e.g. RF00001,RF00005')
    parser.add_argument('--families-file', default=None,
                        help='File with one Rfam accession per line')
    parser.add_argument('--max-families', type=int, default=None,
                        help='Only process the first N families (useful for a trial run)')
    parser.add_argument('--jobs', type=int, default=4,
                        help='Families downloaded/aligned in parallel (default: %(default)s)')
    parser.add_argument('--cmalign-cpu', type=int, default=2,
                        help='Threads per cmalign process (default: %(default)s)')
    parser.add_argument('--mxsize', type=int, default=8192,
                        help='cmalign --mxsize in MB; raise if cmalign reports the DP matrix is too big '
                             '(default: %(default)s)')
    parser.add_argument('--keep-archives', action='store_true', default=True,
                        help='Keep the downloaded .gz files (default: on)')
    parser.add_argument('--discard-archives', dest='keep_archives', action='store_false',
                        help='Delete each .gz after decompressing to save disk')
    parser.add_argument('--overwrite', action='store_true',
                        help="Replace alignments already in stockholm/ when switching between "
                             "the 'seed' and 'full' datasets")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.stage == 'prebuilt':
        stage_prebuilt(args)
        return

    if args.stage == 'summary' and not os.path.isdir(stockholm_dir(args)):
        summarize_processed(args)
        return

    if args.stage == 'seed-all':
        stage_rfam2go(args)
        stage_seed(args)
    elif args.stage == 'full-all':
        stage_rfam2go(args)
        stage_cm(args)
        stage_fasta(args)
        stage_align(args)
    elif args.stage == 'rfam2go':
        stage_rfam2go(args)
    elif args.stage == 'seed':
        stage_seed(args)
    elif args.stage == 'cm':
        stage_cm(args)
    elif args.stage == 'fasta':
        stage_fasta(args)
    elif args.stage == 'align':
        stage_align(args)

    summarize(args)


if __name__ == '__main__':
    main()
