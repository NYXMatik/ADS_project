"""PubMed/arXiv Parquet -> benchmark CSV without loading all vectors into RAM.

Source: https://huggingface.co/datasets/cryptolab-playground/pubmed-arxiv-abstract-embedding-gemma-300m
The parquet train/test shards are from refs/convert/parquet; neighbors.parquet is from main.

Requirements: pip install pyarrow huggingface_hub numpy
Use: python convert_parquet.py --dataset pubmed-arxiv --download --estimate
     python convert_parquet.py --dataset pubmed-arxiv --convert
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np

try:
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError as e:
    raise SystemExit("Missing package: " + str(e) + "\nInstall: python -m pip install pyarrow huggingface_hub numpy")

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "pubmed-arxiv": "cryptolab-playground/pubmed-arxiv-abstract-embedding-gemma-300m",
    "bloomberg": "cryptolab-playground/Bloomberg-Financial-News-embedding-gemma-300m",
}


def source_folder(name):
    return ROOT / "source" / name


def target_folder(name):
    return ROOT / "data" / name


def download(name):
    src = source_folder(name)
    src.mkdir(parents=True, exist_ok=True)
    print(f"Downloading train/test Parquet shards of {DATASETS[name]} ...", flush=True)
    snapshot_download(
        repo_id=DATASETS[name], repo_type="dataset", revision="refs/convert/parquet",
        allow_patterns=["default/train/*.parquet", "default/test/*.parquet"],
        local_dir=str(src),
    )
    print("Downloading ground truth neighbors.parquet ...", flush=True)
    hf_hub_download(
        repo_id=DATASETS[name], repo_type="dataset", filename="neighbors.parquet",
        revision="main", local_dir=str(src),
    )
    print("Download complete", flush=True)


def source_files(name):
    src = source_folder(name)
    files = {
        "train": sorted((src / "default" / "train").glob("*.parquet")),
        "test": sorted((src / "default" / "test").glob("*.parquet")),
        "neighbors": [src / "neighbors.parquet"] if (src / "neighbors.parquet").is_file() else [],
    }
    for role, items in files.items():
        if not items:
            raise FileNotFoundError(f"Missing {role} parquet in {src}; run --download first")
    return files


def sha256_file(path):
    hashobj = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b""):
            hashobj.update(data)
    return hashobj.hexdigest()


def all_batches(paths, batch_rows, columns=None):
    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
            yield batch.to_pylist()


def all_rows_count(paths):
    return sum(pq.ParquetFile(p).metadata.num_rows for p in paths)


def check_source_schema(files):
    for role in ("train", "test"):
        for path in files[role]:
            names = pq.ParquetFile(path).schema_arrow.names
            if "id" not in names or "emb" not in names:
                raise ValueError(f"Expected id and emb in {path}: got {names}")
    for path in files["neighbors"]:
        names = pq.ParquetFile(path).schema_arrow.names
        if "id" not in names or "neighbors_id" not in names:
            raise ValueError(f"Expected id and neighbors_id in {path}: got {names}")


def metadata_fields(files):
    """Keep all nonembedding columns of the train/test inputs."""
    per_role = {}
    for role in ("train", "test"):
        schemas = [pq.ParquetFile(p).schema_arrow.names for p in files[role]]
        if any(set(s) != set(schemas[0]) for s in schemas[1:]):
            raise ValueError(f"Mixed schemas within {role}")
        per_role[role] = [n for n in schemas[0] if n not in ("id", "emb")]
    return per_role


class NumberFormatter:
    def __init__(self):
        self.exact_f32 = 0
        self.requires_f64 = 0

    def as_text(self, x):
        f = float(x)
        if not math.isfinite(f):
            raise ValueError("NaN/Infinity embedding found")
        f32 = float(np.float32(f))
        if f32 == f:
            self.exact_f32 += 1
            return format(f, ".9g")  # float32 round-trip, no value change
        self.requires_f64 += 1
        return format(f, ".17g")  # float64 round-trip, no value change


def record_to_csv(row, fmt, extras):
    emb = row["emb"]
    if emb is None or len(emb) != 768:
        raise ValueError(f"Expected 768-dimensional embedding, got {len(emb) if emb is not None else None}")
    idx = row["id"]
    if idx is None:
        raise ValueError("Null ID")
    vector = "[" + ",".join(fmt.as_text(x) for x in emb) + "]"
    def meta_as_text(x):
        if x is None:
            return ""
        if isinstance(x, (dict, list, tuple)):
            return json.dumps(x, ensure_ascii=False, separators=(",", ":"))
        return str(x)
    return [int(idx), vector] + [meta_as_text(row.get(k)) for k in extras]


def estimate(files, sample_rows=256, batch_rows=64):
    extras = metadata_fields(files)
    formatter = NumberFormatter()
    print("CSV train size estimate (from sampled Parquet rows):")
    paths = files["train"]
    samples = 0
    bytes_written = 0
    header = io.StringIO(newline="")
    csv.writer(header, lineterminator="\n").writerow(["id", "vector"] + extras["train"])
    for block in all_batches(paths, batch_rows=batch_rows):
        for row in block:
            s = io.StringIO(newline="")
            csv.writer(s, lineterminator="\n").writerow(record_to_csv(row, formatter, extras["train"]))
            bytes_written += len(s.getvalue().encode("utf-8"))
            samples += 1
            if samples >= sample_rows:
                break
        if samples >= sample_rows:
            break
    if samples == 0:
        raise ValueError("Empty train parquet")
    n = all_rows_count(paths)
    estimate_bytes = round(bytes_written / samples * n) + len(header.getvalue().encode("utf-8"))
    print(f"  {n:,} train rows × {bytes_written/samples:,.0f} bytes/row (sample)")
    print(f"  ~{estimate_bytes/1e9:.3f} GB CSV = {estimate_bytes:,} bytes (ESTIMATE, not final)")
    print(f"  Sample components exactly float32: {formatter.exact_f32:,}")
    print(f"  Sample components needing float64 precision: {formatter.requires_f64:,}")
    print("  Note: source float64 values are NEVER silently rounded to float32 by this exporter.")
    return estimate_bytes


def export(files, output, batch_rows=64):
    extras = metadata_fields(files)
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    start = time.perf_counter()
    # Track only IDs, not vectors. 400k IDs are small vs the 4GB CSV; this verifies cross-references.
    id_sets = {}
    for role in ("train", "test"):
        dest = output / (role + ".csv")
        tmp = output / (role + ".csv.part")
        fmt = NumberFormatter()
        count = 0
        ids = set()
        with tmp.open("w", newline="", encoding="utf-8", buffering=4 * 1024 * 1024) as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["id", "vector"] + extras[role])
            for block in all_batches(files[role], batch_rows):
                for item in block:
                    idx = int(item["id"])
                    if idx in ids:
                        raise ValueError(f"Duplicate {role} ID: {idx}")
                    ids.add(idx)
                    writer.writerow(record_to_csv(item, fmt, extras[role]))
                    count += 1
                if count % 50000 < len(block):
                    print(f"{role}: {count:,} rows", flush=True)
        if count != all_rows_count(files[role]):
            raise ValueError(f"Incomplete {role}: {count}")
        tmp.replace(dest)
        id_sets[role] = ids
        summary[role + "_rows"] = count
        summary[role + "_csv_bytes"] = dest.stat().st_size
        summary[role + "_float32_exact_components"] = fmt.exact_f32
        summary[role + "_float64_only_components"] = fmt.requires_f64
        print(f"Created {dest} ({dest.stat().st_size/1e9:.3f} GB)", flush=True)

    # SQL ground truth joined on ORIGINAL query ID, not on row position.
    dest = output / "ground_truth.csv"
    tmp = output / "ground_truth.csv.part"
    query_ids = set()
    gt_count = 0
    smallest_k = 10**9
    with tmp.open("w", newline="", encoding="utf-8", buffering=4 * 1024 * 1024) as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["query_id", "rank", "neighbor_id", "distance"])
        for block in all_batches(files["neighbors"], batch_rows):
            for item in block:
                qid = int(item["id"])
                if qid not in id_sets["test"]:
                    raise ValueError(f"Ground truth query ID {qid} not in test set")
                if qid in query_ids:
                    raise ValueError(f"Duplicate ground-truth query ID {qid}")
                query_ids.add(qid)
                ns = item["neighbors_id"]
                if ns is None or len(ns) < 100:
                    raise ValueError(f"No top 100 neighbors for query {qid}")
                smallest_k = min(smallest_k, len(ns))
                if len(set(map(int, ns))) != len(ns):
                    raise ValueError(f"Duplicate neighbor within query {qid}")
                for rank, nid in enumerate(ns):
                    nid = int(nid)
                    if nid not in id_sets["train"]:
                        raise ValueError(f"Neighbor ID {nid} not found in train set")
                    writer.writerow([qid, rank, nid, ""])
                gt_count += 1
    if query_ids != id_sets["test"]:
        raise ValueError("Not every test query has a ground-truth row")
    tmp.replace(dest)
    summary["ground_truth_queries"] = gt_count
    summary["ground_truth_min_k"] = smallest_k
    summary["ground_truth_csv_bytes"] = dest.stat().st_size
    print(f"Created {dest} ({dest.stat().st_size/1e9:.3f} GB)", flush=True)
    # Exact bytes of source assets recorded so datasets can be audited across groups.
    summary.update({
        "dataset": DATASETS[args.dataset], "dimension": 768,
        "train_metadata_columns": extras["train"],
        "distance": "cosine", "source_format": "Parquet",
        "float_serialization": "9 significant digits for exact float32 values; 17 for other float64 values",
        "ids": "Original source IDs preserved (may be non-contiguous); ground truth refers to them",
        "source_files": [str(p.relative_to(source_folder(args.dataset))) for group in files.values() for p in group],
        "source_sha256": {str(p.relative_to(source_folder(args.dataset))): sha256_file(p) for group in files.values() for p in group},
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "warning": "pgvector VECTOR(768) converts any genuinely float64 values to float32; validate recall against exact search after loading",
    })
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done: {output / 'manifest.json'}", flush=True)
    if summary["train_float64_only_components"] or summary["test_float64_only_components"]:
        print("WARNING: some values genuinely require float64; PostgreSQL VECTOR will round them to float32.")
    else:
        print("OK: all source embedding components exactly representable as float32.")


def main():
    global args
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--batch-rows", type=int, default=64)
    args = ap.parse_args()
    if not (args.download or args.estimate or args.convert):
        ap.error("Specify at least one of --download, --estimate, --convert")
    if args.batch_rows < 1:
        ap.error("--batch-rows must be positive")
    if args.download:
        download(args.dataset)
    if args.estimate or args.convert:
        files = source_files(args.dataset)
        check_source_schema(files)
        print(f"Train: {all_rows_count(files['train']):,}, test: {all_rows_count(files['test']):,}, ground truth: {all_rows_count(files['neighbors']):,}")
        if args.estimate:
            estimate(files, batch_rows=args.batch_rows)
        if args.convert:
            export(files, target_folder(args.dataset), batch_rows=args.batch_rows)


if __name__ == "__main__":
    main()
