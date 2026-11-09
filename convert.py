#!/usr/bin/env python3
"""Streaming HDF5 -> CSV exporter for ANN-Benchmarks and VIBE.

No changes to vectors/ground truth. Default CSV structure is designed for
PostgreSQL COPY: id BIGINT, vector VECTOR(d), and any 1-D train metadata.

Conversion uses 9 significant digits for float32 (round-trip safe), 17 for
float64, and exact decimal text for integral arrays. NaN/Inf are rejected.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.request import Request, urlopen

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
CATALOG = json.loads((ROOT / 'datasets.json').read_text(encoding='utf-8'))
KNOWN = {'train', 'test', 'neighbors', 'distances', 'avg_distances'}


def scalar(x):
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, bytes):
        return x.decode('utf-8', errors='replace')
    if isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError(f'Non-finite float encountered: {x}')
        return x
    if isinstance(x, (int, bool, str)):
        return x
    return str(x)


def numstr(x, dtype):
    if np.issubdtype(dtype, np.floating):
        # 9 significant digits round-trip all binary32 numbers accurately.
        p = 9 if np.dtype(dtype).itemsize <= 4 else 17
        f = float(x)
        if not math.isfinite(f):
            raise ValueError('NaN/Inf in vectors or ground truth')
        return format(f, f'.{p}g')
    if np.issubdtype(dtype, np.integer):
        return str(int(x))
    raise TypeError(f'Unsupported vector dtype: {dtype}')


def vec_text(row, dtype):
    return '[' + ','.join(numstr(x, dtype) for x in row) + ']'


def hdf5_summary(src):
    with h5py.File(src, 'r') as h:
        for key in ('train', 'test', 'neighbors'):
            if key not in h:
                raise ValueError(f'Missing benchmark component: {key}')
        train, test, nbrs = h['train'], h['test'], h['neighbors']
        if train.ndim != 2 or test.ndim != 2 or nbrs.ndim != 2:
            raise ValueError('Expected 2-D train, test and neighbors arrays')
        if test.shape[1] != train.shape[1] or nbrs.shape[0] != test.shape[0]:
            raise ValueError('train/test/neighbors shapes do not match')
        if nbrs.shape[1] < 100:
            raise ValueError('Fewer than 100 ground-truth neighbors, not suitable for recall@100')
        if 'distances' in h and h['distances'].shape != nbrs.shape:
            raise ValueError('Ground-truth distances shape mismatches neighbor IDs')
        meta_keys = sorted(k for k, v in h.items() if k not in KNOWN and isinstance(v, h5py.Dataset)
                           and v.ndim == 1 and len(v) == len(train))
        unsupported_keys = sorted(k for k in h if k not in KNOWN and k not in meta_keys)
        if unsupported_keys:
            raise ValueError('Unknown HDF5 datasets would be lost by this exporter: ' + ', '.join(unsupported_keys))
        attrs = {k: scalar(v) if np.ndim(v) == 0 else repr(v) for k, v in h.attrs.items()}
        result = {
            'input': str(src), 'input_bytes': src.stat().st_size,
            'train_rows': int(train.shape[0]), 'queries': int(test.shape[0]),
            'dimension': int(train.shape[1]), 'neighbors_per_query': int(nbrs.shape[1]),
            'dtype': str(train.dtype), 'attributes': attrs,
            'train_metadata_columns': meta_keys, 'has_distances': 'distances' in h,
            'has_avg_distances': 'avg_distances' in h,
        }
        return result


def get_download(src_url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        print(f'Already downloaded: {target} ({target.stat().st_size / 1e9:.3f} GB)')
        return
    part = target.with_suffix(target.suffix + '.part')
    # Restart an incomplete download rather than risking an invalid Range response.
    if part.exists():
        part.unlink()
    req = Request(src_url, headers={'User-Agent': 'ann-hdf5-csv-converter/1.0'})
    start = time.monotonic()
    received = 0
    with urlopen(req, timeout=120) as response, open(part, 'wb') as out:
        total_header = response.headers.get('Content-Length')
        total = int(total_header) if total_header and total_header.isdigit() else 0
        while True:
            buf = response.read(4 * 1024 * 1024)
            if not buf:
                break
            out.write(buf)
            received += len(buf)
            if received % (128 * 1024 * 1024) < len(buf):
                print(f'Downloaded {received/1e9:.2f} GB'
                      + (f' / {total/1e9:.2f} GB' if total else '')
                      + f' in {time.monotonic()-start:.0f}s', flush=True)
    part.replace(target)
    print(f'Download complete: {target}: {received / 1e9:.3f} GB')


def estimate(src, sample_rows=512):
    with h5py.File(src, 'r') as h:
        a = h['train']
        n = len(a)
        indices = sorted(set(list(range(min(n, sample_rows // 3)))
                             + list(range(max(0, n // 2 - sample_rows // 6), min(n, n // 2 + sample_rows // 6)))
                             + list(range(max(0, n - sample_rows // 3), n))))
        meta_keys = [k for k, v in h.items() if k not in KNOWN and isinstance(v, h5py.Dataset)
                     and v.ndim == 1 and len(v) == n]
        total = 0
        for idx in indices:
            s = io.StringIO(newline='')
            writer = csv.writer(s, lineterminator='\n')
            writer.writerow([idx, vec_text(a[idx], a.dtype)] + [scalar(h[k][idx]) for k in meta_keys])
            total += len(s.getvalue().encode('utf-8'))
        header = io.StringIO(newline='')
        csv.writer(header, lineterminator='\n').writerow(['id','vector'] + meta_keys)
        pred = round((total / len(indices)) * n + len(header.getvalue().encode('utf-8')))
        print(f'Estimated train.csv: {pred / 1e9:.3f} GB ({pred:,} bytes) using {len(indices)} source rows')
        print('This is an estimate, NOT a measured final CSV size.')
        return pred


def export(src, output, chunk_rows=256, limit_rows=None):
    summary = hdf5_summary(src)
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with h5py.File(src, 'r') as h:
        train, test, neighbors = h['train'], h['test'], h['neighbors']
        meta_keys = summary['train_metadata_columns']
        train_count = min(len(train), limit_rows) if limit_rows is not None else len(train)
        if limit_rows is not None:
            print('WARNING: --limit-rows creates a smoke-test sample; not a valid benchmark. '
                  'Ground truth refers to the full train set!')
        # One PostgreSQL-native vector literal per row; CSV quoting is done automatically.
        for key, data, count in [('train',train,train_count), ('test',test,len(test))]:
            path = output / f'{key}.csv'
            with open(path, 'w', encoding='utf-8', newline='', buffering=8*1024*1024) as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(['id', 'vector'] + (meta_keys if key == 'train' else []))
                for lo in range(0, count, chunk_rows):
                    hi = min(count, lo + chunk_rows)
                    block = data[lo:hi]
                    extras = {k: h[k][lo:hi] for k in meta_keys} if key == 'train' else {}
                    for j, row in enumerate(block):
                        idx = lo+j
                        writer.writerow([idx, vec_text(row, data.dtype)]
                                        + ([scalar(extras[k][j]) for k in meta_keys] if key == 'train' else []))
                    if hi % 50000 < chunk_rows:
                        print(f'  {key}: {hi:,}/{count:,}', flush=True)
            print(f'Created {path} ({path.stat().st_size / 1e9:.3f} GB)', flush=True)

        # Neighbor IDs and optional distances, long CSV: query id + rank (0-based).
        ntest, width = neighbors.shape
        with open(output/'ground_truth.csv','w', encoding='utf-8', newline='', buffering=4*1024*1024) as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['query_id','rank','neighbor_id','distance'])
            distds = h.get('distances')
            for lo in range(0, ntest, chunk_rows):
                hi = min(ntest, lo + chunk_rows)
                ids = neighbors[lo:hi]
                dd = distds[lo:hi] if distds is not None else None
                for j, idsrow in enumerate(ids):
                    qid = lo+j
                    for k, nid in enumerate(idsrow):
                        writer.writerow([qid, k, int(nid),
                                         numstr(dd[j,k], distds.dtype) if dd is not None else ''])
        if 'avg_distances' in h:
            avg = h['avg_distances']
            if len(avg) != ntest:
                raise ValueError('avg_distances count != number of test queries')
            with open(output/'avg_distances.csv','w',encoding='utf-8',newline='') as f:
                writer=csv.writer(f,lineterminator='\n')
                writer.writerow(['query_id','avg_distance'])
                for lo in range(0, ntest, chunk_rows):
                    for j,x in enumerate(avg[lo:min(lo+chunk_rows, ntest)]):
                        writer.writerow([lo+j, numstr(x,avg.dtype)])

    measured = {p.name: p.stat().st_size for p in sorted(output.glob('*.csv'))}
    summary.update({
        'conversion': 'float32 => 9 significant digits (lossless float32 round-trip)',
        'output_bytes': measured,
        'train_csv_gb': round(measured['train.csv']/1e9, 6),
        'elapsed_s': round(time.monotonic()-start, 2),
        'warning': 'sample only; ground truth invalid until full train is restored' if limit_rows is not None else None,
    })
    (output/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2), encoding='utf-8')
    print('COMPLETE. Manifest:', output/'manifest.json')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=CATALOG, help='Dataset name from datasets.json')
    p.add_argument('--input',type=Path,help='Use an existing .hdf5 file instead of downloading')
    p.add_argument('--download', action='store_true', help='Download HDF5 first (if missing)')
    p.add_argument('--estimate', action='store_true', help='Estimate train.csv size without full conversion')
    p.add_argument('--convert', action='store_true', help='Export all benchmark data to CSV')
    p.add_argument('--limit-rows', type=int, help='ONLY for smoke test, invalidates ground truth')
    p.add_argument('--source-dir',type=Path,default=ROOT/'source')
    p.add_argument('--output-dir',type=Path,default=ROOT/'output')
    p.add_argument('--chunk-rows',type=int,default=256)
    p.add_argument('--list', action='store_true')
    args=p.parse_args()
    if args.list:
        for name, obj in CATALOG.items():
            print(name, '|', obj['category'], '|', f"{obj['rows']:,} x {obj['dimension']}", '|', obj['distance'])
        return
    if not args.dataset and not args.input:
        p.error('Provide --dataset or --input, or use --list')
    if not(args.download or args.convert or args.estimate):
        p.error('Provide --download, --estimate and/or --convert')
    if args.chunk_rows < 1 or (args.limit_rows is not None and args.limit_rows < 1):
        p.error('chunk-rows/limit-rows must be positive')
    name = args.dataset or args.input.stem
    src=args.input or (args.source_dir/f'{name}.hdf5')
    if args.download:
        if not args.dataset:
            p.error('--download requires --dataset')
        get_download(CATALOG[args.dataset]['url'], src)
    if not src.exists():
        p.error(f'HDF5 file not found: {src}. Download with --download or provide --input')
    desc=hdf5_summary(src)
    print(json.dumps({k:v for k,v in desc.items() if k not in ('input','attributes')}, indent=2))
    if args.dataset:
        expected = CATALOG[args.dataset]
        if desc['dimension'] != expected['dimension'] or desc['train_rows'] != expected['rows']:
            raise ValueError('Actual HDF5 dimensions/rows differ from source catalog!')
    if args.estimate:
        estimate(src)
    if args.convert:
        out=args.output_dir/name
        export(src,out,chunk_rows=args.chunk_rows,limit_rows=args.limit_rows)


if __name__=='__main__':
    main()
