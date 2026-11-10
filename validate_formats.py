#!/usr/bin/env python3
"""Validate CSVs produced by convert_formats.py against the source files.

  python validate_formats.py --format bigann --name uform-wiki-1m --train ... --test ... --neighbors ...
  python validate_formats.py --format parquet --name pubmed-arxiv --input source/pubmed-arxiv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import convert_formats as cf

ROOT = Path(__file__).resolve().parent
SAMPLE = 10_000
NQ = 20


def scan_ids_and_sample(path, n_expected, sample_idx, d):
    """Stream the CSV: check ids == 0..n-1 in order, collect sampled vectors."""
    want = set(sample_idx.tolist())
    got = {}
    n = 0
    with open(path, 'r', encoding='utf-8', newline='') as f:
        header = f.readline().rstrip('\n')
        assert header == 'id,vector', header
        for line in f:
            i, rest = line.split(',', 1)
            if int(i) != n:
                raise AssertionError(f'{path}: id {i} at row {n}')
            if n in want:
                v = rest.rstrip('\n').strip('"')[1:-1]
                got[n] = np.array(v.split(','), dtype=np.float32)
                assert got[n].size == d
            n += 1
    assert n == n_expected, (n, n_expected)
    return got


def read_gt(path):
    rows = {}
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        assert next(r) == ['query_id', 'rank', 'neighbor_id', 'distance']
        for q, k, nid, dist in r:
            rows.setdefault(int(q), []).append((int(k), int(nid), dist))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--format', choices=['bigann', 'parquet'], required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--input', type=Path)
    p.add_argument('--train', type=Path)
    p.add_argument('--test', type=Path)
    p.add_argument('--neighbors', type=Path)
    p.add_argument('--train-rows', type=int)
    p.add_argument('--neighbors-with-distances', action='store_true')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'data')
    a = p.parse_args()
    out = a.output_dir / a.name

    # ---- load source
    if a.format == 'bigann':
        tr, te = cf.BigAnn(a.train, '<f4', rows=a.train_rows), cf.BigAnn(a.test, '<f4')
        nb = cf.BigAnnGT(a.neighbors) if a.neighbors_with_distances else cf.BigAnn(a.neighbors, '<i4')
        ntr, nte, d = tr.n, te.n, tr.d
        train = tr.mm
        test = np.array(te.mm[:NQ * 5])
        gt_ids = np.array(nb.mm[:]).astype(np.int64)
    else:
        f = cf.parquet_files(a.input)
        ntr, nte = cf.parquet_rows(f['train']), cf.parquet_rows(f['test'])
        chunk = next(cf.parquet_batches(f['test'], ['emb'], rows=2))
        d = cf.list_to_array(chunk.column('emb'), None).shape[1]
        train = cf.load_f32(f['train'], ntr, d)
        test = cf.load_f32(f['test'], nte, d)
        _, gt_ids, _ = cf.read_neighbors(f['neighbors'][0])
    print(f'source: train {ntr:,} x {d}, test {nte:,}, gt {gt_ids.shape}')

    # ---- 1/2 counts, ids, range
    assert gt_ids.min() >= 0 and gt_ids.max() < ntr
    print('OK ground-truth ids in [0, n_train)')
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(ntr, SAMPLE, replace=False))
    got = scan_ids_and_sample(out / 'train.csv', ntr, idx, d)
    print(f'OK train.csv: {ntr:,} rows, ids 0..n-1 contiguous, dim {d}')
    # ---- 3 round trip
    bad = sum(not np.array_equal(got[i], np.asarray(train[i], dtype=np.float32)) for i in idx.tolist())
    print(f'Round trip train ({SAMPLE:,} random rows, exact float32 equality): {SAMPLE - bad}/{SAMPLE} equal')
    assert bad == 0
    tidx = np.sort(rng.choice(nte, min(SAMPLE, nte), replace=False))
    tgot = scan_ids_and_sample(out / 'test.csv', nte, tidx, d)
    tsrc = np.array(te.mm) if a.format == 'bigann' else test
    tbad = sum(not np.array_equal(tgot[i], tsrc[i]) for i in tidx.tolist())
    print(f'Round trip test ({len(tidx):,} rows): {len(tidx) - tbad}/{len(tidx)} equal; test.csv ids 0..{nte - 1} contiguous')
    assert tbad == 0
    gt = read_gt(out / 'ground_truth.csv')
    assert len(gt) == nte and sorted(gt) == list(range(nte))
    k = gt_ids.shape[1]
    csv_ids = np.array([[x[1] for x in gt[q]] for q in range(nte)])
    assert np.array_equal(csv_ids, gt_ids), 'ground_truth.csv ids differ from source'
    assert all([x[0] for x in gt[q]] == list(range(k)) for q in range(nte))
    print(f'OK ground_truth.csv: {nte:,} queries x k={k}, ids identical to source, ranks 0..k-1')

    # ---- 4 metric check, 20 queries
    qsel = rng.choice(nte, NQ, replace=False)
    ov = {'cosine': [], 'ip': [], 'l2': []}
    tn = None
    for qi in qsel:
        q = np.asarray(test[qi] if a.format == 'parquet' else tsrc[qi], dtype=np.float32)
        if not np.linalg.norm(q):
            print(f'  skip zero-norm query {qi} (cosine undefined)')
            continue
        s_ip = np.empty(ntr, np.float32)
        for lo in range(0, ntr, 100_000):
            s_ip[lo:lo + 100_000] = np.asarray(train[lo:lo + 100_000]) @ q
        if tn is None:
            tn = np.empty(ntr, np.float32)
            for lo in range(0, ntr, 100_000):
                tn[lo:lo + 100_000] = np.einsum('ij,ij->i', np.asarray(train[lo:lo + 100_000]),
                                                np.asarray(train[lo:lo + 100_000]))
        with np.errstate(divide='ignore', invalid='ignore'):
            cs = -s_ip / (np.sqrt(tn) * np.linalg.norm(q))
        cs[~np.isfinite(cs)] = 1.0  # zero-norm train vectors: cosine undefined, rank last
        s = {'ip': -s_ip, 'cosine': cs, 'l2': tn - 2 * s_ip}
        for m, sc in s.items():
            top = np.argpartition(sc, k)[:k]
            ov[m].append(len(set(top.tolist()) & set(gt_ids[qi].tolist())) / k)
    for m, v in ov.items():
        print(f'Metric {m:7s}: mean overlap@{k} with provided ground truth = {np.mean(v):.4f} (min {np.min(v):.2f})')
    # distances column consistency (pubmed)
    dist = [x[2] for x in gt[int(qsel[-1])]]
    print('first selected query distances (rank order):', dist[:5], '... monotone nondecreasing:',
          all(float(x) <= float(y) + 1e-6 for x, y in zip(dist, dist[1:])) if dist[0] else 'n/a (empty)')
    for name in ('train.csv', 'test.csv', 'ground_truth.csv'):
        print(f'{name}: {(out / name).stat().st_size:,} bytes ({(out / name).stat().st_size / 1e9:.3f} GB)')


if __name__ == '__main__':
    main()
