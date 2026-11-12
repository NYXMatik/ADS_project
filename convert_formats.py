#!/usr/bin/env python3
"""Parquet / BigANN-binary -> CSV exporter, same output schema as convert.py.

Output (data/<name>/):
  train.csv         id,vector            vector is a pgvector literal "[a,b,...]"
  test.csv          id,vector
  ground_truth.csv  query_id,rank,neighbor_id,distance   (0-based rank, long format)
  manifest.json

Floats are written with 9 significant digits (exact float32 round trip), like
convert.py. Source data is never modified; only converted (float64 -> float32
for parquet, which is lossless when the source values are float32-representable;
the exporter checks this and reports it).

Formats
  bigann   .fbin / .ibin: little-endian uint32 n, uint32 d, then row-major data.
  parquet  directory with data/train-*.parquet, data/test-*.parquet and
           neighbors.parquet (columns id / emb, id / neighbors_id).

Examples
  python convert_formats.py --format bigann --name uform-wiki-1m \
      --train source/uform-wiki-1m/base.1M.fbin \
      --test source/uform-wiki-1m/query.public.100K.fbin \
      --neighbors source/uform-wiki-1m/groundtruth.public.100K.ibin --estimate
  python convert_formats.py --format parquet --name pubmed-arxiv \
      --input source/pubmed-arxiv --distance cosine --convert
"""
import argparse
import glob
import json
import os
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
CHUNK = 50_000
SAMPLE_ROWS = 10_000
GT_MAX_ERR = [0.0]  # largest |float64 source - float32 written| seen in ground-truth distances


# ---------------------------------------------------------------- CSV writing
def format_vectors(arr):
    """float32 (n, d) -> list of '[a,b,...]' strings, 9 significant digits."""
    if arr.dtype != np.float32:
        raise TypeError('vectors must be float32')
    if not np.isfinite(arr).all():
        raise ValueError('NaN/Inf in vectors')
    fmt = '%.9g'.__mod__
    return ['[' + ','.join(map(fmt, row)) + ']' for row in arr.tolist()]


class CsvWriter:
    """Chunked CSV writer: header written once (unquoted, like csv.writer),
    data through pyarrow.csv.write_csv."""

    def __init__(self, path, header, quote=True):
        self.path = Path(path)
        self.tmp = self.path.with_suffix(self.path.suffix + '.part')
        self.f = open(self.tmp, 'wb', buffering=8 * 1024 * 1024)
        self.f.write((','.join(header) + '\n').encode())
        self.opts = pacsv.WriteOptions(include_header=False, quoting_style='needed' if quote else 'none')
        self.rows = 0

    def write(self, table):
        pacsv.write_csv(table, self.f, self.opts)
        self.rows += table.num_rows

    def close(self):
        self.f.close()
        self.tmp.replace(self.path)
        return self.path.stat().st_size


def vec_table(ids, strings):
    return pa.table({'id': pa.array(ids, pa.int64()), 'vector': pa.array(strings, pa.string())})


def write_vectors(path, chunks):
    """chunks: iterable of float32 (n, d) arrays; ids are the running row position."""
    w = CsvWriter(path, ['id', 'vector'])
    pos = 0
    for block in chunks:
        w.write(vec_table(np.arange(pos, pos + len(block)), format_vectors(block)))
        pos += len(block)
        if pos % 200_000 < len(block):
            print(f'  {Path(path).name}: {pos:,}', flush=True)
    return pos, w.close()


def write_ground_truth(path, ids, dists=None, blocks=None):
    """ids: int array (q, k) (or iterable of such blocks); dists likewise or None."""
    w = CsvWriter(path, ['query_id', 'rank', 'neighbor_id', 'distance'], quote=False)
    q0 = 0
    for ib, db in blocks if blocks is not None else [(ids, dists)]:
        q, k = ib.shape
        if db is None:
            dist_col = pa.array([''] * (q * k), pa.string())
        else:
            # distances are always written as float32 (9 significant digits)
            flat = db.ravel()
            f32 = flat.astype(np.float32)
            err = float(np.nanmax(np.abs(f32.astype(np.float64) - flat))) if flat.size else 0.0
            GT_MAX_ERR[0] = max(GT_MAX_ERR[0], err)
            dist_col = pa.array(['' if x != x else '%.9g' % x for x in f32.tolist()], pa.string())
        w.write(pa.table({
            'query_id': pa.array(np.repeat(np.arange(q0, q0 + q), k), pa.int64()),
            'rank': pa.array(np.tile(np.arange(k), q), pa.int64()),
            'neighbor_id': pa.array(ib.ravel().astype(np.int64), pa.int64()),
            'distance': dist_col,
        }))
        q0 += q
    return w.close()


# ------------------------------------------------------------ BigANN readers
class BigAnn:
    """Read-only view of a BigANN .fbin/.ibin file (uint32 n, uint32 d header)."""

    def __init__(self, path, dtype, rows=None):
        """rows: treat the file as a prefix and use only the first `rows` vectors (the header
        still describes the full file, e.g. a Range-downloaded slice of a larger base file)."""
        self.path = Path(path)
        self.dtype = np.dtype(dtype).newbyteorder('<')
        n, d = np.fromfile(self.path, dtype='<u4', count=2)
        self.n, self.d = int(n), int(d)
        size = self.path.stat().st_size
        if rows is None:
            expected = 8 + self.n * self.d * self.dtype.itemsize
            if size != expected:
                raise ValueError(f'{path}: size {size} != 8 + n*d*itemsize = {expected}')
        else:
            if rows > self.n or size != 8 + rows * self.d * self.dtype.itemsize:
                raise ValueError(f'{path}: size {size} != 8 + {rows}*d*itemsize (header n={self.n})')
            self.n = rows
        self.mm = np.memmap(self.path, dtype=self.dtype, mode='r', offset=8, shape=(self.n, self.d))

    def chunks(self, rows=CHUNK, limit=None):
        n = self.n if limit is None else min(self.n, limit)
        for lo in range(0, n, rows):
            yield np.array(self.mm[lo:min(n, lo + rows)])


class BigAnnGT:
    """big-ann ground-truth file: uint32 nq, uint32 k, ids (nq*k uint32), distances (nq*k float32)."""

    def __init__(self, path):
        self.path = Path(path)
        n, k = np.fromfile(self.path, dtype='<u4', count=2)
        self.n, self.d = int(n), int(k)
        if self.path.stat().st_size != 8 + 2 * self.n * self.d * 4:
            raise ValueError(f'{path}: size != 8 + 2*nq*k*4')
        self.mm = np.memmap(self.path, dtype='<u4', mode='r', offset=8, shape=(self.n, self.d))
        self.dist = np.memmap(self.path, dtype='<f4', mode='r', offset=8 + self.n * self.d * 4,
                              shape=(self.n, self.d))


# ----------------------------------------------------------- parquet readers
def parquet_files(root):
    root = Path(root)
    # layouts: data/{train,test}-*.parquet + neighbors.parquet (pubmed-arxiv)
    #          {train,test,neighbors}/*.parquet                  (open-vdb)
    def find(role, flat):
        return (sorted(glob.glob(str(root / 'data' / f'{role}-*.parquet')))
                or sorted(glob.glob(str(root / role / '*.parquet')))
                or ([str(root / flat)] if (root / flat).is_file() else []))
    files = {'train': find('train', 'train.parquet'), 'test': find('test', 'test.parquet'),
             'neighbors': find('neighbors', 'neighbors.parquet')}
    for k, v in files.items():
        if not v:
            raise FileNotFoundError(f'no {k} parquet under {root}')
    return files


def id_col(path):
    names = pq.ParquetFile(path).schema_arrow.names
    for c in ('id', 'idx'):
        if c in names:
            return c
    raise ValueError(f'no id/idx column in {path}: {names}')


def read_neighbors(path):
    """-> (query_ids, neighbor_ids int64 (q,k), distances float64 (q,k) with NaN for null, or None)"""
    g = pq.read_table(path)
    qids = g[id_col(path)].to_numpy()
    nbr = list_to_array(g['neighbors_id'], np.int64)
    dist = list_to_array(g['neighbors_distance'], np.float64) if 'neighbors_distance' in g.schema.names else None
    return qids, nbr, dist


def parquet_rows(paths):
    return sum(pq.ParquetFile(p).metadata.num_rows for p in paths)


def parquet_batches(paths, columns, rows=CHUNK, limit=None):
    seen = 0
    for p in paths:
        for b in pq.ParquetFile(p).iter_batches(batch_size=rows, columns=columns):
            if limit is not None and seen >= limit:
                return
            if limit is not None and seen + len(b) > limit:
                b = b.slice(0, limit - seen)
            seen += len(b)
            yield b


def list_to_array(col, dtype, width=None):
    """Arrow list<number> column -> (n, d) numpy array; all lists must have equal length."""
    lens = pc.list_value_length(col)
    if lens.null_count or pc.min(lens).as_py() != pc.max(lens).as_py():
        raise ValueError('null or ragged list column')
    d = pc.min(lens).as_py()
    if width is not None and d != width:
        raise ValueError(f'expected width {width}, got {d}')
    flat = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    return flat.flatten().to_numpy(zero_copy_only=False).astype(dtype, copy=False).reshape(-1, d)


class F32Check:
    """Counts float64 source values that are not exactly representable as float32."""
    lossy = 0
    total = 0


def parquet_vector_chunks(paths, check, limit=None, expect_ids=True):
    pos = 0
    ic = id_col(paths[0])
    for b in parquet_batches(paths, [ic, 'emb'], limit=limit):
        ids = b.column(ic).to_numpy()
        if expect_ids and not (ids == np.arange(pos, pos + len(ids))).all():
            raise ValueError(f'id column is not the 0-based row position at row {pos}; remapping required')
        raw = list_to_array(b.column('emb'), None)
        a32 = raw.astype(np.float32)
        if raw.dtype != np.float32:
            check.lossy += int((a32.astype(raw.dtype) != raw).sum())
        check.total += raw.size
        pos += len(ids)
        yield a32


# ------------------------------------------------------------------ helpers
def distances_for(train, queries, ids, metric):
    """Exact distance of each provided neighbour (float64 math)."""
    out = np.empty(ids.shape, np.float64)
    for i in range(len(queries)):
        v = train[ids[i]].astype(np.float64)
        q = queries[i].astype(np.float64)
        if metric == 'cosine':
            out[i] = 1.0 - (v @ q) / (np.linalg.norm(v, axis=1) * np.linalg.norm(q))
        elif metric == 'ip':
            out[i] = -(v @ q)
        elif metric == 'l2':
            out[i] = np.linalg.norm(v - q, axis=1)
        else:
            raise ValueError(metric)
    return out


def load_f32(paths, rows, d):
    a = np.empty((rows, d), np.float32)
    pos = 0
    chk = F32Check()
    for blk in parquet_vector_chunks(paths, chk):
        a[pos:pos + len(blk)] = blk
        pos += len(blk)
    return a


def check_space(est_bytes, out):
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free
    print(f'Free disk: {free / 1e9:.1f} GB; estimated outputs: {est_bytes / 1e9:.2f} GB')
    if free < est_bytes * 1.2:
        raise SystemExit('Not enough free disk space')


# ------------------------------------------------------------------- drivers
def run_bigann(a, out, limit_rows=None, train_only=False):
    tr = BigAnn(a.train, '<f4', rows=a.train_rows)
    te = BigAnn(a.test, '<f4')
    if a.neighbors_with_distances:
        nb = BigAnnGT(a.neighbors)
        dist = type('D', (), {'n': nb.n, 'd': nb.d, 'mm': nb.dist})()
    else:
        nb = BigAnn(a.neighbors, '<i4')
        dist = BigAnn(a.distances, '<f4') if a.distances else None
    if te.d != tr.d or nb.n != te.n or (dist and (dist.n, dist.d) != (nb.n, nb.d)):
        raise ValueError('train/test/neighbors shapes do not match')
    info = {'format': 'bigann', 'train_rows': tr.n, 'queries': te.n, 'dimension': tr.d,
            'neighbors_per_query': nb.d, 'has_distances': dist is not None}
    if int(nb.mm.max()) >= tr.n or int(nb.mm.min()) < 0:
        raise ValueError('ground-truth ids out of range for train')
    print(info, flush=True)
    t0 = time.monotonic()
    sizes = {}
    n, sizes['train.csv'] = write_vectors(out / 'train.csv', tr.chunks(limit=limit_rows))
    if not train_only:
        _, sizes['test.csv'] = write_vectors(out / 'test.csv', te.chunks())
        def gt_block(lo):
            ids = np.array(nb.mm[lo:lo + CHUNK]).astype(np.int64)
            if dist:
                return ids, np.array(dist.mm[lo:lo + CHUNK])
            if a.compute_distances:
                d = distances_for(tr.mm, np.asarray(te.mm[lo:lo + CHUNK]), ids, a.compute_distances)
                return ids, d.astype(np.float32).astype(np.float64)
            return ids, None
        blocks = (gt_block(lo) for lo in range(0, nb.n, CHUNK))
        sizes['ground_truth.csv'] = write_ground_truth(out / 'ground_truth.csv', None, blocks=blocks)
    info.update(distance='from source' if dist else
                (f'computed by converter: {a.compute_distances}, float64 math on float32 vectors, cast to float32'
                 if a.compute_distances else 'none in source: distance column empty'),
                metric_hint=a.distance, elapsed_s=round(time.monotonic() - t0, 1))
    return info, sizes, n


def run_parquet(a, out, limit_rows=None, train_only=False):
    files = parquet_files(a.input)
    ntr, nte = parquet_rows(files['train']), parquet_rows(files['test'])
    d = list_to_array(next(parquet_batches(files['test'], ['emb'], rows=2)).column('emb'), None).shape[1]
    info = {'format': 'parquet', 'train_rows': ntr, 'queries': nte, 'dimension': d}
    print(info, flush=True)
    t0 = time.monotonic()
    sizes = {}
    chk = F32Check()
    n, sizes['train.csv'] = write_vectors(out / 'train.csv', parquet_vector_chunks(files['train'], chk, limit=limit_rows))
    if not train_only:
        _, sizes['test.csv'] = write_vectors(out / 'test.csv', parquet_vector_chunks(files['test'], chk))
        # ground truth: ids refer to train.id, which equals the row position (checked above)
        qids, nbr, src_dist = read_neighbors(files['neighbors'][0])
        if not (qids == np.arange(len(qids))).all():
            raise ValueError('query ids in neighbors file are not 0..n-1 row positions')
        if nbr.min() < 0 or nbr.max() >= ntr:
            raise ValueError('ground-truth ids out of range')
        if len(nbr) != nte:
            raise ValueError('ground-truth query count != test rows')
        if src_dist is not None:
            dists = src_dist
            info['distance_source'] = 'neighbors_distance from source (null -> empty cell)'
            info['queries_with_null_distances'] = int(np.isnan(dists).all(axis=1).sum())
        else:
            print('Computing exact distances for provided neighbours ...', flush=True)
            train = load_f32(files['train'], ntr, d)
            test = load_f32(files['test'], nte, d)
            dists = distances_for(train, test, nbr, a.distance)
            dists = dists.astype(np.float32).astype(np.float64)
            del train
            info['distance_computed'] = f'{a.distance}, float64 math on float32 vectors, cast to float32'
        sizes['ground_truth.csv'] = write_ground_truth(out / 'ground_truth.csv', nbr, dists)
    info.update(float64_values_not_exact_in_float32=chk.lossy, values_checked=chk.total,
                id_policy='train/test id == 0-based row position (verified); no remapping',
                elapsed_s=round(time.monotonic() - t0, 1))
    return info, sizes, n


def run_hdf5(a, out, limit_rows=None, train_only=False):
    """ANN-Benchmarks / VIBE HDF5: train, test, neighbors, optional distances / avg_distances."""
    t0 = time.monotonic()
    sizes = {}
    with h5py.File(a.input, 'r') as h:
        known = {'train', 'test', 'neighbors', 'distances', 'avg_distances'}
        extra = sorted(k for k in h if k not in known)
        if extra:
            raise ValueError('Unknown HDF5 datasets would be lost: ' + ', '.join(extra))
        train, test, nbr = h['train'], h['test'], h['neighbors']
        dist = h['distances'] if 'distances' in h else None
        if train.dtype != np.float32 or test.dtype != np.float32:
            raise ValueError('only float32 vectors are supported')
        if test.shape[1] != train.shape[1] or nbr.shape[0] != test.shape[0] or (dist is not None and dist.shape != nbr.shape):
            raise ValueError('train/test/neighbors shapes do not match')
        info = {'format': 'hdf5', 'train_rows': int(train.shape[0]), 'queries': int(test.shape[0]),
                'dimension': int(train.shape[1]), 'neighbors_per_query': int(nbr.shape[1]),
                'has_distances': dist is not None, 'has_avg_distances': 'avg_distances' in h}
        print(info, flush=True)

        def chunks(ds, limit=None):
            n = len(ds) if limit is None else min(len(ds), limit)
            for lo in range(0, n, CHUNK):
                yield np.array(ds[lo:min(n, lo + CHUNK)])
        n = int(train.shape[0])
        if not a.ground_truth_only:
            n, sizes['train.csv'] = write_vectors(out / 'train.csv', chunks(train, limit_rows))
        if not train_only:
            if not a.ground_truth_only:
                _, sizes['test.csv'] = write_vectors(out / 'test.csv', chunks(test))
            if int(nbr[:].max()) >= train.shape[0]:
                raise ValueError('ground-truth ids out of range')
            blocks = ((np.array(nbr[lo:lo + CHUNK]).astype(np.int64),
                       np.array(dist[lo:lo + CHUNK]).astype(np.float64) if dist is not None else None)
                      for lo in range(0, nbr.shape[0], CHUNK))
            sizes['ground_truth.csv'] = write_ground_truth(out / 'ground_truth.csv', None, blocks=blocks)
            info['ground_truth_max_float32_rounding_error'] = GT_MAX_ERR[0]
            if 'avg_distances' in h:
                avg = h['avg_distances']
                if len(avg) != test.shape[0]:
                    raise ValueError('avg_distances count != number of test queries')
                w = CsvWriter(out / 'avg_distances.csv', ['query_id', 'avg_distance'], quote=False)
                f32 = avg.dtype.itemsize <= 4
                vals = np.array(avg[:])
                w.write(pa.table({'query_id': pa.array(np.arange(len(vals)), pa.int64()),
                                  'avg_distance': pa.array([('%.9g' if f32 else '%.17g') % x for x in vals.tolist()])}))
                sizes['avg_distances.csv'] = w.close()
    info.update(distance='from source' if dist is not None else 'none in source',
                elapsed_s=round(time.monotonic() - t0, 1))
    return info, sizes, n


def estimate(a, runner):
    out = a.output_dir / a.name / '_sample'
    out.mkdir(parents=True, exist_ok=True)
    info, sizes, n = runner(a, out, limit_rows=SAMPLE_ROWS, train_only=True)
    per_row = (sizes['train.csv'] - len('id,vector\n')) / n
    full = round(per_row * info['train_rows'] + len('id,vector\n'))
    shutil.rmtree(out)
    print(f'Measured {per_row:,.1f} bytes/row on first {n:,} rows -> '
          f'train.csv ~ {full / 1e9:.3f} GB ({full:,} bytes). ESTIMATE, not final.')
    return full, info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--format', choices=['bigann', 'parquet', 'hdf5'], required=True)
    p.add_argument('--name', required=True, help='output goes to <output-dir>/<name>/')
    p.add_argument('--input', type=Path, help='parquet: directory with data/*.parquet and neighbors.parquet; hdf5: the .hdf5 file')
    p.add_argument('--train', type=Path, help='bigann: base .fbin')
    p.add_argument('--test', type=Path, help='bigann: query .fbin')
    p.add_argument('--neighbors', type=Path, help='bigann: ground-truth .ibin')
    p.add_argument('--train-rows', type=int,
                   help='bigann: use only the first N vectors; the file may be a prefix of a larger file')
    p.add_argument('--neighbors-with-distances', action='store_true',
                   help='bigann: ground-truth file holds ids then float32 distances (big-ann-benchmarks GT format)')
    p.add_argument('--compute-distances', choices=['cosine', 'ip', 'l2'],
                   help='bigann: compute missing ground-truth distances for the provided neighbours')
    p.add_argument('--distances', type=Path, help='bigann: optional ground-truth distances .fbin')
    p.add_argument('--distance', choices=['cosine', 'ip', 'l2'], default='cosine',
                   help='metric used to compute missing ground-truth distances (parquet)')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'data')
    p.add_argument('--ground-truth-only', action='store_true',
                   help='hdf5: rewrite only ground_truth.csv (and avg_distances.csv); keeps train/test/manifest')
    p.add_argument('--estimate', action='store_true', help='convert 10,000 rows, extrapolate train.csv size')
    p.add_argument('--convert', action='store_true')
    a = p.parse_args()
    if not (a.estimate or a.convert):
        p.error('Provide --estimate and/or --convert')
    if a.format in ('parquet', 'hdf5') and not a.input:
        p.error(f'--format {a.format} needs --input')
    if a.format == 'bigann' and not (a.train and a.test and a.neighbors):
        p.error('--format bigann needs --train, --test and --neighbors')
    runner = {'bigann': run_bigann, 'parquet': run_parquet, 'hdf5': run_hdf5}[a.format]
    out = a.output_dir / a.name
    full = None
    if a.estimate or a.convert:
        full = estimate(a, runner)[0] if not a.ground_truth_only else 0
    if a.convert:
        check_space(int(full * 1.4), out)
        info, sizes, n = runner(a, out)
        if a.ground_truth_only:
            print('Rewrote', sizes, 'max float32 rounding error', info.get('ground_truth_max_float32_rounding_error'))
            return
        info.update(output_bytes=sizes, train_csv_gb=round(sizes['train.csv'] / 1e9, 6),
                    float_format='%.9g (float32 round trip)')
        (out / 'manifest.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        print('COMPLETE', json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
