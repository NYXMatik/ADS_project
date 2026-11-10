#!/usr/bin/env python3
"""Stream train.csv from the existing HDF5-to-CSV converter into pgvector.

Large/full loads pass the original CSV bytes to PostgreSQL COPY FROM STDIN in
1 MiB blocks (no complete file or row parsing on the client). For --limit,
rows are streamed via Python csv.writer in bounded batches.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import time

import psycopg
from psycopg import sql


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='Subfolder of data/, e.g. nytimes-256-angular')
    p.add_argument('--limit', type=int, default=None, help='Smoke test only, not for final benchmark')
    p.add_argument('--reset', action='store_true', help='DROP and re-create this dataset table')
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        p.error('--limit must be positive')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.dataset):
        p.error('dataset name must contain only letters, numbers, _ and -')
    folder = Path('/data') / a.dataset
    src = folder / 'train.csv'
    man = folder / 'manifest.json'
    if not src.is_file() or not man.is_file():
        p.error(f'Expected {src} and {man}; run convert.py --dataset {a.dataset} --convert first')
    manifest = json.loads(man.read_text(encoding='utf-8'))
    dim = int(manifest['dimension'])
    if not (1 <= dim <= 2000):
        p.error(f'Invalid dimension for pgvector VECTOR: {dim}')
    with src.open(encoding='utf-8', newline='') as f:
        header = next(csv.reader(f))
    if len(header) != len(set(header)) or header[:2] != ['id', 'vector']:
        p.error(f'Unexpected or duplicate CSV header: {header}')
    extra = header[2:]
    table = a.dataset.replace('-', '_').lower()
    cols = [sql.SQL('id BIGINT NOT NULL'), sql.SQL('vector VECTOR({})').format(sql.Literal(dim))]
    cols.extend(sql.SQL('{} TEXT').format(sql.Identifier(c)) for c in extra)
    with psycopg.connect() as con:
        with con.cursor() as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
            if a.reset:
                cur.execute(sql.SQL('DROP TABLE IF EXISTS {}').format(sql.Identifier(table)))
            cur.execute(sql.SQL('CREATE TABLE IF NOT EXISTS {} ({})').format(
                sql.Identifier(table), sql.SQL(', ').join(cols)))
        con.commit()
        start = time.perf_counter()
        with con.cursor() as cur:
            stmt = sql.SQL('COPY {} ({}) FROM STDIN WITH (FORMAT CSV)').format(
                sql.Identifier(table), sql.SQL(', ').join(map(sql.Identifier, header)))
            with cur.copy(stmt) as copy:
                if a.limit is None:
                    with src.open('rb') as f:
                        f.readline()  # discard CSV header, not data
                        while part := f.read(1024 * 1024):
                            copy.write(part)
                else:
                    with src.open('r', newline='', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        next(reader)
                        batch = io.StringIO(newline='')
                        writer = csv.writer(batch, lineterminator='\n')
                        batch_rows = 0
                        for i, row in enumerate(reader):
                            if i >= a.limit:
                                break
                            writer.writerow(row)
                            batch_rows += 1
                            if batch_rows >= 128:
                                copy.write(batch.getvalue())
                                batch = io.StringIO(newline='')
                                writer = csv.writer(batch, lineterminator='\n')
                                batch_rows = 0
                        if batch_rows:
                            copy.write(batch.getvalue())
        con.commit()
        duration = time.perf_counter() - start
        with con.cursor() as cur:
            cur.execute(sql.SQL('SELECT count(*), min(vector_dims(vector)), max(vector_dims(vector)), '
                                'pg_total_relation_size({}) FROM {}').format(
                                    sql.Literal(table), sql.Identifier(table)))
            rows, min_dim, max_dim, table_bytes = cur.fetchone()
            cur.execute(sql.SQL('SELECT count(DISTINCT id) FROM {}').format(sql.Identifier(table)))
            unique_ids, = cur.fetchone()
    result = {
        'dataset': a.dataset, 'table': table, 'rows_after_load': rows,
        'unique_ids': unique_ids, 'vector_dims_min': min_dim,
        'vector_dims_max': max_dim, 'table_bytes': table_bytes,
        'copy_seconds': round(duration, 3),
        'sample_only': a.limit is not None,
        'source_csv_bytes': src.stat().st_size,
        'extra_columns': extra,
    }
    Path('/results').mkdir(exist_ok=True, parents=True)
    suffix = 'sample' if a.limit is not None else 'full'
    path = Path('/results') / f'{table}_{suffix}.json'
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(f'Saved results: {path}')
    if unique_ids != rows or min_dim != dim or max_dim != dim:
        raise SystemExit('VALIDATION FAILED: id uniqueness or vector dimension mismatch')
    if a.limit is None and rows != manifest['train_rows']:
        raise SystemExit('VALIDATION FAILED: loaded row count does not match the source manifest')


if __name__ == '__main__':
    main()
