# ADS_project — ADS pgvector Docker Compose starter

Extract **the contents** of this zip into the existing Windows folder
`C:\Users\mateu\Desktop\ann_hdf5_to_csv` (same directory as `convert.py`).
Do not extract into an additional nested folder.

Expected layout:

```text
ann_hdf5_to_csv/
  convert.py
  compose.yaml
  client/Dockerfile
  client/load.py
  data/nytimes-256-angular/train.csv
  data/nytimes-256-angular/manifest.json
  results/  (created during import)
```

## 1. Convert the source HDF5 to CSV first (only if you haven't already)

Copy `.env.example` to `.env` and set a local database password in `.env`.
The `.env` file is ignored by Git; use the same password when connecting with DBeaver.

```powershell
Copy-Item .env.example .env
python convert.py --dataset nytimes-256-angular --download --convert
```

## 2. Start isolated PostgreSQL 18 + pgvector 0.8.7

```powershell
docker compose up -d db
docker compose ps
docker compose exec db psql -U benchmark -d ads_benchmark -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

The last PowerShell command quoting may vary by shell: if needed connect with DBeaver to
`localhost:5433`, database `ads_benchmark`, user `benchmark`, password from `.env`
and run `CREATE EXTENSION IF NOT EXISTS vector; SELECT extversion FROM pg_extension WHERE extname = 'vector';`.

## 3. Build importer

```powershell
docker compose build client
```

## 4. Sample import (recreates the dataset's table)

```powershell
docker compose run --rm client python load.py --dataset nytimes-256-angular --limit 10000 --reset
```

## 5. Monitor Docker memory and CPU in a second terminal

```powershell
docker stats
```

## 6. Full import (WARNING: --reset deletes the sample table)

```powershell
docker compose run --rm client python load.py --dataset nytimes-256-angular --reset
```

This runs PostgreSQL `COPY FROM STDIN` on raw file bytes in 1-MiB chunks. It **does not** load the full CSV in Python memory. A source read-only volume is mounted into the separate client container. The script records ingestion duration, table bytes, row count, ID uniqueness, vector dimensions, and metadata column names in `results/`.

**Important:** test data and ground-truth neighbors remain outside PostgreSQL, as agreed in the shared project plan. This starter does not yet measure recall or index build times. It validates loading first.

## Database connection in DBeaver

- host: `localhost`
- port: `5433`
- database: `ads_benchmark`
- username: `benchmark`
- password: the value of `POSTGRES_PASSWORD` in `.env`

Simple verification:

```sql
SELECT count(*), min(vector_dims(vector)), max(vector_dims(vector)) FROM nytimes_256_angular;
SELECT pg_size_pretty(pg_total_relation_size('nytimes_256_angular'));
```

## Stop without deleting data

```powershell
docker compose down
```

**Never use `docker compose down -v` unless you intentionally want to delete the PostgreSQL volume.**

This is a local experiment, not a production-hardened deployment. For official benchmark runs add a pinned host/VM CPU and RAM limit, measurements of peak container memory, HNSW/IVFFlat index creation, query latency and recall, and non-default credentials. For the Linux container PG18 data volume is attached to `/var/lib/postgresql`.

## Parquet and BigANN-binary datasets (`convert_formats.py`)

Same output schema as `convert.py` (`train.csv`/`test.csv` = `id,vector` with a `[a,b,...]` literal,
`ground_truth.csv` = `query_id,rank,neighbor_id,distance`, floats with 9 significant digits = exact float32
round trip). The source data is only converted, never altered. Source files go in `source/` (git-ignored).

```bash
# BigANN .fbin/.ibin (UForm Wiki). 10,000-row size estimate first, then full conversion
python convert_formats.py --format bigann --name uform-wiki-1m \
  --train source/uform-wiki-1m/base.1M.fbin --test source/uform-wiki-1m/query.public.100K.fbin \
  --neighbors source/uform-wiki-1m/groundtruth.public.100K.ibin --estimate
# same command with --convert instead of --estimate for the full run

# Parquet (pubmed-arxiv); missing ground-truth distances are computed (cosine = 1 - dot)
python convert_formats.py --format parquet --name pubmed-arxiv --input source/pubmed-arxiv --distance cosine --convert

# Wikipedia-Cohere 100K: only the first 307,200,008 bytes of the 107.5 GB wikipedia_base.bin were downloaded
# (curl -r 0-307200007); its header still says 35M vectors, hence --train-rows. GT file = ids + distances.
python convert_formats.py --format bigann --name wikipedia-cohere-100k \
  --train source/wikipedia-cohere-100k/wikipedia_base.first100K.bin --train-rows 100000 \
  --test source/wikipedia-cohere-100k/wikipedia_query.bin \
  --neighbors source/wikipedia-cohere-100k/wikipedia-100K.gt --neighbors-with-distances --distance ip --convert

# Checks against the source: ids, round trip, metric overlap
python validate_formats.py --format parquet --name pubmed-arxiv --input source/pubmed-arxiv
python -m unittest test_convert_formats
```

| Dataset | Source | Format | Shape (train / queries) | Distance | Ground truth | train.csv | test.csv | ground_truth.csv |
|---|---|---|---|---|---|---|---|---|
| uform-wiki-1m | HF `unum-cloud/ann-wiki-1m` | BigANN `.fbin`/`.ibin`, float32 | 1,000,000 x 256 / 100,000 | inner product (vectors unit-norm) | k=10 ids; no distances in source, computed by the converter (cosine, `--compute-distances cosine`) | 3.503 GB | 0.350 GB | 0.016 GB |
| pubmed-arxiv | HF `cryptolab-playground/pubmed-arxiv-abstract-embedding-gemma-300m` | parquet, float64 | 400,335 x 768 / 1,000 | cosine | k=1000 ids; distances computed by the converter (1 - dot) | 4.189 GB | 0.010 GB | 0.026 GB |
| nytimes-256-angular-openvdb | HF `open-vdb/nytimes-256-angular` | parquet (`idx`, `emb` float32) | 290,000 x 256 / 10,000 | cosine | k=1000 ids + source distances (9 queries have null distances, left empty) | 0.998 GB | 0.034 GB | 0.273 GB |
| cohere-small-100k | VectorDBBench `assets.zilliz.com/benchmark/cohere_small_100k/` | parquet, float32 | 100,000 x 768 / 1,000 | cosine (vectors not normalized, norm 12-17) | k=1000 ids; distances computed by the converter (1 - cosine) | 0.963 GB | 0.010 GB | 0.034 GB |
| wikipedia-cohere-100k | big-ann-benchmarks `wikipedia-100K` (Azure `comp21storage.z5.web.core.windows.net/wiki-cohere-35M/`); Cohere wikipedia-22-12 embeddings | BigANN `.bin`, float32 (first 100,000 vectors of the 35M base) | 100,000 x 768 / 5,000 | inner product (not normalized, norm 12-14) | k=100 ids + distances (negative inner product), ids all < 100,000, verified by brute force | 0.962 GB | 0.053 GB | 0.013 GB |

Notes: pubmed-arxiv ids already equal the row position, so no remapping; its float64 values are all exactly
float32-representable, so the cast is lossless. uform-wiki-1m and pubmed-arxiv have (near) unit-norm vectors, so cosine,
inner product and L2 give the same neighbours. nytimes-256-angular-openvdb and cohere-small-100k are not
normalized: their ground truth is cosine (cosine overlap 1.00; inner product 0.88 / 0.41, L2 0.49 / 0.90).
