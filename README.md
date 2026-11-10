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
