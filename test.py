
import csv
import os
import time

import numpy as np
import psycopg
from dotenv import load_dotenv

load_dotenv()

TABLE = "nytimes_256_angular"
TEST_CSV = "data/nytimes-256-angular/test.csv"

N_TRAIN = 10_000
N_QUERIES = 10
K = 10


def parse_vector(value):
    if isinstance(value, str):
        return np.fromstring(
            value.strip().strip("[]"),
            sep=",",
            dtype=np.float32
        )
    return np.asarray(value, dtype=np.float32)


def load_queries(path, limit):
    vectors = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            try:
                if len(row) == 2:
                    vector = parse_vector(row[1])
                elif len(row) == 257:
                    vector = np.asarray(row[1:], dtype=np.float32)
                else:
                    vector = np.asarray(row, dtype=np.float32)

                if vector.size != 256:
                    continue

                vectors.append(vector)

                if len(vectors) == limit:
                    break

            except ValueError:
                continue

    if len(vectors) != limit:
        raise ValueError("Not enough valid test vectors")

    return vectors


def exact_knn_numpy(vectors, ids, query, k):
    # Cosine distance, independently of pgvector
    q = query.astype(np.float64)
    x = vectors.astype(np.float64)

    similarities = (x @ q) / (
        np.linalg.norm(x, axis=1) * np.linalg.norm(q)
    )

    positions = np.argsort(-similarities)[:k]
    return ids[positions].tolist()


def main():
    total_start = time.perf_counter()

    python_time = 0.0
    sql_client_time = 0.0
    pg_time = 0.0

    t = time.perf_counter()
    queries = load_queries(TEST_CSV, N_QUERIES)
    python_time += time.perf_counter() - t

    with psycopg.connect(
        host="127.0.0.1",
        port=5433,
        dbname="ads_benchmark",
        user="benchmark",
        password=os.environ["POSTGRES_PASSWORD"],
        connect_timeout=5
    ) as conn:

        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '30000'")

            print("Creating temporary subset...", flush=True)

            # Temporary table, without vector indexes
            cur.execute(f"""
                CREATE TEMP TABLE sample_vectors AS
                SELECT id, vector
                FROM {TABLE}
                ORDER BY id
                LIMIT {N_TRAIN}
            """)

            cur.execute("""
                SELECT id, vector::text
                FROM sample_vectors
                ORDER BY id
            """)

            rows = cur.fetchall()

            ids = np.array([r[0] for r in rows])
            vectors = np.stack([
                parse_vector(r[1]) for r in rows
            ])

            print(f"Loaded {len(rows)} sample vectors")
            print("Running exact KNN...\n", flush=True)

            query_sql = """
                SELECT id
                FROM sample_vectors
                ORDER BY vector <=> %s::vector
                LIMIT %s
            """

            recalls = []

            for i, query in enumerate(queries, start=1):
                q = "[" + ",".join(map(str, query)) + "]"

                # Independent ground truth (NumPy)
                t = time.perf_counter()
                expected = exact_knn_numpy(vectors, ids, query, K)
                python_time += time.perf_counter() - t

                # Client-side timing
                t = time.perf_counter()

                cur.execute(query_sql, (q, K))
                actual = [r[0] for r in cur.fetchall()]

                client_ms = (time.perf_counter() - t) * 1000
                sql_client_time += client_ms / 1000

                # Database-side timing (separate execution)
                cur.execute(
                    "EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) "
                    + query_sql,
                    (q, K)
                )

                plan = cur.fetchone()[0][0]
                db_ms = plan["Execution Time"]
                pg_time += db_ms / 1000

                # Recall@10
                t = time.perf_counter()
                recall = len(set(actual) & set(expected)) / K
                recalls.append(recall)
                python_time += time.perf_counter() - t

                print(
                    f"Query {i:02d} | "
                    f"Recall@10: {recall:.2f} | "
                    f"Client: {client_ms:.2f} ms | "
                    f"PG: {db_ms:.2f} ms",
                    flush=True
                )

            total_time = time.perf_counter() - total_start

            print("\n========== RESULTS ==========")
            print(f"Train subset: {N_TRAIN}")
            print(f"Queries:      {N_QUERIES}")
            print(f"k:            {K}")
            print(f"Mean Recall:  {np.mean(recalls):.4f}")

            print("\nTIMES")
            print(f"All:          {total_time:.3f} s")
            print(f"Python work:  {python_time:.3f} s")
            print(f"SQL client:   {sql_client_time:.3f} s")
            print(f"PostgreSQL:   {pg_time:.3f} s")
            print("=============================")


if __name__ == "__main__":
    main()
