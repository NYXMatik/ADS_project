FROM python:3.12-slim
WORKDIR /app
RUN python -m pip install --no-cache-dir 'psycopg[binary]==3.2.10'
COPY load.py /app/load.py
