FROM python:3.11.15-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /project
COPY requirements.txt /tmp/els-requirements.txt
RUN pip install --no-cache-dir -r /tmp/els-requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/els-mpl
CMD ["python", "scripts/run.py", "verify", "--run-dir", "runs/docker_verify"]
