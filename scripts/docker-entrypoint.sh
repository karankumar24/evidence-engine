#!/bin/sh
# Container entrypoint: chown the persistent-volume mount so the non-root
# `ee` user can write uploads + BM25 indexes + HF model cache, then drop
# privileges and exec the CMD. Runs as root on first boot of a Fly volume;
# on subsequent boots chown is a no-op.

set -eu

if [ -d /data ]; then
    mkdir -p /data/uploads /data/indexes /data/huggingface

    # Seed the persistent HF cache from the Docker-baked model on first boot.
    # The Docker image pre-downloads ms-marco to /app/.cache/huggingface; copying
    # it to /data/huggingface means the reranker is immediately available without
    # a network download. bge-reranker-v2-m3 is not baked in the image — it
    # downloads to /data/huggingface on first retrieval run, then stays cached.
    if [ -d /app/.cache/huggingface ] && [ -z "$(ls -A /data/huggingface 2>/dev/null)" ]; then
        cp -r /app/.cache/huggingface/. /data/huggingface/ 2>/dev/null || true
    fi

    chown -R ee:ee /data
fi

exec gosu ee "$@"
