#!/bin/sh
# Container entrypoint: chown the persistent-volume mount so the non-root
# `ee` user can write uploads + BM25 indexes, then drop privileges and exec
# the CMD. Runs as root on first boot of a Fly volume; on subsequent boots
# chown is a no-op.

set -eu

# /data is the Fly volume mount point (see fly.toml [mounts]).
if [ -d /data ]; then
    mkdir -p /data/uploads /data/indexes /data/.cache/huggingface
    chown -R ee:ee /data
fi

exec gosu ee "$@"
