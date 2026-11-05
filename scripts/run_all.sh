#!/bin/bash
# Run the pipeline over every image in output/floorplan_images/, in batches of 150 images per
# fresh process (bounds memory on an 8 GB machine), skipping images over 1 megapixel. Resumes from
# output/run_stats.csv, so it can be stopped and restarted at any time.
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True PYTHONPATH=. MAX_IMAGES_PER_RUN=150 MAX_IMAGE_PX=1000000
while true; do
  "$PY" floorplan_pipeline.py >> output/full_run.log 2>&1
  n_new=$(grep "DONE (this invocation)" output/full_run.log | tail -1 | grep -oE '[0-9]+ newly processed' | grep -oE '^[0-9]+')
  if [ "$n_new" = "0" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') ALL IMAGES DONE" >> output/full_run.log
    break
  fi
done
