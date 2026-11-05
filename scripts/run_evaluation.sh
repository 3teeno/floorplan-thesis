#!/bin/bash
# Re-run the 500-plan evaluation set (output/final_500/images.txt) with the current pipeline, in
# batches of 100 per fresh process; resumes from its own stats CSV. Then compare with:
#   python evaluation/compare.py output/final_500/images.txt output/final_500
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True PYTHONPATH=.
export FP_IMAGE_LIST=output/final_500/images.txt FP_OUT_DIR=output/final_500/dxf
export FP_STATS_CSV=output/final_500/run_stats.csv FP_ROOMS_CSV=output/final_500/run_rooms.csv
while true; do
  MAX_IMAGES_PER_RUN=100 "$PY" floorplan_pipeline.py >> output/final_500/run.log 2>&1
  n_new=$(grep "DONE (this invocation)" output/final_500/run.log | tail -1 | grep -oE '[0-9]+ newly processed' | grep -oE '^[0-9]+')
  [ "$n_new" = "0" ] && break
done
echo "ALL DONE" >> output/final_500/run.log
