#!/bin/zsh
# Daily incremental data collection: fetch up to $SCRAPE_LIMIT newly-listed
# for-sale plans in $REGION (default: Pirkanmaa + Uusimaa) with floorplan_scraper.py.
# Resumes from output/listings.jsonl; Oikotie's sitemap refreshes daily.
#
# Schedule with launchd: see scripts/com.floorplan.dailycollect.plist

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"
SCRAPE_LIMIT="${SCRAPE_LIMIT:-120}"
REGION="${REGION:-both}"
LOG="$REPO/output/daily_collect.log"

cd "$REPO"
echo "===== $(date '+%Y-%m-%d %H:%M:%S') start (region=$REGION limit=$SCRAPE_LIMIT) =====" >> "$LOG"

"$PY" floorplan_scraper.py --region "$REGION" --limit "$SCRAPE_LIMIT" --only-with-floorplan >> "$LOG" 2>&1 \
  && echo "scrape ok" >> "$LOG" || echo "scrape FAILED ($?)" >> "$LOG"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') done =====" >> "$LOG"
