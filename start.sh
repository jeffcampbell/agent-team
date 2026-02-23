#!/bin/bash
# Yamanote orchestrator launcher — restarts automatically on exit (e.g. after ops self-restart)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Yamanote starting..."

while true; do
    [ -f "$SCRIPT_DIR/.env" ] && source "$SCRIPT_DIR/.env"
    python3 orchestrator.py "$@"
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Orchestrator exited (rc=$EXIT_CODE), restarting in 3 seconds..."
    sleep 3
done
