#!/usr/bin/env bash
# Batch evaluation over every scene of a split, all dynamic levels and both task types.
#
#   SPLIT=test LEVELS="static small all" TASKS="nav eqa" NPROC=4 DATA=./dataset/Dynamic_Strategy bash batch_run.sh
#
# Each (scene, level, task) job writes $OUT/<split>_<scene>_<level>_<task>.json.gz and is skipped
# if that file already exists, so an interrupted run can be resumed.
# Under slurm, submit as an array job (sbatch --array=0-N batch_run.sh): each array task runs one job.
set -uo pipefail
cd "$(dirname "$0")"

export AGENT=${AGENT:-agents.random_agent}
export CFG=${CFG:-./configs/procthor.yaml}
export DATA=${DATA:-./dataset/Dynamic_Strategy}
export SPLIT=${SPLIT:-test}
export START=${START:-0}
export END=${END:-2000}
export OUT=${OUT:-./Result/metrics}
export LOG=${LOG:-./Result/logs}
LEVELS=${LEVELS:-"static small all"}
TASKS=${TASKS:-"nav eqa"}
NPROC=${NPROC:-1}
mkdir -p "$OUT" "$LOG"

list_jobs() {
    for scene in $(ls "$DATA/$SPLIT" | sort -n); do
        for level in $LEVELS; do
            for task in $TASKS; do
                echo "$scene $level $task"
            done
        done
    done
}

run_job() {
    scene=$1 level=$2 task=$3
    name=${SPLIT}_${scene}_${level}_${task}
    [[ -f "$OUT/$name.json.gz" ]] && return 0
    if python runner.py -a "$AGENT" -c "$CFG" -d "$DATA" -p "$SPLIT" -l "$level" -i "$scene" -t "$task" \
            -es "$START" -ee "$END" -o "$OUT/$name.json.gz.tmp" > "$LOG/$name.log" 2>&1; then
        mv "$OUT/$name.json.gz.tmp" "$OUT/$name.json.gz"
    else
        echo "FAILED $name (see $LOG/$name.log)"
    fi
}
export -f run_job

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    run_job $(list_jobs | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
else
    list_jobs | xargs -P "$NPROC" -L 1 bash -c 'run_job "$@"' _
fi
