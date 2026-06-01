#!/usr/bin/env bash
# Sequential GPU backlog queue — dry-run validation or nohup live run.
# Usage:
#   ./scripts/run_gpu_queue.sh dry-run
#   ./scripts/run_gpu_queue.sh run
#   ./scripts/run_gpu_queue.sh resume
#   ./scripts/run_gpu_queue.sh pipeline-dry-run   # aug_confirm → gpu_queue_full (print plan)
#   ./scripts/run_gpu_queue.sh pipeline-run     # nohup: confirm then full (auto-waits if confirm in flight)
#   ./scripts/run_gpu_queue.sh pipeline-resume  # nohup: resume confirm if needed, then full
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export DATASET_ROOT="${DATASET_ROOT:-$ROOT/data/raw/extracted/dataset}"
export HARCHOC_EXPORT_DEVICE="${HARCHOC_EXPORT_DEVICE:-cpu}"
export HARCHOC_MAX_EPOCHS="${HARCHOC_MAX_EPOCHS:-120}"
export HARCHOC_MAX_BATCH="${HARCHOC_MAX_BATCH:-1}"
export HARCHOC_MAX_IMGSZ="${HARCHOC_MAX_IMGSZ:-2048}"

MANIFEST_CONFIRM="${GPU_QUEUE_MANIFEST_CONFIRM:-configs/experiments/gpu_queue_aug_confirm.json}"
MANIFEST_FULL="${GPU_QUEUE_MANIFEST_FULL:-configs/experiments/gpu_queue_full.json}"
MANIFEST_CLOSE_PHASE_A="${GPU_QUEUE_MANIFEST_CLOSE_PHASE_A:-configs/experiments/gpu_queue_aug_close_phase_a.json}"
MANIFEST_CLOSE_100EP="${GPU_QUEUE_MANIFEST_CLOSE_100EP:-configs/experiments/gpu_queue_aug_close_100ep.json}"
MANIFEST="${GPU_QUEUE_MANIFEST:-configs/experiments/gpu_queue_aug_pending.json}"
MODE="${1:-dry-run}"
MAMBA_ENV="${HARCHOC_MAMBA_ENV:-harchoc}"
NOHUP_LOG="${GPU_QUEUE_NOHUP_LOG:-reports/gpu_queue/nohup.log}"
NOHUP_PID="${GPU_QUEUE_NOHUP_PID:-reports/gpu_queue/nohup.pid}"
mkdir -p reports/gpu_queue/logs

_resolve_mamba_bin() {
  if [[ -n "${MAMBA_BIN:-}" && -x "${MAMBA_BIN}" ]]; then
    echo "${MAMBA_BIN}"
    return 0
  fi
  local candidates=(
    /root/miniforge3/bin/mamba
    /opt/conda/bin/mamba
    "${HOME}/miniforge3/bin/mamba"
    "${HOME}/mambaforge/bin/mamba"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -x "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  if command -v mamba >/dev/null 2>&1; then
    command -v mamba
    return 0
  fi
  echo "mamba not found; set MAMBA_BIN=/path/to/mamba" >&2
  return 1
}

MAMBA="$(_resolve_mamba_bin)"

_queue_py() {
  "$MAMBA" run -n "$MAMBA_ENV" python scripts/run_gpu_queue.py "$@"
}

_run_manifest() {
  _queue_py --manifest "$MANIFEST" "$@"
}

_confirm_job_done() {
  "$MAMBA" run -n "$MAMBA_ENV" python -c "from pathlib import Path; from harchoc.gpu_queue import load_gpu_queue_manifest, should_skip_job; repo=Path('.').resolve(); m=load_gpu_queue_manifest('${MANIFEST_CONFIRM}', repo_root=repo); job=next(j for j in m['jobs'] if j.get('id')=='aug_confirm_winner_100ep'); skip, reason=should_skip_job(job, repo_root=repo); raise SystemExit(0 if skip and 'summary complete' in reason else 1)"
}

_confirm_in_flight() {
  pgrep -f 'run_gpu_queue\.py.*gpu_queue_aug_confirm\.json' >/dev/null 2>&1 \
    || pgrep -f 'train\.py.*--name aug_confirm_winner_100ep' >/dev/null 2>&1 \
    || pgrep -f 'mamba run -n '"$MAMBA_ENV"' python scripts/train\.py.*aug_confirm_winner_100ep' >/dev/null 2>&1
}

_wait_confirm_or_run() {
  local resume_flag="${1:-}"
  if _confirm_job_done; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) pipeline: aug_confirm already complete — skipping"
    return 0
  fi
  if _confirm_in_flight; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) pipeline: aug_confirm in flight — waiting (not restarting)"
    while _confirm_in_flight; do
      sleep 60
      if _confirm_job_done; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) pipeline: aug_confirm finished"
        return 0
      fi
    done
  fi
  if _confirm_job_done; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) pipeline: aug_confirm complete"
    return 0
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) pipeline: starting aug_confirm ($MANIFEST_CONFIRM)"
  if [[ -n "$resume_flag" ]]; then
    _queue_py --manifest "$MANIFEST_CONFIRM" --run --resume
  else
    _queue_py --manifest "$MANIFEST_CONFIRM" --run
  fi
}

_run_pipeline() {
  local mode="$1"
  local resume_flag=""
  if [[ "$mode" == "resume" ]]; then
    resume_flag=1
  fi
  echo "========== gpu pipeline $mode $(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown) =========="
  _wait_confirm_or_run "$resume_flag"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) pipeline: starting full queue ($MANIFEST_FULL)"
  if [[ "$mode" == "resume" ]]; then
    _queue_py --manifest "$MANIFEST_FULL" --run --resume
  else
    _queue_py --manifest "$MANIFEST_FULL" --run
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) pipeline: complete"
}

if [[ "${1:-}" == "_pipeline_inner" ]]; then
  # Inner pipeline must survive terminal/agent disconnect.
  set +e
  shift
  _run_pipeline "${1:-run}"
  exit $?
fi

case "$MODE" in
  dry-run)
    _run_manifest --dry-run 2>&1 | tee reports/gpu_queue/dry_run_plan.txt
    ;;
  run)
    setsid nohup "$MAMBA" run -n "$MAMBA_ENV" python scripts/run_gpu_queue.py --manifest "$MANIFEST" --run >>"$NOHUP_LOG" 2>&1 < /dev/null &
    echo $! >"$NOHUP_PID"
    echo "started pid $(cat "$NOHUP_PID"); tail -f $NOHUP_LOG"
    ;;
  resume)
    setsid nohup "$MAMBA" run -n "$MAMBA_ENV" python scripts/run_gpu_queue.py --manifest "$MANIFEST" --run --resume >>"$NOHUP_LOG" 2>&1 < /dev/null &
    echo $! >"$NOHUP_PID"
    echo "resumed pid $(cat "$NOHUP_PID")"
    ;;
  pipeline-dry-run)
    echo "# phase 1: $MANIFEST_CONFIRM"
    _queue_py --manifest "$MANIFEST_CONFIRM" --dry-run
    echo
    echo "# phase 2: $MANIFEST_FULL"
    _queue_py --manifest "$MANIFEST_FULL" --dry-run 2>&1 | tee reports/gpu_queue/dry_run_plan.txt
    ;;
  pipeline-run)
    setsid nohup "$ROOT/scripts/run_gpu_queue.sh" _pipeline_inner run >>"$NOHUP_LOG" 2>&1 < /dev/null &
    echo $! >"$NOHUP_PID"
    echo "pipeline pid $(cat "$NOHUP_PID"); tail -f $NOHUP_LOG"
    ;;
  pipeline-resume)
    setsid nohup "$ROOT/scripts/run_gpu_queue.sh" _pipeline_inner resume >>"$NOHUP_LOG" 2>&1 < /dev/null &
    echo $! >"$NOHUP_PID"
    echo "pipeline resume pid $(cat "$NOHUP_PID"); tail -f $NOHUP_LOG"
    ;;
  close-phase-dry-run)
    _queue_py --manifest "$MANIFEST_CLOSE_PHASE_A" --dry-run 2>&1 | tee reports/gpu_queue/dry_run_close_phase_a.txt
    ;;
  close-phase-run)
    setsid nohup "$MAMBA" run -n "$MAMBA_ENV" python scripts/run_gpu_queue.py --manifest "$MANIFEST_CLOSE_PHASE_A" --run >>"$NOHUP_LOG" 2>&1 < /dev/null &
    echo $! >"$NOHUP_PID"
    echo "close Phase A pid $(cat "$NOHUP_PID"); tail -f $NOHUP_LOG"
    ;;
  close-phase-resume)
    setsid nohup "$MAMBA" run -n "$MAMBA_ENV" python scripts/run_gpu_queue.py --manifest "$MANIFEST_CLOSE_PHASE_A" --run --resume >>"$NOHUP_LOG" 2>&1 < /dev/null &
    echo $! >"$NOHUP_PID"
    echo "close Phase A resume pid $(cat "$NOHUP_PID")"
    ;;
  close-100ep-dry-run)
    _queue_py --manifest "$MANIFEST_CLOSE_100EP" --dry-run 2>&1 | tee reports/gpu_queue/dry_run_close_100ep.txt
    ;;
  close-100ep-run)
    setsid nohup "$MAMBA" run -n "$MAMBA_ENV" python scripts/run_gpu_queue.py --manifest "$MANIFEST_CLOSE_100EP" --run >>"$NOHUP_LOG" 2>&1 < /dev/null &
    echo $! >"$NOHUP_PID"
    echo "close 100ep tail pid $(cat "$NOHUP_PID"); tail -f $NOHUP_LOG"
    ;;
  close-100ep-resume)
    setsid nohup "$MAMBA" run -n "$MAMBA_ENV" python scripts/run_gpu_queue.py --manifest "$MANIFEST_CLOSE_100EP" --run --resume >>"$NOHUP_LOG" 2>&1 < /dev/null &
    echo $! >"$NOHUP_PID"
    echo "close 100ep resume pid $(cat "$NOHUP_PID")"
    ;;
  *)
    echo "usage: $0 {dry-run|run|resume|pipeline-dry-run|pipeline-run|pipeline-resume|close-phase-dry-run|close-phase-run|close-phase-resume|close-100ep-dry-run|close-100ep-run|close-100ep-resume}" >&2
    exit 2
    ;;
esac
