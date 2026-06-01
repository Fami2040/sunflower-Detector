#!/usr/bin/env bash
# Stop ad-hoc subagent/orchestrator GPU jobs (NOT gpu_queue).
# Use before ./scripts/run_gpu_queue.sh run so one owner holds the GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Stopping stray trains/orchestrators (gpu_queue / run_gpu_queue.py untouched)..."

for pat in \
  'scripts/train.py' \
  'scripts/benchmark_matrix.py' \
  'scripts/finetune.py' \
  'python -u scripts/train' \
  'python -u train.py' \
  'mamba run.*train.py' \
  'mamba run.*benchmark_matrix' \
  'from scripts.train import main' \
  'torch.distributed.run' \
  'harchoc_train_overlay' \
  'runs/hsp_zoo/deim_' \
  'runs/hsp_zoo/dfine_' \
  'runs/hsp_zoo/rtdetrv2_' \
  '/tmp/run_rtdetr' \
  '/tmp/p1_rtdetr' \
  '/tmp/wait_s2' \
  'orchestrator' \
  'batch_probe_rtdetr' \
  'aug_smoke_close3' \
  'rtdetr_queries_smoke'
do
  pkill -9 -f "$pat" 2>/dev/null || true
done

STUB='#!/usr/bin/env bash
echo "DISABLED: use ./scripts/run_gpu_queue.sh — ad-hoc orchestrators superseded" >&2
exit 1
'
for f in /tmp/run_rtdetr*.sh /tmp/p1_rtdetr*.sh /tmp/wait_s2*.sh /tmp/run_rtdetr_queries*.sh; do
  [ -f "$f" ] && printf '%s' "$STUB" > "$f" && chmod -x "$f" && echo "stubbed $f"
done

echo "--- GPU compute apps ---"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || nvidia-smi
remaining=$(ps aux | grep -E 'train\.py|benchmark_matrix|finetune\.py|harchoc_train_overlay|torch\.distributed\.run|/tmp/run_|/tmp/p1_|orchestrator' | grep -v grep | grep -v kill_stray | grep -v run_gpu_queue || true)
if [ -n "$remaining" ]; then
  echo "WARNING: still running:" >&2
  echo "$remaining" >&2
  exit 1
fi
echo "OK: no stray GPU trains"

python3 -c "from harchoc.gpu_exclusive import acquire_gpu_exclusive; acquire_gpu_exclusive(repo_root='$ROOT', owner='kill_stray_gpu_jobs')"
echo "GPU exclusive lock set: reports/gpu_queue/GPU_EXCLUSIVE.lock (blocks ad-hoc train.py)"
