"""Sequential GPU backlog queue: manifest-driven jobs with staged logging."""

from __future__ import annotations

from harchoc.equivalence_index import (
    complete_preds_sha_owners as _complete_preds_sha_owners,
    filter_duplicate_preds_sha as _filter_duplicate_preds_sha,
    index_preds_sha_by_smoke_id as _index_preds_sha_by_smoke_id,
    job_dedup_id as _job_dedup_id,
    parse_equivalence_classes,
    preds_sha_for_job as _job_preds_sha_for_dedup,
    preds_sha_from_summary_obj as _preds_sha_from_summary_obj,
    preds_sha_from_verified_summary_path as _preds_sha_from_verified_summary_path,
)
from harchoc.gpu_queue_dedup import (
    complete_recipe_owners as _complete_recipe_owners,
    filter_duplicate_train_recipes as _filter_duplicate_train_recipes,
    job_train_recipe_fingerprint as _job_train_recipe_fingerprint,
)
from harchoc.gpu_queue_manifest import (
    AUG_SMOKE_PENDING_STATUSES,
    GPU_QUEUE_MANIFEST_SCHEMA,
    expand_aug_smoke_jobs_from_index,
    load_gpu_queue_manifest,
    merge_aug_smoke_jobs as _merge_aug_smoke_jobs,
)
from harchoc.gpu_queue_runner import (
    DEFAULT_EVAL_OUT_DIR,
    DEFAULT_JOBS_ROOT,
    DEFAULT_LOG_ROOT,
    DEFAULT_MIN_FREE_MIB,
    DEFAULT_STATE_PATH,
    DEFAULT_SUMMARIES_ROOT,
    GPU_QUEUE_RUN_SCHEMA,
    GpuQueueError,
    load_run_state,
    repair_resume_state,
    run_gpu_queue,
    run_job,
    save_run_state,
    _run_internal_stage,
    _run_subprocess_stage,
)
from harchoc.gpu_queue_skip import (
    DEFAULT_GPU_POLL_S,
    _prune_dry_run_log_stubs,
    should_skip_job,
    wait_gpu_free,
)
from harchoc.gpu_queue_stages import build_job_stages, validate_job_files

_validate_job_files = validate_job_files

__all__ = [
    "AUG_SMOKE_PENDING_STATUSES",
    "DEFAULT_EVAL_OUT_DIR",
    "DEFAULT_GPU_POLL_S",
    "DEFAULT_JOBS_ROOT",
    "DEFAULT_LOG_ROOT",
    "DEFAULT_MIN_FREE_MIB",
    "DEFAULT_STATE_PATH",
    "DEFAULT_SUMMARIES_ROOT",
    "GPU_QUEUE_MANIFEST_SCHEMA",
    "GPU_QUEUE_RUN_SCHEMA",
    "GpuQueueError",
    "build_job_stages",
    "expand_aug_smoke_jobs_from_index",
    "load_gpu_queue_manifest",
    "load_run_state",
    "parse_equivalence_classes",
    "repair_resume_state",
    "run_gpu_queue",
    "run_job",
    "save_run_state",
    "should_skip_job",
    "validate_job_files",
    "wait_gpu_free",
]
