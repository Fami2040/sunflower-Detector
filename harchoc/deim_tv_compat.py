"""Torchvision 0.21 compat for upstream DEIM (no edits under ``external/DEIM``).

DEIM's ``ConvertPILImage`` / ``ConvertBoxes`` only define ``_transform()``; torchvision
v2 ``Transform.forward()`` calls ``transform()`` → ``NotImplementedError``. D-FINE
already ships the ``transform()`` shim; we apply the same at runtime after ``external/DEIM``
is on ``sys.path`` (see ``run_external_train.py``).
"""

from __future__ import annotations

from typing import Any

_APPLIED = False


def apply_deim_torchvision_compat() -> None:
    """Idempotent monkeypatch for DEIM transform classes (call once per process)."""
    global _APPLIED
    if _APPLIED:
        return
    from engine.data.transforms import _transforms as deim_transforms

    def _patch(cls: type) -> None:
        if getattr(cls, "_harchoc_tv21_transform", False):
            return

        def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
            return self._transform(inpt, params)

        cls.transform = transform  # type: ignore[method-assign]
        cls._harchoc_tv21_transform = True

    _patch(deim_transforms.ConvertPILImage)
    _patch(deim_transforms.ConvertBoxes)
    _APPLIED = True
