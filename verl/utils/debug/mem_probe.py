# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host / device memory probes for OOM debugging on NPU.

Two utilities:

* ``mem_snapshot(tag)``: prints a one-line snapshot of process RSS, system memory,
  NPU allocation, and the bytes freed by an immediate ``gc.collect()``. Cheap
  enough to call at every phase transition in the training loop.
* ``opt_offload_check(engine, tag)``: one-shot diagnostic that reports the
  optimizer type, whether it is the distributed variant, the offload flags on
  the engine config, and an estimate of the optimizer state size. Call once
  after the optimizer is built.

Output formats (all single-line, grep-friendly):
    [MEM r0 step=1 rollout_done] proc_rss=12.3GB ...
    [OPT-CHECK r0 init] type=DistributedOptimizer is_distributed=True ...

Disable everything by exporting ``VERL_MEM_PROBE=0``.
"""

import gc
import os
import time
from typing import Any, Optional

# Cache the psutil Process handle so probing is cheap.
_proc: Any = None


def _enabled() -> bool:
    return os.environ.get("VERL_MEM_PROBE", "1") != "0"


def _get_proc():
    global _proc
    if _proc is None:
        try:
            import psutil  # noqa: WPS433 (local import for optional dep)

            _proc = psutil.Process()
        except Exception:
            _proc = False  # mark "tried and failed" so we don't re-import
    return _proc if _proc else None


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def mem_snapshot(tag: str, only_rank: Optional[int] = 0, force_gc: bool = True) -> None:
    """Print a single-line memory snapshot.

    Args:
        tag: Free-form label identifying the phase being entered/exited.
        only_rank: If set, only emit when ``RANK`` env matches this value. The
            default (``0``) keeps logs from exploding across all workers.
            Set to ``None`` to emit on every rank.
        force_gc: Call ``gc.collect()`` and report freed bytes. A large
            ``gc_freed`` value usually means stale Python references; near-zero
            with rising RSS points at C++ / NPU driver allocations.
    """
    if not _enabled():
        return
    if only_rank is not None and _rank() != only_rank:
        return

    proc = _get_proc()
    if proc is None:
        return

    try:
        import psutil  # noqa: WPS433

        rss = proc.memory_info().rss / 1024**3
        vms = proc.memory_info().vms / 1024**3
        sys_mem = psutil.virtual_memory()
        sys_used = sys_mem.used / 1024**3
        sys_avail = sys_mem.available / 1024**3

        # NPU memory (only if torch_npu is wired up).
        npu_alloc = -1.0
        npu_reserved = -1.0
        try:
            import torch  # noqa: WPS433

            if hasattr(torch, "npu") and torch.npu.is_available():
                npu_alloc = torch.npu.memory_allocated() / 1024**3
                npu_reserved = torch.npu.memory_reserved() / 1024**3
        except Exception:
            pass

        gc_freed = 0.0
        if force_gc:
            before = proc.memory_info().rss
            gc.collect()
            after = proc.memory_info().rss
            gc_freed = max(before - after, 0) / 1024**3

        ts = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"

        print(
            f"[MEM r{_rank()} {tag}] proc_rss={rss:.1f}GB vms={vms:.1f}GB "
            f"sys_used={sys_used:.0f}GB avail={sys_avail:.0f}GB "
            f"npu_alloc={npu_alloc:.1f}GB npu_reserved={npu_reserved:.1f}GB "
            f"gc_freed={gc_freed:.2f}GB t={ts}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[MEM r{_rank()} {tag}] probe failed: {type(exc).__name__}: {exc}",
            flush=True,
        )


def opt_offload_check(engine: Any, tag: str = "init") -> None:
    """One-shot diagnostic for distributed optimizer + offload state.

    Args:
        engine: The ``MegatronEngine`` (or subclass) instance, after the
            optimizer has been built.
        tag: Free-form label, surfaces in the log line.

    Reports:
        * Optimizer class name and whether it's Megatron's ``DistributedOptimizer``.
        * The three offload flags off ``engine.engine_config``.
        * A rough total of optimizer state tensor bytes (sums ``numel *
          element_size`` over ``optimizer.state``). Compare against expected
          ``2 * model_params * 4B / DP`` for AdamW + distributed optimizer.
    """
    if not _enabled():
        return
    if _rank() != 0:
        return

    try:
        opt = getattr(engine, "optimizer", None)
        opt_type = type(opt).__name__ if opt is not None else "None"

        is_dist_opt = False
        try:
            from megatron.core.optimizer.distrib_optimizer import (  # noqa: WPS433
                DistributedOptimizer,
            )

            is_dist_opt = isinstance(opt, DistributedOptimizer)
        except Exception:
            pass

        engine_config = getattr(engine, "engine_config", None)
        param_offload = getattr(engine_config, "param_offload", "?")
        grad_offload = getattr(engine_config, "grad_offload", "?")
        opt_offload = getattr(engine_config, "optimizer_offload", "?")
        use_dist_opt = getattr(engine_config, "use_distributed_optimizer", "?")

        opt_state_gb = -1.0
        try:
            if opt is not None and hasattr(opt, "state"):
                total_bytes = 0
                for group_state in opt.state.values():
                    for value in group_state.values():
                        if hasattr(value, "numel") and hasattr(value, "element_size"):
                            total_bytes += value.numel() * value.element_size()
                opt_state_gb = total_bytes / 1024**3
        except Exception:
            pass

        print(
            f"[OPT-CHECK r{_rank()} {tag}] type={opt_type} is_distributed={is_dist_opt} "
            f"use_distributed_optimizer_cfg={use_dist_opt} "
            f"param_offload={param_offload} grad_offload={grad_offload} "
            f"opt_offload={opt_offload} opt_state_size={opt_state_gb:.2f}GB",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[OPT-CHECK r{_rank()} {tag}] failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
