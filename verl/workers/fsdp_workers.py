# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
rllm-compat shim — re-export :mod:`verl.workers.engine_workers` symbols
under the legacy :mod:`verl.workers.fsdp_workers` module path.

Companion to :mod:`verl.workers.megatron_workers` (same shim, different
legacy module name). See that file for the full rationale.

Stage1 uses the megatron strategy, so this shim is defensive only — it
exists so that ``train_agent_ppo.py``'s if/elif on
``config.actor_rollout_ref.actor.strategy`` doesn't import-fail when the
fsdp branch is reached during module load (e.g. via static analysis or
inadvertent eager imports).
"""

from __future__ import annotations

try:
    from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker
except ImportError as e:
    raise ImportError(
        "rllm-compat shim verl.workers.fsdp_workers needs "
        "verl.workers.engine_workers to be importable but it isn't. "
        f"Underlying error: {e}. "
        "Either this fork is older than expected (no engine_workers) or "
        "engine_workers itself has a broken transitive import."
    ) from e


AsyncActorRolloutRefWorker = ActorRolloutRefWorker
CriticWorker = TrainingWorker


__all__ = [
    "ActorRolloutRefWorker",
    "AsyncActorRolloutRefWorker",
    "CriticWorker",
]
