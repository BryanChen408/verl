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
under the legacy :mod:`verl.workers.megatron_workers` module path.

Why this exists
---------------
verl <= 0.7.x exposed backend-specific worker modules
(``megatron_workers.py``, ``fsdp_workers.py``) each defining their own
``ActorRolloutRefWorker`` / ``AsyncActorRolloutRefWorker`` / ``CriticWorker``.

The current fork merges them into a single backend-agnostic
:mod:`verl.workers.engine_workers` with just two classes:
    * ``ActorRolloutRefWorker`` — unified, async/sync chosen by config
      (``actor_rollout_ref.rollout.mode``)
    * ``TrainingWorker`` — generic training worker; the new critic
      role uses this (no more backend-specific ``CriticWorker``)

rllm <= 2026-05 still imports the legacy names (see e.g.
``rllm/trainer/verl/train_agent_ppo.py:105,134``). Without this shim
those imports raise ``ModuleNotFoundError: No module named
'verl.workers.megatron_workers'`` at trainer construct time.

Mappings
--------
================================  ===========================================
legacy symbol                     forwarded to
================================  ===========================================
ActorRolloutRefWorker             engine_workers.ActorRolloutRefWorker
AsyncActorRolloutRefWorker        engine_workers.ActorRolloutRefWorker
                                  (alias; async is now config-driven via
                                   ``actor_rollout_ref.rollout.mode=async``)
CriticWorker                      engine_workers.TrainingWorker
                                  (critic role is now a generic TrainingWorker
                                   specialization; ``TrainingWorkerConfig`` is
                                   required at construct time, which the legacy
                                   call sites don't pass — but stage1 GRPO
                                   doesn't instantiate critic, so the import
                                   succeeds and the runtime mismatch only
                                   surfaces if you actually use critic, in
                                   which case you should already be on the
                                   new engine_workers API)
================================  ===========================================

Remove once rllm upgrades to the engine_workers API.
"""

from __future__ import annotations

try:
    from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker
except ImportError as e:
    raise ImportError(
        "rllm-compat shim verl.workers.megatron_workers needs "
        "verl.workers.engine_workers to be importable but it isn't. "
        f"Underlying error: {e}. "
        "Either this fork is older than expected (no engine_workers) or "
        "engine_workers itself has a broken transitive import."
    ) from e


# async/sync is now config-driven inside the unified ActorRolloutRefWorker.
AsyncActorRolloutRefWorker = ActorRolloutRefWorker

# Critic role is now served by the generic TrainingWorker. Stage1 (GRPO,
# actor-only) does NOT instantiate critic, so this alias only needs to make
# imports succeed. If a caller actually instantiates CriticWorker(...) they
# will hit TrainingWorker's constructor which requires TrainingWorkerConfig,
# at which point they need to migrate to the engine_workers API.
CriticWorker = TrainingWorker


__all__ = [
    "ActorRolloutRefWorker",
    "AsyncActorRolloutRefWorker",
    "CriticWorker",
]
