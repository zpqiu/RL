# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class SetupTimingMetrics:
    """Driver-side per-phase timings collected during setup."""

    # Generation-backend init.
    generation_init_time_s: Optional[float] = None
    # When overlapping NeMo Gym init, the total decomposes as reserve + load.
    generation_init_reserve_time_s: Optional[float] = None
    generation_init_load_time_s: Optional[float] = None

    policy_init_time_s: Optional[float] = None
    # PPO only: the critic shares the training GPUs, so it is built after the policy.
    value_init_time_s: Optional[float] = None
    nemo_gym_init_time_s: Optional[float] = None
    collective_init_time_s: Optional[float] = None
    # Non-colocated megatron's post-init weight sync into the engine.
    weight_sync_time_s: Optional[float] = None

    # Non-colocated only. (grpo.py only)
    parallel_wall_time_s: Optional[float] = None
    parallel_init_enabled: Optional[float] = None

    # Optional setup phases. OPD teacher timings are shared by legacy GRPO and SC;
    # sparse refit and checkpoint-engine timings remain legacy-GRPO-only.
    teacher_reservation_time_s: Optional[float] = None
    teacher_model_init_time_s: Optional[float] = None
    teacher_init_time_s: Optional[float] = None
    vllm_checkpoint_engine_init_time_s: Optional[float] = None

    total_setup_time_s: Optional[float] = None
    worker_setup_time_s: Optional[float] = None
    other_setup_time_s: Optional[float] = None

    # Overflow bucket for dynamic-keyed metrics (e.g. one entry per active
    # sparse refit transport: vllm_<transport>_sparse_init_time_s).
    extras: dict[str, float] = field(default_factory=dict)

    def to_metrics_dict(self) -> dict[str, Any]:
        """Serialize for Logger.log_metrics; drops unset (None) fields."""
        base = {
            k: v for k, v in asdict(self).items() if k != "extras" and v is not None
        }
        base.update(self.extras)
        return base


def print_setup_timing_summary(metrics: SetupTimingMetrics) -> None:
    """Print the setup-phase summary block.

    Args:
        metrics: Populated timing metrics.
    """
    print("\n▶ Worker Initialization Timing:")

    assert metrics.generation_init_time_s is not None
    if metrics.generation_init_reserve_time_s:
        # gym-on: an address was reserved, so the total decomposes as reserve + load.
        print(
            f"  Generation init: {metrics.generation_init_time_s:.1f}s"
            f" (reserve {metrics.generation_init_reserve_time_s:.1f}s"
            f" + load {metrics.generation_init_load_time_s:.1f}s)"
        )
    else:
        print(f"  Generation init: {metrics.generation_init_time_s:.1f}s")

    print(f"  Policy init: {metrics.policy_init_time_s:.1f}s")

    if metrics.value_init_time_s:
        print(f"  Value init: {metrics.value_init_time_s:.1f}s")

    if metrics.nemo_gym_init_time_s:
        print(f"  NeMo-Gym init: {metrics.nemo_gym_init_time_s:.1f}s")

    if metrics.teacher_init_time_s:
        print(f"  Teacher init: {metrics.teacher_init_time_s:.1f}s")

    print(f"  Other setup: {metrics.other_setup_time_s:.1f}s")
    print(f"  Total setup: {metrics.total_setup_time_s:.1f}s", flush=True)


def extract_vllm_request_time_metrics(
    metrics: dict[str, Any] | None,
) -> dict[str, float]:
    """Remove cumulative latency samples and return request-weighted window means.

    Series are separated by actor and Prometheus labels. Reset counters are
    excluded; windows with no completed requests do not report a false zero.
    """
    if not metrics:
        return {}
    totals: dict[str, tuple[float, int]] = {}
    for series in metrics.pop("request_time_histograms", {}).values():
        for (name, _labels), snapshots in series.items():
            if len(snapshots) < 2:
                continue
            first_sum, first_count = snapshots[0]
            last_sum, last_count = snapshots[-1]
            count = last_count - first_count
            elapsed = last_sum - first_sum
            if count <= 0 or elapsed < 0:
                continue
            total_sum, total_count = totals.get(name, (0.0, 0))
            totals[name] = (total_sum + elapsed, total_count + count)
    return {
        "vllm/" + name.removeprefix("vllm:").removesuffix("_seconds") + "_mean_s": total
        / count
        for name, (total, count) in totals.items()
    }
