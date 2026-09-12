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

"""CPU contracts for diagnostics; extract helpers to avoid importing GPU actors."""

import ast
import copy
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Optional
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.algorithms.metric_utils import extract_vllm_request_time_metrics


def load_helper(relative_path, name):
    source = Path(__file__).resolve().parents[2] / relative_path
    tree = ast.parse(source.read_text())
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    namespace = {
        "torch": torch,
        "Optional": Optional,
        "BatchedDataDict": dict,
        "Any": Any,
        "Iterator": Iterator,
        "copy": copy,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.int64]
)
def test_refit_budget_covers_payload_and_logical_weights(dtype):
    estimate = load_helper(
        "nemo_rl/models/policy/workers/megatron_policy_worker.py",
        "_estimate_refit_tensor_size_in_bytes",
    )
    param = torch.empty(3, dtype=dtype)
    size = estimate(param, export_dtype=torch.bfloat16, tp_size=2, ep_size=4)
    assert size >= param.numel() * param.element_size() * 8
    if param.is_floating_point():
        assert size >= 3 * 2 * 8


def test_logprob_means_are_token_weighted_and_pre_filter():
    compute = load_helper(
        "nemo_rl/algorithms/grpo.py", "compute_and_apply_seq_logprob_error_masking"
    )
    data = {
        "token_mask": torch.tensor([[0, 1, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1]]),
        "sample_mask": torch.tensor([1, 1, 0]),
        "prev_logprobs": torch.tensor(
            [[99.0, -1.0, 0.0, 0.0], [99.0, -3.0, -3.0, -3.0], [99.0, 80.0, 80.0, 80.0]]
        ),
        "generation_logprobs": torch.tensor(
            [[0.0, -2.0, 0.0, 0.0], [0.0, -2.0, -2.0, -2.0], [0.0, 0.0, 0.0, 0.0]]
        ),
    }
    result = compute(data, torch.zeros(3), 1.5)
    assert result["token_logprob_diff_mean"] == pytest.approx(-0.5)
    assert result["token_logprob_abs_diff_mean"] == pytest.approx(1.0)
    assert result["token_logprob_diff_valid_tokens"] == 4
    assert data["sample_mask"].sum() == 0


def test_logprob_means_empty_and_skipped_are_not_false_zero():
    compute = load_helper(
        "nemo_rl/algorithms/grpo.py", "compute_and_apply_seq_logprob_error_masking"
    )
    data = {
        key: torch.zeros(1, 3)
        for key in ("token_mask", "prev_logprobs", "generation_logprobs")
    }
    data["sample_mask"] = torch.ones(1)
    result = compute(data, torch.zeros(1), None)
    assert result["token_logprob_diff_valid_tokens"] == 0
    assert "token_logprob_diff_mean" not in result
    assert "seq_logprob_abs_diff_mean" not in result
    placeholder = load_helper(
        "nemo_rl/algorithms/grpo.py", "_placeholder_seq_logprob_error_metrics"
    )()
    assert placeholder["token_logprob_diff_valid_tokens"] == 0
    assert "token_logprob_diff_mean" not in placeholder
    assert "seq_logprob_abs_diff_mean" not in placeholder


def test_logprob_masked_nan_does_not_pollute_token_mean():
    compute = load_helper(
        "nemo_rl/algorithms/grpo.py", "compute_and_apply_seq_logprob_error_masking"
    )
    data = {
        "token_mask": torch.tensor([[0, 1, 0]]),
        "sample_mask": torch.ones(1),
        "prev_logprobs": torch.tensor([[0.0, -1.0, float("nan")]]),
        "generation_logprobs": torch.tensor([[0.0, -2.0, 0.0]]),
    }
    result = compute(data, torch.zeros(1), None)
    assert result["token_logprob_diff_mean"] == 1.0
    assert result["seq_logprob_abs_diff_mean"] == 1.0


def test_sequence_mean_weights_nonempty_sequences_equally_before_filter():
    compute = load_helper(
        "nemo_rl/algorithms/grpo.py", "compute_and_apply_seq_logprob_error_masking"
    )
    data = {
        "token_mask": torch.tensor(
            [[0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0], [0, 1, 1, 1]]
        ),
        "sample_mask": torch.tensor([1, 1, 1, 0]),
        "prev_logprobs": torch.tensor(
            [
                [0.0, -1.0, 0.0, 0.0],
                [0.0, -3.0, -4.0, -5.0],
                [0.0, 90.0, 90.0, 90.0],
                [0.0, 90.0, 90.0, 90.0],
            ]
        ),
        "generation_logprobs": torch.zeros(4, 4),
    }
    result = compute(data, torch.zeros(4), 1.5)
    # Sequence means are 1 and 4; empty and sample-masked rows are excluded.
    assert result["seq_logprob_abs_diff_mean"] == pytest.approx(2.5)
    assert result["token_logprob_abs_diff_mean"] == pytest.approx(3.25)
    assert result["token_logprob_diff_valid_tokens"] == 4
    assert data["sample_mask"][:2].sum() == 0


def test_latency_means_weight_requests_and_keep_label_series_separate():
    name = "vllm:request_queue_time_seconds"
    metrics = {
        "inflight_batch_sizes": {0: [1]},
        "request_time_histograms": {
            0: {
                (name, (("engine", "0"),)): [(10.0, 2), (14.0, 4)],
                (name, (("engine", "1"),)): [(100.0, 20), (130.0, 23)],
            },
            1: {(name, ()): [(0.0, 0), (6.0, 1)]},
        },
    }
    assert extract_vllm_request_time_metrics(metrics) == {
        "vllm/request_queue_time_mean_s": pytest.approx(40.0 / 6)
    }
    assert metrics == {"inflight_batch_sizes": {0: [1]}}


@pytest.mark.parametrize(
    "snapshots",
    [
        [],
        [(1.0, 1)],
        [(1.0, 1), (1.0, 1)],
        [(10.0, 4), (1.0, 1)],
        [(10.0, 1), (1.0, 4)],
    ],
)
def test_latency_empty_and_reset_windows_are_omitted(snapshots):
    metrics = {
        "request_time_histograms": {
            0: {("vllm:request_decode_time_seconds", ()): snapshots}
        }
    }
    assert extract_vllm_request_time_metrics(metrics) == {}


def test_hf_refit_iterator_preserves_mixed_payload_dtypes(monkeypatch):
    module = "nemo_rl.models.generation.vllm.quantization.fp8_train_utils"
    monkeypatch.setitem(
        sys.modules, module, SimpleNamespace(get_vllm_qkv_scale_names=MagicMock())
    )
    export = load_helper(
        "nemo_rl/models/policy/workers/megatron_policy_worker.py",
        "_iter_params_with_optional_kv_scales",
    )
    weights = [
        ("weight", torch.empty(4, dtype=torch.float8_e4m3fn)),
        ("weight_scale_inv", torch.ones(1, dtype=torch.float32)),
        ("norm", torch.ones(2, dtype=torch.bfloat16)),
    ]
    bridge = SimpleNamespace(export_hf_weights=MagicMock(return_value=iter(weights)))
    worker = SimpleNamespace(
        refit_conversion_tasks=[],
        refit_payload_mode="hf_export",
        model=object(),
        megatron_bridge=bridge,
        cfg={},
    )
    result = list(export(worker, include_draft=False))
    for (name, tensor), (expected_name, expected_tensor) in zip(
        result, weights, strict=True
    ):
        assert name == expected_name
        assert tensor is expected_tensor
    assert "weight_dtype" not in bridge.export_hf_weights.call_args.kwargs


def test_latency_clear_keeps_only_last_baseline_and_get_is_a_copy():
    path = "nemo_rl/models/generation/vllm/vllm_worker_async.py"
    clear = load_helper(path, "clear_vllm_logger_metrics")
    get = load_helper(path, "get_vllm_logger_metrics")
    key = ("vllm:request_decode_time_seconds", ())
    worker = SimpleNamespace(
        cfg={"vllm_cfg": {"enable_vllm_metrics_logger": True}},
        _vllm_metrics_lock=threading.Lock(),
        inflight_batch_sizes=[1],
        num_pending_samples=[2],
        kv_cache_usage_perc=[0.1],
        generation_tokens=[10],
        request_time_histograms={key: [(1.0, 1), (7.0, 3)]},
    )
    saved = get(worker)
    clear(worker)
    assert worker.request_time_histograms == {key: [(7.0, 3)]}
    assert saved["request_time_histograms"][key] == [(1.0, 1), (7.0, 3)]
    assert worker.generation_tokens == []
