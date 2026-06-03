"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import gc
import statistics
import time

import pytest
import torch


def _clear_trace_source_caches(template_mod):
    for name in (
        "_render_init_source",
        "_render_reference_source",
        "_get_callable_source",
    ):
        target = getattr(template_mod, name, None)
        cache_clear = getattr(target, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()


@pytest.fixture()
def isolated_trace_dump_state(monkeypatch):
    import flashinfer.trace.template as template_mod

    previous_dumped_names = set(template_mod._DUMPED_NAMES)
    previous_workload_counts = dict(template_mod._WORKLOAD_AXIS_COUNTS)
    previous_workload_record_count = template_mod._WORKLOAD_AXIS_RECORD_COUNT
    template_mod._DUMPED_NAMES.clear()
    template_mod._WORKLOAD_AXIS_COUNTS.clear()
    template_mod._WORKLOAD_AXIS_RECORD_COUNT = 0
    _clear_trace_source_caches(template_mod)
    monkeypatch.delenv("FLASHINFER_TRACE_DUMP", raising=False)
    monkeypatch.delenv("FLASHINFER_TRACE_DUMP_DIR", raising=False)
    monkeypatch.delenv("FLASHINFER_TRACE_WORKLOAD_DUMP", raising=False)
    monkeypatch.delenv("FLASHINFER_TRACE_WORKLOAD_DUMP_DIR", raising=False)
    try:
        yield
    finally:
        template_mod._DUMPED_NAMES.clear()
        template_mod._DUMPED_NAMES.update(previous_dumped_names)
        template_mod._WORKLOAD_AXIS_COUNTS.clear()
        template_mod._WORKLOAD_AXIS_COUNTS.update(previous_workload_counts)
        template_mod._WORKLOAD_AXIS_RECORD_COUNT = previous_workload_record_count
        _clear_trace_source_caches(template_mod)


def _median_ns_per_call(fn, *, calls=512, repeats=7):
    samples = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            for _ in range(calls):
                fn()
            samples.append((time.perf_counter_ns() - start) / calls)
    finally:
        if gc_was_enabled:
            gc.enable()
    return statistics.median(samples)


def _make_rmsnorm_inputs(batch_size, hidden_size=16):
    return (
        torch.empty((batch_size, hidden_size), dtype=torch.bfloat16),
        torch.empty((hidden_size,), dtype=torch.bfloat16),
    )


def _make_wrapped_rmsnorm_noop():
    from flashinfer.api_logging import _attach_fi_trace
    from flashinfer.trace.templates.norm import rmsnorm_trace

    def rmsnorm_noop(input, weight, eps=1e-6, out=None):
        del weight, eps, out
        return input

    return _attach_fi_trace(rmsnorm_noop, rmsnorm_noop, trace_template=rmsnorm_trace)


def test_fi_trace_workload_axis_overhead_stays_low(
    tmp_path, monkeypatch, isolated_trace_dump_state
):
    from flashinfer.trace.template import flush_workload_axis_dumps

    wrapped = _make_wrapped_rmsnorm_noop()
    x, weight = _make_rmsnorm_inputs(4)

    monkeypatch.setenv("FLASHINFER_TRACE_DUMP_DIR", str(tmp_path / "defs"))
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP", "1")
    monkeypatch.setenv("FLASHINFER_TRACE_WORKLOAD_DUMP_DIR", str(tmp_path / "workloads"))
    monkeypatch.setenv("FLASHINFER_TRACE_WORKLOAD_DUMP", "1")
    wrapped(x, weight)

    enabled_ns = _median_ns_per_call(lambda: wrapped(x, weight), calls=2048)
    monkeypatch.setenv("FLASHINFER_TRACE_WORKLOAD_DUMP", "0")
    dump_only_ns = _median_ns_per_call(lambda: wrapped(x, weight), calls=2048)
    workload_overhead_ns = max(0.0, enabled_ns - dump_only_ns)

    assert workload_overhead_ns < 15_000, (
        "FI trace workload-axis collection overhead regressed: "
        f"enabled={enabled_ns:.1f} ns/call dump_only={dump_only_ns:.1f} ns/call "
        f"overhead={workload_overhead_ns:.1f} ns/call"
    )
    assert flush_workload_axis_dumps() == 1
