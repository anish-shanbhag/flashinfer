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
import inspect
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
        cache_clear = getattr(getattr(template_mod, name), "cache_clear", None)
        if cache_clear is not None:
            cache_clear()


@pytest.fixture()
def isolated_trace_dump_state(monkeypatch):
    import flashinfer.trace.template as template_mod

    previous_dumped_names = set(template_mod._DUMPED_NAMES)
    previous_workload_counts = dict(template_mod._WORKLOAD_AXIS_COUNTS)
    previous_workload_record_count = template_mod._WORKLOAD_AXIS_RECORD_COUNT
    previous_workload_flush_every = template_mod._WORKLOAD_AXIS_FLUSH_EVERY
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
        template_mod._WORKLOAD_AXIS_FLUSH_EVERY = previous_workload_flush_every
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


def _elapsed_ns(fn):
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter_ns()
        fn()
        return time.perf_counter_ns() - start
    finally:
        if gc_was_enabled:
            gc.enable()


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


def test_fi_trace_auto_dump_fast_binding_preserves_defaults(
    monkeypatch, isolated_trace_dump_state
):
    from flashinfer.api_logging import _attach_fi_trace

    seen_kwargs = []

    def trace_dispatch(save_dir=None, name=None, **kwargs):
        del save_dir, name
        seen_kwargs.append(kwargs)
        return None

    def api(input, weight, eps=1e-6, *, out=None):
        del weight, eps, out
        return input

    wrapped = _attach_fi_trace(api, api, trace_template=trace_dispatch)
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP", "1")

    x, weight = _make_rmsnorm_inputs(2)
    wrapped(x, weight, out=x)

    assert seen_kwargs == [
        {
            "input": x,
            "weight": weight,
            "eps": 1e-6,
            "out": x,
        }
    ]


def test_fi_trace_fast_binding_stays_below_signature_bind_cost(
    isolated_trace_dump_state,
):
    from flashinfer.api_logging import _build_fast_bound_arguments

    def api(input, weight, eps=1e-6, *, out=None):
        del input, weight, eps, out

    x, weight = _make_rmsnorm_inputs(2)
    sig = inspect.signature(api)
    fast_bind = _build_fast_bound_arguments(sig)
    assert fast_bind is not None

    def signature_bind():
        bound = sig.bind(x, weight, out=x)
        bound.apply_defaults()
        dict(bound.arguments)

    def direct_bind():
        fast_bind((x, weight), {"out": x})

    signature_ns = _median_ns_per_call(signature_bind, calls=4096)
    direct_ns = _median_ns_per_call(direct_bind, calls=4096)

    assert direct_ns < signature_ns * 0.75, (
        "FI trace fast argument binding regressed: "
        f"direct={direct_ns:.1f} ns/call signature={signature_ns:.1f} ns/call"
    )


def test_fi_trace_auto_dump_hot_repeated_shape_overhead_stays_low(
    tmp_path, monkeypatch, isolated_trace_dump_state
):
    wrapped = _make_wrapped_rmsnorm_noop()
    x, weight = _make_rmsnorm_inputs(4)

    monkeypatch.setenv("FLASHINFER_TRACE_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP", "1")
    wrapped(x, weight)

    enabled_ns = _median_ns_per_call(lambda: wrapped(x, weight), calls=2048)
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP", "0")
    disabled_ns = _median_ns_per_call(lambda: wrapped(x, weight), calls=2048)
    overhead_ns = max(0.0, enabled_ns - disabled_ns)

    assert overhead_ns < 15_000, (
        "Hot repeated-shape FI trace auto-dump overhead regressed: "
        f"enabled={enabled_ns:.1f} ns/call disabled={disabled_ns:.1f} ns/call "
        f"overhead={overhead_ns:.1f} ns/call"
    )
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_fi_trace_auto_dump_mixed_repeated_shape_overhead_stays_low(
    tmp_path, monkeypatch, isolated_trace_dump_state
):
    wrapped = _make_wrapped_rmsnorm_noop()
    inputs = [_make_rmsnorm_inputs(batch_size) for batch_size in range(1, 9)]
    idx = 0

    monkeypatch.setenv("FLASHINFER_TRACE_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP", "1")
    for x, weight in inputs:
        wrapped(x, weight)

    def call_next():
        nonlocal idx
        x, weight = inputs[idx % len(inputs)]
        idx += 1
        wrapped(x, weight)

    enabled_ns = _median_ns_per_call(call_next, calls=2048)
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP", "0")
    idx = 0
    disabled_ns = _median_ns_per_call(call_next, calls=2048)
    overhead_ns = max(0.0, enabled_ns - disabled_ns)

    assert overhead_ns < 20_000, (
        "Mixed repeated-shape FI trace auto-dump overhead regressed: "
        f"enabled={enabled_ns:.1f} ns/call disabled={disabled_ns:.1f} ns/call "
        f"overhead={overhead_ns:.1f} ns/call"
    )
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_fi_trace_workload_axis_overhead_stays_low(
    tmp_path, monkeypatch, isolated_trace_dump_state
):
    import flashinfer.trace.template as template_mod
    from flashinfer.trace.template import flush_workload_axis_dumps

    wrapped = _make_wrapped_rmsnorm_noop()
    x, weight = _make_rmsnorm_inputs(4)

    template_mod._WORKLOAD_AXIS_FLUSH_EVERY = 1_000_000
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


def test_fi_trace_dump_plus_workload_axis_combined_overhead_stays_low(
    tmp_path, monkeypatch, isolated_trace_dump_state
):
    import flashinfer.trace.template as template_mod
    from flashinfer.trace.template import flush_workload_axis_dumps

    wrapped = _make_wrapped_rmsnorm_noop()
    x, weight = _make_rmsnorm_inputs(4)

    template_mod._WORKLOAD_AXIS_FLUSH_EVERY = 1_000_000
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP_DIR", str(tmp_path / "defs"))
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP", "1")
    monkeypatch.setenv("FLASHINFER_TRACE_WORKLOAD_DUMP_DIR", str(tmp_path / "workloads"))
    monkeypatch.setenv("FLASHINFER_TRACE_WORKLOAD_DUMP", "1")
    wrapped(x, weight)

    enabled_ns = _median_ns_per_call(lambda: wrapped(x, weight), calls=2048)
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP", "0")
    monkeypatch.setenv("FLASHINFER_TRACE_WORKLOAD_DUMP", "0")
    disabled_ns = _median_ns_per_call(lambda: wrapped(x, weight), calls=2048)
    overhead_ns = max(0.0, enabled_ns - disabled_ns)

    assert overhead_ns < 15_000, (
        "Combined FI trace dump + workload-axis overhead regressed: "
        f"enabled={enabled_ns:.1f} ns/call disabled={disabled_ns:.1f} ns/call "
        f"overhead={overhead_ns:.1f} ns/call"
    )
    assert len(list((tmp_path / "defs").glob("*.json"))) == 1
    assert flush_workload_axis_dumps() == 1


def test_fi_trace_unique_shape_generation_amortizes_source_rendering(
    isolated_trace_dump_state,
):
    import flashinfer.trace.template as template_mod
    from flashinfer.trace.templates.norm import rmsnorm_trace

    _clear_trace_source_caches(template_mod)
    fi_trace_fn = rmsnorm_trace.build_fi_trace_fn("flashinfer.tests.rmsnorm_perf")
    first_x, first_weight = _make_rmsnorm_inputs(1)
    first_ns = _elapsed_ns(lambda: fi_trace_fn(input=first_x, weight=first_weight))

    inputs = [_make_rmsnorm_inputs(batch_size) for batch_size in range(2, 66)]
    idx = 0

    def call_unique_shape():
        nonlocal idx
        x, weight = inputs[idx % len(inputs)]
        idx += 1
        fi_trace_fn(input=x, weight=weight)

    steady_ns = _median_ns_per_call(call_unique_shape, calls=len(inputs), repeats=5)

    assert steady_ns < first_ns * 0.60, (
        "FI trace source rendering is no longer amortized across definitions: "
        f"first={first_ns:.1f} ns steady={steady_ns:.1f} ns/call"
    )
