"""Benchmark harness tests — offline end-to-end over a method subset."""
from __future__ import annotations

import pytest

from core import Runtime

from .test_architectures import build_fake_llm


@pytest.fixture(scope="module")
def resources():
    from benchmark import build_resources

    return build_resources(Runtime.for_testing(llm=build_fake_llm()),
                           with_graph=False, with_raptor=False, with_chunk_variants=False)


def test_run_subset_shape(resources):
    from benchmark import run

    results = run(method_names=["naive", "hybrid"], limit=3, resources=resources)
    assert set(results["methods"]) == {"naive", "hybrid"}
    assert results["ranked"] and results["_meta"]["n_questions"] == 3
    for row in results["methods"].values():
        assert 0.0 <= row["recall_at_k"] <= 1.0
        assert 0.0 <= row["answer_acc"] <= 1.0
        assert len(row["per_question"]) == 3
        assert all("diagnostics" in q for q in row["per_question"])


def test_run_rejects_unknown_method(resources):
    from benchmark import run

    with pytest.raises(KeyError):
        run(method_names=["definitely_not_a_method"], resources=resources)


def test_resource_flags_prune_registry(resources):
    from benchmark import method_registry

    registry = method_registry(resources)
    assert "graphrag" not in registry and "raptor" not in registry
    assert not any(name.startswith("chunk:") for name in registry)
