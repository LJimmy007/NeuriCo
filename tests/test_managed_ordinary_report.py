"""Completion-contract tests for managed ordinary research."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import core.pipeline_orchestrator as pipeline  # noqa: E402
from core.pipeline_orchestrator import ResearchPipelineOrchestrator  # noqa: E402


def _ordinary_orchestrator(work_dir: Path) -> ResearchPipelineOrchestrator:
    return ResearchPipelineOrchestrator(
        work_dir=work_dir,
        templates_dir=Path(__file__).resolve().parents[1] / "templates",
        managed_initial_run=True,
    )


def test_managed_ordinary_stage_records_report_as_expected_output(tmp_path, monkeypatch):
    orchestrator = _ordinary_orchestrator(tmp_path)

    def stop_after_stage_start(_stage):
        raise RuntimeError("stop after stage start")

    monkeypatch.setattr(orchestrator, "_create_hitl_runtime", stop_after_stage_start)

    with pytest.raises(RuntimeError, match="stop after stage start"):
        orchestrator._run_experiment_runner_hitl(
            idea={"idea": {}},
            provider="codex",
            timeout=None,
            full_permissions=True,
        )

    stage = orchestrator.state.state["stages"]["experiment_runner"]
    assert stage["expected_outputs"] == ["REPORT.md"]


def test_managed_ordinary_finish_requires_report(tmp_path):
    orchestrator = _ordinary_orchestrator(tmp_path)
    validator = orchestrator._experiment_runner_artifact_validator(scoring_enabled=False)

    assert validator is not None
    missing = validator()
    assert missing["valid"] is False
    assert missing["missing"] == ["REPORT.md"]
    assert missing["issues"] == [
        "Required ordinary research output is missing: REPORT.md"
    ]

    (tmp_path / "REPORT.md").write_text("# Completed research\n", encoding="utf-8")

    present = validator()
    assert present["valid"] is True
    assert present["missing"] == []
    assert present["issues"] == []


def test_resumed_managed_ordinary_approval_revalidates_report(tmp_path, monkeypatch):
    orchestrator = _ordinary_orchestrator(tmp_path)
    pending = {
        "status": "resolved",
        "response": {"final": True, "status": "approved"},
        "workspace_fingerprint": "reviewed",
    }
    monkeypatch.setattr(
        orchestrator,
        "_initial_stage_request",
        lambda _stage: pending,
    )
    monkeypatch.setattr(
        pipeline,
        "_require_reviewed_workspace_unchanged",
        lambda _work_dir, _fingerprint: None,
    )
    validator = orchestrator._experiment_runner_artifact_validator(scoring_enabled=False)

    with pytest.raises(pipeline.HitlValidationError, match="failed artifact validation"):
        orchestrator._resume_initial_worker(
            SimpleNamespace(pipeline_stage="experiment_runner"),
            launch_worker=lambda *args, **kwargs: {},
            worker_prompt_contexts={},
            validator=validator,
        )


def test_unmanaged_non_scoring_runner_keeps_existing_validator_behavior(tmp_path):
    orchestrator = ResearchPipelineOrchestrator(
        work_dir=tmp_path,
        templates_dir=Path(__file__).resolve().parents[1] / "templates",
    )

    assert orchestrator._experiment_runner_artifact_validator(scoring_enabled=False) is None


def test_scored_runner_keeps_existing_artifact_validator(tmp_path, monkeypatch):
    expected = {"valid": False, "issues": ["scored contract"]}
    monkeypatch.setattr(
        pipeline,
        "validate_required_artifact_contract",
        lambda _work_dir: expected,
    )
    orchestrator = ResearchPipelineOrchestrator(
        work_dir=tmp_path,
        templates_dir=Path(__file__).resolve().parents[1] / "templates",
        hitl_autoresearch=True,
    )

    validator = orchestrator._experiment_runner_artifact_validator(scoring_enabled=True)

    assert validator is not None
    assert validator() is expected
