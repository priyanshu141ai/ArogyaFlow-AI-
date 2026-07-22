import subprocess
from pathlib import Path

import pytest

import arogyaflow.demo as demo
from arogyaflow.demo import DemoArtifacts, load_demo_config, prepare_demo


def _test_config() -> demo.DemoConfig:
    config = load_demo_config(Path("config/demo.json"))
    return config.model_copy(
        update={
            "track_experiment": False,
            "wait_output_dir": Path("wait"),
            "arrival_output_dir": Path("arrivals"),
            "no_show_output_dir": Path("no-show"),
            "occupancy_output_dir": Path("occupancy"),
            "monitoring_report_path": Path("reports/monitoring.json"),
        }
    )


def test_prepare_demo_builds_all_artifacts(tmp_path: Path) -> None:
    result = prepare_demo(_test_config(), tmp_path)
    assert result.wait_model.is_file()
    assert result.arrival_model.is_file()
    assert result.no_show_model.is_file()
    assert result.occupancy_model.is_file()
    assert result.monitoring_report.is_file()


def test_launch_demo_runs_infrastructure_prepare_and_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    prepared: list[Path] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    def fake_prepare(config: demo.DemoConfig, project_root: Path) -> DemoArtifacts:
        del config
        prepared.append(project_root)
        artifact = project_root / "artifact"
        return DemoArtifacts(artifact, artifact, artifact, artifact, artifact)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(demo, "prepare_demo", fake_prepare)
    demo.launch_demo(_test_config(), tmp_path)
    assert commands == [
        ["docker", "compose", "up", "-d", "db"],
        ["docker", "compose", "up", "--build"],
    ]
    assert prepared == [tmp_path]
    assert (tmp_path / ".arogyaflow-demo.env").is_file()
