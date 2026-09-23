"""Exercise three real, local NeuriCo CLI submissions on a Linux runner.

Usage: python cli_smoke.py --repo /path/to/checkout --output-dir /path/to/artifacts
The caller should use the checkout's locked uv environment as its interpreter.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import yaml


def _run(repo: Path, output: Path, children: list[tuple[str, subprocess.Popen[str], int]]) -> None:
    if not (repo / "src/cli/submit.py").is_file():
        raise AssertionError(f"Not a NeuriCo checkout: {repo}")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise AssertionError(f"Smoke output directory must start empty: {output}")

    source_dir = output / "source"
    resources_dir = output / "resources"
    ideas_dir = output / "ideas"
    source_dir.mkdir()
    resources_dir.mkdir()

    env = os.environ.copy()
    env["NEURICO_IDEAS"] = str(ideas_dir)
    env["PYTHONIOENCODING"] = "utf-8"
    # The real CLI imports GitHub support, but --no-github must prevent calls.
    for key in ("GITHUB_TOKEN", "GH_TOKEN", "OPENAI_API_KEY", "S2_API_KEY"):
        env.pop(key, None)

    labels = ("claude", "gemini", "codex")
    expected: dict[str, dict[str, str]] = {}
    for label in labels:
        resource = resources_dir / f"{label}.csv"
        resource.write_text(f"source,value\n{label},1\n", encoding="utf-8")
        hypothesis = f"The independent {label} submission keeps its own evaluation data."
        idea = {
            "idea": {
                "title": "Identical title from real CLI smoke",
                "domain": "machine_learning",
                "hypothesis": hypothesis,
                "metadata": {"smoke_marker": label},
                "local_resources": {
                    "datasets": [{"path": str(resource), "usage": "Evaluation data"}]
                },
            }
        }
        source = source_dir / f"{label}.yaml"
        source.write_text(yaml.safe_dump(idea, sort_keys=False), encoding="utf-8")
        expected[label] = {"hypothesis": hypothesis, "resource": str(resource)}

    # Launch all three processes together. Their recorded created_at seconds
    # show whether this smoke also covered the same-second collision window;
    # deterministic frozen-clock and multiprocessing coverage lives in pytest.
    wait = 1.04 - (time.time() % 1)
    if wait > 0:
        time.sleep(wait)
    for label in labels:
        source = source_dir / f"{label}.yaml"
        started_ns = time.time_ns()
        child = subprocess.Popen(
            [
                sys.executable,
                "-X",
                "utf8",
                str(repo / "src/cli/submit.py"),
                str(source),
                "--no-github",
                "--provider",
                label,
            ],
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        children.append((label, child, started_ns))

    results: dict[str, dict[str, object]] = {}
    for label, child, started_ns in children:
        try:
            stdout, _ = child.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()
            raise AssertionError(f"CLI submission timed out: {label}") from None
        (output / f"cli-{label}.log").write_text(stdout, encoding="utf-8")
        if child.returncode != 0:
            raise AssertionError(f"CLI submission {label} exited {child.returncode}:\n{stdout}")
        if "Creating GitHub repository" in stdout:
            raise AssertionError(f"--no-github was ignored for {label}")
        match = re.search(r"^Idea ID:\s*(\S+)\s*$", stdout, flags=re.MULTILINE)
        if match is None:
            raise AssertionError(f"CLI did not print an idea ID for {label}:\n{stdout}")
        results[label] = {"idea_id": match.group(1), "started_ns": started_ns}

    idea_ids = [str(results[label]["idea_id"]) for label in labels]
    if len(set(idea_ids)) != len(labels):
        raise AssertionError(f"CLI reused an ID: {results}")
    submitted_files = list((ideas_dir / "submitted").glob("*.yaml"))
    if len(submitted_files) != len(labels):
        raise AssertionError(f"Expected 3 saved ideas, found {submitted_files}")

    for label in labels:
        idea_id = str(results[label]["idea_id"])
        saved = yaml.safe_load((ideas_dir / "submitted" / f"{idea_id}.yaml").read_text())
        inner = saved["idea"]
        metadata = inner["metadata"]
        expected_resource = expected[label]["resource"]
        assert inner["title"] == "Identical title from real CLI smoke"
        assert inner["hypothesis"] == expected[label]["hypothesis"]
        assert inner["local_resources"]["datasets"][0]["path"] == expected_resource
        assert Path(expected_resource).is_file()
        assert metadata["smoke_marker"] == label
        assert metadata["idea_id"] == idea_id
        assert metadata["status"] == "submitted"
        mount = ideas_dir / "mounts" / f"{idea_id}.txt"
        assert mount.read_text(encoding="utf-8").splitlines() == [expected_resource]
        results[label]["created_at"] = metadata["created_at"]

    seconds = {
        datetime.fromisoformat(str(results[label]["created_at"])).strftime("%Y-%m-%dT%H:%M:%S")
        for label in labels
    }
    summary = {
        "submissions": results,
        "distinct_ids": True,
        "preserved_content_and_mounts": True,
        "same_created_at_second": len(seconds) == 1,
        "source_and_saved_ideas": str(output),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run(repo: Path, output: Path) -> None:
    children: list[tuple[str, subprocess.Popen[str], int]] = []
    try:
        _run(repo, output, children)
    finally:
        # An early failure must not leave other concurrent CLI jobs running.
        for _, child, _ in children:
            if child.poll() is None:
                child.kill()
        for _, child, _ in children:
            try:
                child.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.output_dir:
        run(repo, args.output_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="neurico-cli-smoke-") as temporary:
            run(repo, Path(temporary))


if __name__ == "__main__":
    main()
