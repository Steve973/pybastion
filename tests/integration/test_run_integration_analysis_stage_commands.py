from __future__ import annotations

from pathlib import Path

import pytest

from pybastion_integration import run_integration_analysis as driver

pytestmark = pytest.mark.integration


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "analysis_output_root": tmp_path / "dist" / "pybastion",
        "inventories_root": tmp_path / "dist" / "pybastion" / "inventory",
        "output_dir": tmp_path / "dist" / "pybastion" / "integration-output",
        "stage1_output": tmp_path
        / "dist"
        / "pybastion"
        / "integration-output"
        / "stage1-ei-cfg.pkl",
        "stage2_output": tmp_path
        / "dist"
        / "pybastion"
        / "integration-output"
        / "stage2-feature-flow-cases.yaml",
        "stage3_output": tmp_path
        / "dist"
        / "pybastion"
        / "integration-output"
        / "stage3-integration-test-specs.yaml",
        "stage2_marker_inventory_output": tmp_path
        / "dist"
        / "pybastion"
        / "integration-output"
        / "stage2-feature-marker-inventory.yaml",
        "stage2_branch_points_output": tmp_path
        / "dist"
        / "pybastion"
        / "integration-output"
        / "stage2-feature-branch-points.yaml",
        "stage2_converge_points_output": tmp_path
        / "dist"
        / "pybastion"
        / "integration-output"
        / "stage2-feature-converge-points.yaml",
        "spec_split_output_dir": tmp_path
        / "dist"
        / "pybastion"
        / "integration-output"
        / "specs",
        "graph_check_report": tmp_path
        / "dist"
        / "pybastion"
        / "integration-output"
        / "inventory-graph-report.yaml",
        "graph_checker_script": tmp_path / "check_inventory_graph.py",
    }


def test_emit_all_output_is_passed_to_stage2_only(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    stage2_cmd = driver.build_stage_cmd(
        2,
        target_root=tmp_path,
        paths=paths,
        graph_format="pickle",
        verbose=False,
        emit_all_output=True,
    )

    stage3_cmd = driver.build_stage_cmd(
        3,
        target_root=tmp_path,
        paths=paths,
        graph_format="pickle",
        verbose=False,
        emit_all_output=True,
    )

    assert "--emit-all-output" in stage2_cmd
    assert "--emit-all-output" not in stage3_cmd


def test_stage2_omits_emit_all_output_by_default(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    stage2_cmd = driver.build_stage_cmd(
        2,
        target_root=tmp_path,
        paths=paths,
        graph_format="pickle",
        verbose=False,
        emit_all_output=False,
    )

    assert "--emit-all-output" not in stage2_cmd
