#!/usr/bin/env python3
"""
Feature Flow Tracer

Loads the CFG and traces execution paths from FeatureStart to FeatureEnd /
FeatureEndConditional decorators, then outputs structured YAML.

This version assumes feature markers live on specific EI nodes, not callable
entry EIs.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from pybastion_integration import config


# =============================================================================
# Helper Functions
# =============================================================================


def choose_representative_path_id(group_paths: list[dict[str, Any]]) -> str:
    """Pick the shortest path as representative, breaking ties by path_id."""
    best = min(
        group_paths,
        key=lambda record: (len(record["path"]), record["path_id"]),
    )
    return best["path_id"]



def summarize_divergence_points(
    cfg: nx.DiGraph,
    group_paths: list[dict[str, Any]],
    max_points: int = 5,
) -> list[dict[str, Any]]:
    """
    Produce a compact list of meaningful divergence anchors for a group.

    Prefer real EI nodes with source backed line numbers over external or
    collapsed nodes.
    """
    if len(group_paths) < 2:
        return []

    raw_paths = [record["path"] for record in group_paths]
    min_len = min(len(path) for path in raw_paths)
    summaries: list[dict[str, Any]] = []

    def node_rank(node_id: str) -> tuple[int, int]:
        node_text = str(node_id)

        if node_text.startswith("external::"):
            return (3, 1)

        if node_text.startswith("collapsed::"):
            return (2, 1)

        node_data = cfg.nodes.get(node_id, {})
        line = node_data.get("line")
        return (0 if line is not None else 1, 0)

    def node_summary(node_id: str) -> tuple[int | None, str]:
        node_text = str(node_id)

        if node_text.startswith("external::"):
            return (None, node_text)

        if node_text.startswith("collapsed::"):
            node_data = cfg.nodes.get(node_id, {})
            description = node_data.get("description", "collapsed call")
            return (None, description)

        node_data = cfg.nodes.get(node_id, {})
        line = node_data.get("line")
        description = (
            node_data.get("description")
            or node_data.get("condition")
            or node_data.get("stmt_type")
            or node_text
        )
        return (line, description)

    for position in range(min_len):
        node_ids_at_position = [path[position] for path in raw_paths]
        unique_node_ids = list(dict.fromkeys(node_ids_at_position))

        if len(unique_node_ids) <= 1:
            continue

        preferred_node_id = sorted(unique_node_ids, key=node_rank)[0]
        line, description = node_summary(preferred_node_id)

        summaries.append(
            {
                "ei_id": preferred_node_id,
                "line": line,
                "description": description,
            }
        )

        if len(summaries) >= max_points:
            break

    return summaries


# =============================================================================
# CFG Loading
# =============================================================================


def load_cfg(cfg_path: Path) -> nx.DiGraph:
    """Load the CFG from pickle file."""
    with open(cfg_path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Decorator Analysis
# =============================================================================


def find_nodes_with_decorator(cfg: nx.DiGraph, decorator_name: str) -> list[tuple[str, dict]]:
    """
    Find all EI nodes that have a specific decorator.

    Feature markers like FeatureStart / FeatureEnd / FeatureEndConditional
    live on the specific EI node, not on the callable entry EI.
    """
    matching_nodes: list[tuple[str, dict]] = []

    for node_id, node_data in cfg.nodes(data=True):
        if node_data.get("category") != "execution_instance":
            continue

        decorators = node_data.get("decorators", []) or []
        for dec in decorators:
            if isinstance(dec, dict) and dec.get("name") == decorator_name:
                matching_nodes.append((node_id, node_data))
                break

    return matching_nodes



def get_decorator_kwargs(node_data: dict, decorator_name: str) -> dict[str, Any]:
    """Extract kwargs from a specific decorator on a node."""
    decorators = node_data.get("decorators", []) or []
    for dec in decorators:
        if isinstance(dec, dict) and dec.get("name") == decorator_name:
            kwargs = dec.get("kwargs", {})
            return kwargs if isinstance(kwargs, dict) else {}
    return {}


# =============================================================================
# Path Finding
# =============================================================================


def find_feature_flows(
    cfg: nx.DiGraph,
    max_paths_per_flow: int = 20,
    path_strategy: str = "shortest",
) -> list[dict[str, Any]]:
    """Find all feature flows from FeatureStart to FeatureEnd / FeatureEndConditional."""
    start_nodes = find_nodes_with_decorator(cfg, "FeatureStart")
    end_nodes = (
        find_nodes_with_decorator(cfg, "FeatureEnd")
        + find_nodes_with_decorator(cfg, "FeatureEndConditional")
    )

    if not start_nodes:
        print("No FeatureStart decorators found in CFG")
        return []

    if not end_nodes:
        print("No FeatureEnd / FeatureEndConditional decorators found in CFG")
        return []

    print(f"Found {len(start_nodes)} FeatureStart node(s)")
    print(f"Found {len(end_nodes)} FeatureEnd node(s)")

    flows: list[dict[str, Any]] = []

    for start_id, start_data in start_nodes:
        start_kwargs = get_decorator_kwargs(start_data, "FeatureStart")
        feature_name = start_kwargs.get("name", "unnamed")

        for end_id, end_data in end_nodes:
            end_kwargs = get_decorator_kwargs(end_data, "FeatureEnd")
            if not end_kwargs:
                end_kwargs = get_decorator_kwargs(end_data, "FeatureEndConditional")

            end_feature_name = end_kwargs.get("name", "unnamed")

            if (
                feature_name != end_feature_name
                and feature_name != "unnamed"
                and end_feature_name != "unnamed"
            ):
                continue

            print(f"\nFeature tracing: {feature_name} ({start_id} -> {end_id})")

            try:
                if path_strategy == "shortest":
                    print("  Finding shortest path...")
                    try:
                        shortest = nx.shortest_path(cfg, start_id, end_id)
                        print(f"  Shortest path length: {len(shortest)}")
                        shortest_len = len(shortest)
                    except nx.NetworkXNoPath:
                        print("  No shortest path, trying bounded simple paths...")
                        shortest_len = 50

                    max_len = int(shortest_len * 1.5)

                    collected_paths: list[list[str]] = []
                    for path in nx.all_simple_paths(cfg, start_id, end_id, cutoff=max_len):
                        collected_paths.append(path)
                        if len(collected_paths) >= max_paths_per_flow * 2:
                            break

                    collected_paths.sort(key=len)
                    selected_paths = collected_paths[:max_paths_per_flow]

                elif path_strategy == "diverse":
                    shortest = nx.shortest_path(cfg, start_id, end_id)
                    shortest_len = len(shortest)
                    max_len = int(shortest_len * 2.0)

                    collected_paths: list[list[str]] = []
                    for path in nx.all_simple_paths(cfg, start_id, end_id, cutoff=max_len):
                        collected_paths.append(path)
                        if len(collected_paths) >= max_paths_per_flow * 3:
                            break

                    collected_paths.sort(key=len)
                    selected_paths = []
                    step = max(1, len(collected_paths) // max_paths_per_flow)
                    for i in range(0, len(collected_paths), step):
                        selected_paths.append(collected_paths[i])
                        if len(selected_paths) >= max_paths_per_flow:
                            break

                else:  # all
                    selected_paths = []
                    for path in nx.all_simple_paths(cfg, start_id, end_id, cutoff=100):
                        selected_paths.append(path)
                        if len(selected_paths) >= max_paths_per_flow:
                            break

            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                print(f"  Path finding failed: {type(e).__name__}: {e}")
                continue

            if selected_paths:
                print(f"  Found {len(selected_paths)} path(s)")
                flows.append(
                    {
                        "feature_name": feature_name,
                        "start_node": start_id,
                        "end_node": end_id,
                        "start_unit": start_data.get("unit"),
                        "start_callable": start_data.get("callable_name"),
                        "end_unit": end_data.get("unit"),
                        "end_callable": end_data.get("callable_name"),
                        "paths": selected_paths,
                        "path_count": len(selected_paths),
                    }
                )

    return flows


# =============================================================================
# Path Similarity Analysis
# =============================================================================


def calculate_path_similarity(path1: list[str], path2: list[str]) -> float:
    set1 = set(path1)
    set2 = set(path2)

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0



def group_similar_paths(
    cfg: nx.DiGraph,
    paths: list[list[str]],
    similarity_threshold: float = 0.8,
) -> list[dict[str, Any]]:
    del cfg  # similarity grouping is path based only

    if not paths:
        return []

    groups: list[list[int]] = [[i] for i in range(len(paths))]

    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                path_i = paths[groups[i][0]]
                path_j = paths[groups[j][0]]

                if calculate_path_similarity(path_i, path_j) >= similarity_threshold:
                    groups[i].extend(groups[j])
                    groups.pop(j)
                    merged = True
                    break
            if merged:
                break

    result_groups: list[dict[str, Any]] = []
    for group_idx, path_indices in enumerate(groups, start=1):
        result_groups.append(
            {
                "group_id": group_idx,
                "path_indices": path_indices,
            }
        )

    return result_groups


# =============================================================================
# YAML Output
# =============================================================================


def extract_path_metadata(cfg: nx.DiGraph, path: list[str]) -> dict[str, Any]:
    """Extract call chain and branch points from a path."""
    call_chain: list[str] = []
    branch_points: list[str] = []

    for node_id in path:
        node_data = cfg.nodes[node_id]
        node_category = node_data.get("category")

        if node_category in {"external_node", "external_call"}:
            external_type = node_data.get("type", "unknown")
            operation_target = node_data.get("operation_target", "unknown")
            call_chain.append(f"{node_id}: EXTERNAL_{external_type.upper()}({operation_target})")
            continue

        if str(node_id).startswith("collapsed::"):
            unit = node_data.get("unit", "unknown")
            callable_name = node_data.get("callable_name", "unknown")
            description = node_data.get("description", "collapsed call")
            call_chain.append(f"{node_id}: {unit}::{callable_name}::{description}")
            continue

        unit = node_data.get("unit", "unknown")
        callable_name = node_data.get("callable_name", "unknown")
        description = node_data.get("description", "")
        condition = node_data.get("condition", "")
        stmt_type = node_data.get("stmt_type", "")
        label = description or condition or stmt_type or "ei"

        call_chain.append(f"{node_id}: {unit}::{callable_name}::{label}")

        constraint = node_data.get("constraint") or {}
        constraint_type = constraint.get("constraint_type")
        if constraint_type in {"condition", "iteration"}:
            line = node_data.get("line") or constraint.get("line")
            desc = description or condition or stmt_type
            if desc:
                branch_points.append(f"L{line}: {desc}" if line else desc)

    return {
        "call_chain": call_chain,
        "branch_points": list(dict.fromkeys(branch_points)),
    }



def build_output_dict(cfg: nx.DiGraph, flows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {
        "flows": [],
        "groupings": [],
    }

    for flow in flows:
        feature_name = flow["feature_name"]
        paths = flow["paths"]

        flow_entry = {
            "feature_name": feature_name,
            "start_node": flow["start_node"],
            "end_node": flow["end_node"],
            "start_location": f"{flow['start_unit']}::{flow['start_callable']}",
            "end_location": f"{flow['end_unit']}::{flow['end_callable']}",
            "total_paths": flow["path_count"],
            "paths": [],
        }

        path_records: list[dict[str, Any]] = []

        for i, path in enumerate(paths, start=1):
            path_id = f"{feature_name}_path_{i:03d}"
            path_metadata = extract_path_metadata(cfg, path)

            flow_entry["paths"].append(
                {
                    "path_id": path_id,
                    "length": len(path),
                    "call_chain": path_metadata["call_chain"],
                    "branch_points": path_metadata["branch_points"],
                }
            )

            path_records.append(
                {
                    "path_id": path_id,
                    "path": path,
                }
            )

        output["flows"].append(flow_entry)

        groups = group_similar_paths(cfg, [record["path"] for record in path_records])

        grouping_entry = {
            "feature_name": feature_name,
            "groups": [],
        }

        for group in groups:
            group_path_records = [path_records[idx] for idx in group["path_indices"]]

            representative_path_id = choose_representative_path_id(
                [
                    {
                        "path_id": record["path_id"],
                        "path": record["path"],
                    }
                    for record in group_path_records
                ]
            )

            divergence_points = summarize_divergence_points(
                cfg,
                [
                    {
                        "path_id": record["path_id"],
                        "path": record["path"],
                    }
                    for record in group_path_records
                ],
                max_points=5,
            )

            grouping_entry["groups"].append(
                {
                    "group_id": group["group_id"],
                    "path_ids": [record["path_id"] for record in group_path_records],
                    "path_count": len(group_path_records),
                    "representative_path_id": representative_path_id,
                    "divergence_points": divergence_points,
                }
            )

        if grouping_entry["groups"]:
            output["groupings"].append(grouping_entry)

    return output


# =============================================================================
# Main
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--cfg",
        type=Path,
        default=None,
        help="Path to CFG pickle file (default: from config)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output YAML file (default: feature-flows.yaml in integration output dir)",
    )
    ap.add_argument(
        "--max-paths",
        type=int,
        default=20,
        help="Maximum paths per feature flow (default: 20)",
    )
    ap.add_argument(
        "--path-strategy",
        choices=["shortest", "diverse", "all"],
        default="shortest",
        help="Path selection strategy: shortest (default), diverse, or all",
    )
    ap.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for grouping (default: 0.8)",
    )
    ap.add_argument(
        "--target-root",
        type=Path,
        help="Target project root (sets config defaults)",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = ap.parse_args(argv)

    if args.target_root:
        config.set_target_root(args.target_root)

    if args.cfg:
        cfg_path = args.cfg
    else:
        cfg_path = config.get_integration_output_dir() / "stage1-ei-cfg.pkl"

    if not cfg_path.exists():
        print(f"ERROR: CFG file not found: {cfg_path}", file=sys.stderr)
        return 1

    if args.output:
        output_path = args.output
    else:
        output_path = config.get_integration_output_dir() / "feature-flows.yaml"

    if args.verbose:
        print(f"Loading CFG from {cfg_path}...")

    cfg = load_cfg(cfg_path)

    if args.verbose:
        print(f"CFG loaded: {cfg.number_of_nodes()} nodes, {cfg.number_of_edges()} edges")

    if args.verbose:
        print("\nSearching for feature flows...")

    flows = find_feature_flows(
        cfg,
        max_paths_per_flow=args.max_paths,
        path_strategy=args.path_strategy,
    )

    if not flows:
        print("\nNo complete feature flows found from FeatureStart to FeatureEnd")
        return 0

    if args.verbose:
        print(f"Found {len(flows)} feature flow(s)")
        for flow in flows:
            print(f"  {flow['feature_name']}: {flow['path_count']} path(s)")

    if args.verbose:
        print("\nAnalyzing path similarities and building output...")

    output_dict = build_output_dict(cfg, flows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            output_dict,
            f,
            default_flow_style=False,
            sort_keys=False,
            width=float("inf"),
            indent=config.get_yaml_indent(),
        )

    print(f"\n✓ Feature flow analysis complete → {output_path}")
    print(f"  Features: {len(flows)}")
    print(f"  Total paths: {sum(f['path_count'] for f in flows)}")
    print(f"  Total groups: {sum(len(g['groups']) for g in output_dict['groupings'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
