import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

FEATURE_FLOW_NAVIGATION_EDGE_TYPES: set[str] = {
    "diagnostic_statement_outcome",
    "diagnostic_conditional_target",
    "diagnostic_disruptive_outcome",
    "derived_control_route_execution_item",
    "derived_control_route_execution_item_terminal",
    "call",
    "return",
}


@dataclass(frozen=True)
class FeatureFlowGraphs:
    cfg: nx.MultiDiGraph
    cfg_nav: nx.DiGraph


def load_cfg(cfg_path: Path, cfg_format: str | None = None) -> nx.MultiDiGraph:
    suffix = cfg_path.suffix.lower()
    graph_format = (
        cfg_format
        if cfg_format is not None
        else (
            "pickle"
            if suffix in {".pkl", ".pickle"}
            else "yaml" if suffix in {".yaml", ".yml"} else None
        )
    )

    if graph_format == "pickle":
        with open(cfg_path, "rb") as f:
            graph = pickle.load(f)
        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError(
                f"Expected NetworkX MultiDiGraph in {cfg_path}, got {type(graph)!r}"
            )
        return graph

    if graph_format == "yaml":
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return nx.node_link_graph(data, edges="edges")

    raise ValueError(f"Unsupported graph format: {graph_format}")


def is_feature_flow_navigation_edge(edge_data: dict[str, Any]) -> bool:
    return edge_data.get("edge_type") in FEATURE_FLOW_NAVIGATION_EDGE_TYPES


def navigation_graph(cfg: nx.MultiDiGraph) -> nx.DiGraph:
    graph = nx.DiGraph()

    for node_id, data in cfg.nodes(data=True):
        graph.add_node(node_id, **dict(data))

    for source, target, key, data in cfg.edges(keys=True, data=True):
        if not is_feature_flow_navigation_edge(data):
            continue

        edge_record = {
            "key": key,
            "source": source,
            "target": target,
            **dict(data),
        }

        if graph.has_edge(source, target):
            graph[source][target].setdefault("source_edges", []).append(edge_record)
            graph[source][target].setdefault("edge_types", set()).add(
                data.get("edge_type")
            )
            continue

        graph.add_edge(
            source,
            target,
            edge_type=data.get("edge_type"),
            edge_types={data.get("edge_type")},
            source_edges=[edge_record],
        )

    return graph


def load_graph(
    cfg_path: Path, cfg_format: str | None = None
) -> FeatureFlowGraphs | None:
    if cfg_path:
        cfg = load_cfg(cfg_path, cfg_format)
        return FeatureFlowGraphs(cfg=cfg, cfg_nav=navigation_graph(cfg))
    else:
        return None
