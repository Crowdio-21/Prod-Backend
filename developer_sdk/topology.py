
"""SDK topology helpers for DNN pipeline validation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class TopologyNode:
    node_id: str
    role: str = "intermediate"
    model_partition_id: Optional[str] = None
    requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyEdge:
    source: str
    target: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TopologyValidationError(ValueError):
    pass


def validate_topology(
    stages: List[Dict[str, Any]],
    topology_nodes: List[Dict[str, Any]],
    topology_edges: List[Dict[str, Any]],
) -> None:
    """Validate topology consistency against stage definitions."""
    nodes = [TopologyNode(**n) for n in topology_nodes]
    edges = [TopologyEdge(**e) for e in topology_edges]

    if not nodes:
        raise TopologyValidationError("topology_nodes must not be empty")

    node_ids = {n.node_id for n in nodes}
    if len(node_ids) != len(nodes):
        raise TopologyValidationError("Duplicate node_id values are not allowed")

    stage_names = {s.get("name") for s in stages if s.get("name")}
    partition_ids = {n.model_partition_id for n in nodes if n.model_partition_id}
    missing_partitions = partition_ids - stage_names
    if missing_partitions:
        raise TopologyValidationError(
            f"Topology model_partition_id values have no matching stage name: {sorted(missing_partitions)}"
        )

    for edge in edges:
        if edge.source not in node_ids:
            raise TopologyValidationError(f"Unknown edge source: {edge.source}")
        if edge.target not in node_ids:
            raise TopologyValidationError(f"Unknown edge target: {edge.target}")

    _ensure_dag(node_ids, edges)


def _ensure_dag(node_ids: Set[str], edges: List[TopologyEdge]) -> None:
    adjacency: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}
    for edge in edges:
        adjacency[edge.source].append(edge.target)

    visited: Set[str] = set()
    active: Set[str] = set()

    def dfs(node_id: str) -> None:
        if node_id in active:
            raise TopologyValidationError("Topology graph contains a cycle")
        if node_id in visited:
            return
        active.add(node_id)
        for nxt in adjacency[node_id]:
            dfs(nxt)
        active.remove(node_id)
        visited.add(node_id)

    for node_id in node_ids:
        dfs(node_id)
