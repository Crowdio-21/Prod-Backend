"""
Topology manager for DNN inference jobs.

This module provides a minimal, validated graph abstraction that can be
extended later for advanced optimization and dynamic adaptation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class TopologyNode:
    """Represents one logical compute node in a DNN inference graph."""

    node_id: str
    role: str = "intermediate"  # source, intermediate, sink
    worker_id: Optional[str] = None
    model_partition_id: Optional[str] = None
    requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyEdge:
    """Represents a directed feature flow edge between topology nodes."""

    source: str
    target: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TopologyValidationError(ValueError):
    """Raised when a topology graph is invalid."""


class TopologyManager:
    """Manages parsing, validation, and serialization of inference topologies."""

    def __init__(self):
        self._graphs: Dict[str, Dict[str, Any]] = {}

    def register_graph(
        self,
        inference_graph_id: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        parsed_nodes = [TopologyNode(**n) for n in nodes]
        parsed_edges = [TopologyEdge(**e) for e in edges]

        self._validate(parsed_nodes, parsed_edges)

        graph = {
            "inference_graph_id": inference_graph_id,
            "nodes": [self._node_to_dict(n) for n in parsed_nodes],
            "edges": [self._edge_to_dict(e) for e in parsed_edges],
            "metadata": metadata or {},
        }
        self._graphs[inference_graph_id] = graph
        return graph

    def get_graph(self, inference_graph_id: str) -> Optional[Dict[str, Any]]:
        return self._graphs.get(inference_graph_id)

    def list_graphs(self) -> List[str]:
        return list(self._graphs.keys())

    def update_assignments(
        self,
        inference_graph_id: str,
        node_assignments: Dict[str, str],
    ) -> Dict[str, Any]:
        graph = self._graphs.get(inference_graph_id)
        if not graph:
            raise TopologyValidationError(
                f"Unknown inference graph id: {inference_graph_id}"
            )

        for node in graph["nodes"]:
            node_id = node["node_id"]
            if node_id in node_assignments:
                node["worker_id"] = node_assignments[node_id]

        return graph

    def _validate(self, nodes: List[TopologyNode], edges: List[TopologyEdge]) -> None:
        node_ids: Set[str] = {n.node_id for n in nodes}
        if not node_ids:
            raise TopologyValidationError("Topology must contain at least one node")

        if len(node_ids) != len(nodes):
            raise TopologyValidationError("Duplicate node_id values are not allowed")

        for edge in edges:
            if edge.source not in node_ids:
                raise TopologyValidationError(
                    f"Edge source does not exist: {edge.source}"
                )
            if edge.target not in node_ids:
                raise TopologyValidationError(
                    f"Edge target does not exist: {edge.target}"
                )

        # DAG requirement for deterministic execution ordering.
        self._ensure_acyclic(node_ids, edges)

    def _ensure_acyclic(self, node_ids: Set[str], edges: List[TopologyEdge]) -> None:
        adjacency: Dict[str, List[str]] = {n: [] for n in node_ids}
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
            for child in adjacency[node_id]:
                dfs(child)
            active.remove(node_id)
            visited.add(node_id)

        for node_id in node_ids:
            dfs(node_id)

    @staticmethod
    def _node_to_dict(node: TopologyNode) -> Dict[str, Any]:
        return {
            "node_id": node.node_id,
            "role": node.role,
            "worker_id": node.worker_id,
            "model_partition_id": node.model_partition_id,
            "requirements": node.requirements,
        }

    @staticmethod
    def _edge_to_dict(edge: TopologyEdge) -> Dict[str, Any]:
        return {
            "source": edge.source,
            "target": edge.target,
            "metadata": edge.metadata,
        }
