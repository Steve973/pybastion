"""
Core data structures for integration flow generation.

This module provides dataclasses for all 6 stages of the integration
flow analysis pipeline. Each stage's structures build on the previous,
creating a clear data flow from raw integration points to test windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# =============================================================================
# Stage 1: Integration Point Collection
# =============================================================================

@dataclass
class TargetRef:
    """
    Reference to a resolved integration target.

    Targets start as 'unresolved' (just a string) and are resolved
    in Stage 3 to specific callables with IDs.
    """
    status: Literal['ambiguous', 'boundary', 'extlib', 'interunit', 'resolved', 'stdlib', 'unresolved']
    raw: str  # Original target string from ledger

    # Resolved fields (populated in Stage 3)
    unit_name: str | None = None
    unit_id: str | None = None
    callable_id: str | None = None
    callable_name: str | None = None
    name: str | None = None

    # For ambiguous resolution
    matches: list[str] = field(default_factory=list)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result: dict[str, Any] = {
            'status': self.status,
            'raw': self.raw
        }
        if self.unit_name:
            result['unit_name'] = self.unit_name
        if self.unit_id:
            result['unit_id'] = self.unit_id
        if self.callable_id:
            result['callable_id'] = self.callable_id
        if self.callable_name:
            result['callable_name'] = self.callable_name
        if self.name:
            result['name'] = self.name
        if self.matches:
            result['matches'] = self.matches
        if self.note:
            result['note'] = self.note
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetRef:
        return cls(
            status=data['status'],
            raw=data['raw'],
            unit_name=data.get('unit_name'),
            unit_id=data.get('unit_id'),
            callable_id=data.get('callable_id'),
            callable_name=data.get('callable_name'),
            name=data.get('name'),
            matches=data.get('matches', []),
            note=data.get('note')
        )


@dataclass
class BoundarySummary:
    """
    Summary of boundary integration details.

    Boundaries represent crossings to external systems (filesystem,
    database, network, etc.) and are terminal nodes in flow enumeration.
    """
    kind: str  # filesystem | database | network | subprocess | etc.
    protocol: str | None = None
    system: str | None = None
    endpoint: str | None = None
    operation: str | None = None
    resource: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result: dict[str, Any] = {'kind': self.kind}
        if self.protocol:
            result['protocol'] = self.protocol
        if self.system:
            result['system'] = self.system
        if self.endpoint:
            result['endpoint'] = self.endpoint
        if self.operation:
            result['operation'] = self.operation
        if self.resource:
            result['resource'] = self.resource
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundarySummary:
        return cls(
            kind=data['kind'],
            protocol=data.get('protocol'),
            system=data.get('system'),
            endpoint=data.get('endpoint'),
            operation=data.get('operation'),
            resource=data.get('resource')
        )


@dataclass
class IntegrationPoint:
    """
    A single integration point (seam) extracted from a unit ledger.

    This represents one integration fact (either interunit, extlib, or boundary)
    with its full context preserved. This is the fundamental node in the
    integration graph.
    """
    id: str  # Integration ID (e.g., IC000F001E0004)
    integration_type: str
    source_unit: str
    source_callable_id: str
    source_callable_name: str
    target_raw: str
    target_resolved: TargetRef | None
    kind: str  # call | construct | import | dispatch | io | other
    execution_paths: list[list[str]]  # List of EI ID sequences
    condition: str | None = None
    boundary: BoundarySummary | None = None
    signature: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result: dict[str, Any] = {
            'id': self.id,
            'integration_type': self.integration_type,
            'source_unit': self.source_unit,
            'source_callable_id': self.source_callable_id,
            'source_callable_name': self.source_callable_name,
            'target': self.target_raw,
            'kind': self.kind,
            'execution_paths': self.execution_paths,
        }

        if self.target_resolved:
            target_dict: dict[str, Any] = self.target_resolved.to_dict()
            result['target_resolved'] = target_dict

        if self.condition:
            result['condition'] = self.condition

        if self.boundary:
            boundary_dict: dict[str, Any] = self.boundary.to_dict()
            result['boundary'] = boundary_dict

        if self.signature:
            result['signature'] = self.signature

        if self.notes:
            result['notes'] = self.notes

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntegrationPoint:
        """Create IntegrationPoint from dictionary (e.g., from YAML)."""
        # Parse TargetRef
        target_resolved = None
        if 'target_resolved' in data:
            tr = data['target_resolved']
            target_resolved = TargetRef(
                status=tr.get('status', 'unresolved'),
                raw=tr.get('raw', data.get('target', '')),
                unit_name=tr.get('unit_name'),
                unit_id=tr.get('unit_id'),
                callable_id=tr.get('callable_id'),
                callable_name=tr.get('callable_name'),
                name=tr.get('name'),
                matches=tr.get('matches', []),
                note=tr.get('note')
            )

        # Parse BoundarySummary
        boundary = None
        if 'boundary' in data:
            b = data['boundary']
            boundary = BoundarySummary(
                kind=b['kind'],
                protocol=b.get('protocol'),
                system=b.get('system'),
                endpoint=b.get('endpoint'),
                operation=b.get('operation'),
                resource=b.get('resource')
            )

        return cls(
            id=data['id'],
            integration_type=data.get('integration_type', 'unknown'),
            source_unit=data.get('source_unit', 'unknown'),
            source_callable_id=data.get('source_callable_id', 'unknown'),
            source_callable_name=data.get('source_callable_name', 'unknown'),
            target_raw=data.get('target', ''),
            target_resolved=target_resolved,
            kind=data.get('kind', 'call'),
            execution_paths=data.get('execution_paths', []),
            condition=data.get('condition'),
            boundary=boundary,
            signature=data.get('signature'),
            notes=data.get('notes')
        )


@dataclass
class IntegrationPointCollection:
    """
    Output from Stage 1: Collection of all integration points.

    This is the top-level output structure for Stage 1, containing
    all integration points extracted from unit ledgers.
    """
    points: list[IntegrationPoint]

    # Optional metadata
    ledger_count: int | None = None
    ledgers_root: str | None = None
    explicit_ledgers: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result: dict[str, Any] = {
            'stage': 'integration-points-collection',
            'integration_points': [p.to_dict() for p in self.points]
        }

        # Add metadata section if any metadata is present
        metadata: dict[str, Any] = {}
        if self.ledger_count is not None:
            metadata['ledger_count'] = self.ledger_count
        metadata['integration_point_count'] = len(self.points)
        metadata['interunit_count'] = sum(1 for p in self.points if p.boundary is None)
        metadata['boundary_count'] = sum(1 for p in self.points if p.boundary is not None)
        if self.ledgers_root is not None:
            metadata['ledgers_root'] = self.ledgers_root
        if self.explicit_ledgers is not None:
            metadata['explicit_ledgers'] = self.explicit_ledgers
        if metadata:
            result['metadata'] = metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntegrationPointCollection:
        ledger_count = None
        ledgers_root = None
        explicit_ledgers = None
        if 'metadata' in data:
            ledger_count = data.get('metadata', {}).get('ledger_count')
            ledgers_root = data.get('metadata', {}).get('ledgers_root')
            explicit_ledgers = data.get('metadata', {}).get('explicit_ledgers')
        return cls(
            points=data['integration_points'],
            ledger_count=ledger_count,
            ledgers_root=ledgers_root,
            explicit_ledgers=explicit_ledgers
        )


# =============================================================================
# Stage 2: Classification
# =============================================================================

@dataclass
class IntegrationPointClassification:
    """
    Classification of integration points by their role in flows.

    - Entry points: Callables with no incoming interunit calls in scope
    - Intermediate: Has both incoming and outgoing interunit calls
    - Terminal nodes: Boundaries OR no outgoing interunit calls
    """
    entry_points: list[str] = field(default_factory=list)  # Integration IDs
    intermediate_seams: list[str] = field(default_factory=list)
    terminal_nodes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            'stage': 'integration-point-classification',
            'entry_points': self.entry_points,
            'intermediate_seams': self.intermediate_seams,
            'terminal_nodes': self.terminal_nodes,
            'metadata': {
                'entry_point_count': len(self.entry_points),
                'intermediate_count': len(self.intermediate_seams),
                'terminal_node_count': len(self.terminal_nodes),
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntegrationPointClassification:
        """Create Classification from dictionary."""
        return cls(
            entry_points=data.get('entry_points', []),
            intermediate_seams=data.get('intermediate_seams', []),
            terminal_nodes=data.get('terminal_nodes', [])
        )


# =============================================================================
# Stage 3: Integration Graph
# =============================================================================

@dataclass
class CallableIndexEntry:
    unit: str
    callable_id: str
    qualified_name: str
    fully_qualified: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            'unit': self.unit,
            'callable_id': self.callable_id,
            'qualified_name': self.qualified_name,
            'fully_qualified': self.fully_qualified,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CallableIndexEntry:
        """Create CallableIndexEntry from a dictionary."""
        return cls(
            unit=data['unit'],
            callable_id=data['callable_id'],
            qualified_name=data['qualified_name'],
            fully_qualified=data['fully_qualified']
        )


@dataclass
class TargetResolution:
    status: str
    unit_id: str | None = None
    unit_name: str | None = None
    callable_id: str | None = None
    name: str | None = None
    qualified_name: str | None = None
    callable_name: str | None = None
    matches: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            'status': self.status,
            'unit_id': self.unit_id,
            'unit_name': self.unit_name,
            'callable_id': self.callable_id,
            'name': self.name,
            'qualified_name': self.qualified_name,
            'callable_name': self.callable_name,
            'matches': self.matches,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetResolution:
        """Create TargetResolution from a dictionary."""
        return cls(
            status=data['status'],
            unit_id=data.get('unit_id'),
            unit_name=data.get('unit_name'),
            callable_id=data.get('callable_id'),
            name=data.get('name'),
            qualified_name=data.get('qualified_name'),
            callable_name=data.get('callable_name'),
            matches=data.get('matches'),
        )


@dataclass
class GraphNode:
    """
    A node in the integration graph (integration point with graph context).

    Extends IntegrationPoint with additional fields needed for graph
    traversal and flow enumeration (classification, exclusion status, etc.).
    """
    # Core integration point data
    id: str
    integration_type: str
    source_unit: str
    source_callable_id: str
    source_callable_name: str
    target: str  # Raw target string
    target_resolved: TargetResolution  # Resolution details
    kind: str
    execution_paths: list[list[str]]

    # Optional integration point fields
    condition: str | None = None
    boundary: BoundarySummary | None = None
    signature: str | None = None
    notes: str | None = None

    # Graph-specific fields
    exclude_from_flows: bool = False  # Set for decorated operations
    fixture_callable_id: str | None = None  # If excluded, what's the fixture?

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result: dict[str, Any] = {
            'id': self.id,
            'integration_type': self.integration_type,
            'source_unit': self.source_unit,
            'source_callable_id': self.source_callable_id,
            'source_callable_name': self.source_callable_name,
            'target': self.target,
            'target_resolved': self.target_resolved,
            'kind': self.kind,
            'execution_paths': self.execution_paths,
        }

        if self.condition:
            result['condition'] = self.condition
        if self.boundary:
            result['boundary'] = self.boundary
        if self.signature:
            result['signature'] = self.signature
        if self.notes:
            result['notes'] = self.notes
        if self.exclude_from_flows:
            result['exclude_from_flows'] = True
        if self.fixture_callable_id:
            result['fixture_callable_id'] = self.fixture_callable_id

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphNode:
        """Create GraphNode from dictionary."""
        return cls(
            id=data['id'],
            integration_type=data.get('integration_type', 'unknown'),
            source_unit=data.get('source_unit', 'unknown'),
            source_callable_id=data.get('source_callable_id', 'unknown'),
            source_callable_name=data.get('source_callable_name', 'unknown'),
            target=data.get('target', ''),
            target_resolved=data.get('target_resolved', {}),
            kind=data.get('kind', 'call'),
            execution_paths=data.get('execution_paths', []),
            condition=data.get('condition'),
            boundary=data.get('boundary'),
            signature=data.get('signature'),
            notes=data.get('notes'),
            exclude_from_flows=data.get('exclude_from_flows', False),
            fixture_callable_id=data.get('fixture_callable_id')
        )


@dataclass
class IntegrationEdge:
    """
    Edge in the integration graph.

    An edge from I1 to I2 exists when I1's target resolves to the
    callable that contains I2.
    """
    from_id: str  # Integration point ID (source)
    to_id: str  # Integration point ID (target)
    reason: str | None = None  # Why this edge exists (for debugging)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result: dict[str, Any] = {
            'from': self.from_id,
            'to': self.to_id,
        }
        if self.reason:
            result['reason'] = self.reason
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntegrationEdge:
        """Create IntegrationEdge from dictionary."""
        return cls(
            from_id=data['from'],
            to_id=data['to'],
            reason=data.get('reason'),
        )


@dataclass
class IntegrationGraph:
    """
    Complete integration graph with nodes and edges.

    This is the output of Stage 3 - a directed graph where:
    - Nodes are integration points (with resolution and classification)
    - Edges connect I1 → I2 when I1 calls the callable containing I2
    """
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[IntegrationEdge] = field(default_factory=list)
    classification: IntegrationPointClassification | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result: dict[str, Any] = {
            'stage': 'integration-graph',
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges],
            'metadata': {
                'node_count': len(self.nodes),
                'edge_count': len(self.edges),
            }
        }

        if self.classification:
            result['classification'] = self.classification.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntegrationGraph:
        """Create IntegrationGraph from dictionary."""
        return cls(
            nodes=data['nodes'],
            edges=data['edges'],
            classification=data['classification'],
        )


# =============================================================================
# Stage 3B: Decorator Candidate Analysis
# =============================================================================

@dataclass
class TargetAccumulator:
    target_name: str
    unit_name: str | None = None
    callable_id: str | None = None
    callable_name: str | None = None
    resolved: bool = False
    excluded: bool = False
    incoming_nodes: list[str] = field(default_factory=list)
    incoming_edges: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'target_name': self.target_name,
            'unit_name': self.unit_name,
            'callable_id': self.callable_id,
            'callable_name': self.callable_name,
            'resolved': self.resolved,
            'excluded': self.excluded,
            'incoming_nodes': self.incoming_nodes,
            'incoming_edges': self.incoming_edges,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetAccumulator:
        """Create TargetAccumulator from dictionary."""
        return cls(
            target_name=data['target_name'],
            unit_name=data.get('unit_name'),
            callable_id=data.get('callable_id'),
            callable_name=data.get('callable_name'),
            resolved=data['resolved'],
            excluded=data['excluded'],
            incoming_nodes=data.get('incoming_nodes', []),
            incoming_edges=data.get('incoming_edges', 0),
        )


@dataclass
class TargetAnalysis:
    """
    Analysis of a single target callable for decorator candidates.

    Tracks how many integration points call this target and suggests
    appropriate decorators to reduce flow explosion.
    """
    key: str  # "unit::callable_id" or "UNRESOLVED::target_name"
    target_name: str
    unit_name: str | None
    callable_id: str | None
    callable_name: str | None
    resolved: bool
    excluded: bool
    incoming_node_count: int
    incoming_edge_count: int
    incoming_nodes: list[str] = field(default_factory=list)
    suggested_decorator: str | None = None
    suggested_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        result: dict[str, Any] = {
            'key': self.key,
            'target_name': self.target_name,
            'resolved': self.resolved,
            'excluded': self.excluded,
            'incoming_node_count': self.incoming_node_count,
            'incoming_edge_count': self.incoming_edge_count,
            'incoming_nodes': self.incoming_nodes,
        }
        if self.unit_name:
            result['unit_name'] = self.unit_name
        if self.callable_id:
            result['callable_id'] = self.callable_id
        if self.callable_name:
            result['callable_name'] = self.callable_name
        if self.suggested_decorator:
            result['suggested_decorator'] = self.suggested_decorator
        if self.suggested_type:
            result['suggested_type'] = self.suggested_type
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TargetAnalysis':
        """Create TargetAnalysis from dictionary."""
        return cls(
            key=data['key'],
            target_name=data['target_name'],
            unit_name=data.get('unit_name'),
            callable_id=data.get('callable_id'),
            callable_name=data.get('callable_name'),
            resolved=data['resolved'],
            excluded=data['excluded'],
            incoming_node_count=data['incoming_node_count'],
            incoming_edge_count=data['incoming_edge_count'],
            incoming_nodes=data.get('incoming_nodes', []),
            suggested_decorator=data.get('suggested_decorator'),
            suggested_type=data.get('suggested_type'),
        )


# =============================================================================
# Stage 4: Pattern Analysis
# =============================================================================

@dataclass
class CallableReference:
    """Reference to a callable in a pattern."""
    integration_id: str
    unit_name: str
    callable_name: str
    fully_qualified: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            'integration_id': self.integration_id,
            'unit_name': self.unit_name,
            'callable_name': self.callable_name,
            'fully_qualified': self.fully_qualified,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'CallableReference':
        """Create CallableReference from dictionary."""
        return cls(
            integration_id=data.get('integration_id', 'unknown'),
            unit_name=data.get('unit_name', 'unknown'),
            callable_name=data.get('callable_name', 'unknown'),
            fully_qualified=data.get('fully_qualified', 'unknown')
        )


@dataclass
class SubsequencePattern:
    """A common subsequence pattern found in flows."""
    pattern: list[str]  # Integration IDs
    length: int
    occurrences: int
    callables: list[CallableReference] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            'pattern': self.pattern,
            'length': self.length,
            'occurrences': self.occurrences,
            'callables': [c.to_dict() for c in self.callables],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SubsequencePattern':
        """Create SubsequencePattern from dictionary."""
        return cls(
            pattern=data.get('pattern', []),
            length=data.get('length', 0),
            occurrences=data.get('occurrences', 0),
            callables=[CallableReference.from_dict(c) for c in data.get('callables', [])]
        )


@dataclass
class CyclePattern:
    """A cycle detected in flow traversal."""
    pattern: list[str]  # Integration IDs forming the cycle
    length: int
    occurrences: int
    callables: list[CallableReference] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            'pattern': self.pattern,
            'length': self.length,
            'occurrences': self.occurrences,
            'callables': [c.to_dict() for c in self.callables],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'CyclePattern':
        """Create CyclePattern from dictionary."""
        return cls(
            pattern=data.get('pattern', []),
            length=data.get('length', 0),
            occurrences=data.get('occurrences', 0),
            callables=[CallableReference.from_dict(c) for c in data.get('callables', [])]
        )


@dataclass
class PatternAnalysisSummary:
    """Summary statistics from pattern analysis."""
    total_flows_analyzed: int
    long_flows: int
    long_flow_threshold: int
    unique_subsequences: int
    cycles_detected: int
    average_flow_length: float
    max_flow_length: int
    min_flow_length: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            'total_flows_analyzed': self.total_flows_analyzed,
            'long_flows': self.long_flows,
            'long_flow_threshold': self.long_flow_threshold,
            'unique_subsequences': self.unique_subsequences,
            'cycles_detected': self.cycles_detected,
            'average_flow_length': self.average_flow_length,
            'max_flow_length': self.max_flow_length,
            'min_flow_length': self.min_flow_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PatternAnalysisSummary':
        """Create PatternAnalysisSummary from dictionary."""
        return cls(
            total_flows_analyzed=data.get('total_flows_analyzed', 0),
            long_flows=data.get('long_flows', 0),
            long_flow_threshold=data.get('long_flow_threshold', 0),
            unique_subsequences=data.get('unique_subsequences', 0),
            cycles_detected=data.get('cycles_detected', 0),
            average_flow_length=data.get('average_flow_length', 0.0),
            max_flow_length=data.get('max_flow_length', 0),
            min_flow_length=data.get('min_flow_length', 0),
        )


@dataclass
class PatternAnalysisResult:
    """Complete pattern analysis results."""
    subsequences: list[SubsequencePattern] = field(default_factory=list)
    cycles: list[CyclePattern] = field(default_factory=list)
    flow_length_distribution: dict[int, int] = field(default_factory=dict)
    summary: PatternAnalysisSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result: dict[str, Any] = {
            'stage': 'pattern-analysis',
            'subsequences': [s.to_dict() for s in self.subsequences],
            'cycles': [c.to_dict() for c in self.cycles],
            'flow_length_distribution': self.flow_length_distribution,
        }
        if self.summary:
            result['summary'] = self.summary.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PatternAnalysisResult':
        """Create PatternAnalysisResult from dictionary."""
        return cls(
            subsequences=[SubsequencePattern.from_dict(s) for s in data.get('subsequences', [])],
            cycles=[CyclePattern.from_dict(c) for c in data.get('cycles', [])],
            flow_length_distribution=data.get('flow_length_distribution', {}),
            summary=PatternAnalysisSummary.from_dict(data.get('summary', {})) if 'summary' in data else None
        )


# =============================================================================
# Stage 5: Flow Enumeration
# =============================================================================

@dataclass
class EntryPointInfo:
    """
    Information about the entry point of a flow.

    Describes how to reach the first integration point in a flow.
    """
    integration_id: str  # First integration point ID
    unit_id: str  # Unit containing entry point
    callable_id: str  # Callable ID of entry point
    callable_name: str  # Human-readable callable name
    way_in: str  # Description of how to enter

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            'integration_id': self.integration_id,
            'unit_id': self.unit_id,
            'callable_id': self.callable_id,
            'callable_name': self.callable_name,
            'way_in': self.way_in,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntryPointInfo:
        """Create EntryPointInfo from the dictionary."""
        return cls(
            integration_id=data.get('integration_id', 'unknown'),
            unit_id=data.get('unit_id', 'unknown'),
            callable_id=data.get('callable_id', 'unknown'),
            callable_name=data.get('callable_name', 'unknown'),
            way_in=data.get('way_in', '')
        )


@dataclass
class FlowTermination:
    """Why traversal stopped - structural reasons only."""
    integration_id: str  # Last integration point reached
    reason: str  # 'no_outgoing_edges' | 'depth_limit' | 'cycle_detected'
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result: dict[str, Any] = {
            'integration_id': self.integration_id,
            'reason': self.reason,
        }
        if self.note:
            result['note'] = self.note
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowTermination:
        """Create TerminalNodeInfo from the dictionary."""
        return cls(
            integration_id=data.get('integration_id', 'unknown'),
            reason=data.get('reason'),
            note=data.get('note'),
        )


@dataclass
class Flow:
    """
    A complete flow from entry point to terminal node.

    Represents one possible execution path through the integration graph,
    starting at an entry point and ending at a terminal node (boundary
    or excluded operation).

    The sequence can contain either GraphNode objects or fixture placeholder
    strings like "FIXTURE_C000F001".
    """
    flow_id: str  # e.g., FLOW_0001
    description: str  # Human-readable flow description
    length: int  # Number of integration points
    sequence: list[GraphNode]  # Nodes or fixtures
    entry_point: EntryPointInfo
    termination: FlowTermination

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        # Convert sequence items
        sequence_dicts = [gn.to_dict() for gn in self.sequence]

        return {
            'flow_id': self.flow_id,
            'description': self.description,
            'length': self.length,
            'sequence': sequence_dicts,
            'entry_point': self.entry_point.to_dict(),
            'termination': self.termination.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Flow:
        """Create Flow from the dictionary."""
        # Parse sequence
        sequence: list[GraphNode] = []
        for item in data.get('sequence', []):
            node = GraphNode.from_dict(item) if isinstance(item, dict) else None
            if node is not None:
                sequence.append(node)

        return cls(
            flow_id=data.get('flow_id', 'unknown'),
            description=data.get('description', ''),
            length=data.get('length', 0),
            sequence=sequence,
            entry_point=EntryPointInfo.from_dict(data.get('entry_point', {})),
            termination=FlowTermination.from_dict(data.get('termination', {}))
        )


# =============================================================================
# Stage 6: Test Window Generation
# =============================================================================

@dataclass
class WindowEntryPoint:
    """Entry point information for a test window."""
    integration_id: str
    unit: str
    callable: str
    target: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            'integration_id': self.integration_id,
            'unit': self.unit,
            'callable': self.callable,
            'target': self.target,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowEntryPoint:
        """Create WindowEntryPoint from dictionary."""
        return cls(
            integration_id=data.get('integration_id', 'unknown'),
            unit=data.get('unit', 'unknown'),
            callable=data.get('callable', 'unknown'),
            target=data.get('target', 'unknown')
        )


@dataclass
class WindowExitPoint:
    """Exit point information for a test window."""
    integration_id: str
    unit: str
    callable: str
    target: str
    is_boundary: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            'integration_id': self.integration_id,
            'unit': self.unit,
            'callable': self.callable,
            'target': self.target,
            'is_boundary': self.is_boundary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowExitPoint:
        """Create WindowExitPoint from dictionary."""
        return cls(
            integration_id=data.get('integration_id', 'unknown'),
            unit=data.get('unit', 'unknown'),
            callable=data.get('callable', 'unknown'),
            target=data.get('target', 'unknown'),
            is_boundary=data.get('is_boundary', False)
        )


@dataclass
class TestWindow:
    """
    A sliding window of a flow representing a potential test scope.

    Windows are contiguous subsequences of integration points that form
    testable scopes. Multiple overlapping windows provide complete coverage.
    """
    window_id: str  # e.g., WINDOW_00001
    source_flow_id: str  # Flow this window came from
    start_position: int  # Index in original flow
    length: int  # Number of integration points
    integration_ids: list[str]  # Integration IDs in this window
    entry_point: WindowEntryPoint
    exit_point: WindowExitPoint
    description: str  # Human-readable window description
    sequence: list[GraphNode]  # Full node data for the window

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            'window_id': self.window_id,
            'source_flow_id': self.source_flow_id,
            'start_position': self.start_position,
            'length': self.length,
            'integration_ids': self.integration_ids,
            'entry_point': self.entry_point.to_dict(),
            'exit_point': self.exit_point.to_dict(),
            'description': self.description,
            'sequence': [node.to_dict() for node in self.sequence],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestWindow:
        return cls(
            window_id=data.get('window_id', 'unknown'),
            source_flow_id=data.get('source_flow_id', 'unknown'),
            start_position=data.get('start_position', 0),
            length=data.get('length', 0),
            integration_ids=data.get('integration_ids', []),
            entry_point=WindowEntryPoint.from_dict(data.get('entry_point', {})),
            exit_point=WindowExitPoint.from_dict(data.get('exit_point', {})),
            description=data.get('description', ''),
            sequence=data.get('sequence', [])
        )


# =============================================================================
# Helper Functions
# =============================================================================

def load_integration_points(data: dict[str, Any]) -> list[IntegrationPoint]:
    """Load integration points from Stage 1 output dictionary."""
    points_data = data.get('integrationPoints', [])
    return [IntegrationPoint.from_dict(p) for p in points_data]


def load_classification(data: dict[str, Any]) -> IntegrationPointClassification:
    """Load classification from Stage 2 output dictionary."""
    return IntegrationPointClassification.from_dict(data)


def load_graph_nodes(data: dict[str, Any]) -> list[GraphNode]:
    """Load graph nodes from Stage 3 output dictionary."""
    nodes_data = data.get('nodes', [])
    return [GraphNode.from_dict(n) for n in nodes_data]


def load_flows(data: dict[str, Any]) -> list[Flow]:
    """Load flows from Stage 5 output dictionary."""
    flows_data = data.get('flows', [])
    return [Flow.from_dict(f) for f in flows_data]
