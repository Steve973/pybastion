# Integration Flow Generation Procedure

## 1. Introduction and Overview

### 1.1 Purpose

This specification defines the procedure for generating integration flow test specifications
from unit ledgers. Integration flow testing focuses on testing SEAMS between units, not
internal execution paths within units.

**Key Distinction:**
- **Unit tests**: Verify all execution items (EIs) within a single unit
- **Integration tests**: Verify how units compose across their boundaries

### 1.2 Goals

- Enumerate all integration points (seams) across units in scope
- Construct integration graph showing relationships between integration points
- Identify complete flows from entry points to terminal nodes
- Generate sliding windows for each flow to enable focused integration testing

### 1.3 Input

- Collection of unit ledgers (YAML files following unit-ledger-spec)
- Ledger discovery index (optional, for resolving unit names to ledger files)

### 1.4 Output

Multi-stage outputs:
1. Integration points collection (flat list)
2. Classified integration points (entry/intermediate/terminal)
3. Integration graph (nodes + edges)
4. Complete flows (entry → terminal sequences)
5. Test windows (sliding windows of each flow)

## 2. Stage-by-Stage Procedure

### Stage 1: Collect Integration Points

[To be completed]

### Stage 2: Classify Integration Points

[To be completed]

### Stage 3: Build Integration Graph

[To be completed]

### Stage 4: Enumerate Flows

[To be completed]

### Stage 5: Generate Windows

[To be completed]

## 3. Verification Gates

[To be completed]

## 4. Examples

[To be completed]
