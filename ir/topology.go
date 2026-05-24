package ir

import "time"

/*
Topology is the IR of a model architecture.

The Nodes/Edges/timestamps fields are the original lightweight
representation used by the recipe compiler. The Workspace/InputPorts/
OutputPorts fields are the planner's output added per
ARCHITECTURE.md §6 — they are populated by the static memory planner
(GAPS.md §3.1 Phase 4) and remain zero/nil until that pass runs.

Treat the planner-output fields as optional in any code that consumes
Topology directly: legacy callers can ignore them; future executor
code reads them to set up the workspace and entry/exit port mapping.
*/
type Topology struct {
	Kind        Kind
	Name        string
	Description string
	Created     *time.Time
	Updated     *time.Time
	Nodes       []*Node
	Edges       []*Edge

	// Planner-output fields (ARCHITECTURE.md §6).
	//
	// Workspace describes the single contiguous device memory region
	// the executor must allocate at session init. Zero-valued before
	// the planner runs.
	Workspace WorkspaceLayout
	// InputPorts maps human-readable manifest input names to the
	// workspace offset where the executor expects the host to place
	// the input tensor before each dispatch.
	InputPorts map[string]int32
	// OutputPorts maps manifest output names to the workspace offset
	// the host reads at an execution boundary.
	OutputPorts map[string]int32
}
