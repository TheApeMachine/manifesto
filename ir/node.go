package ir

import (
	"time"
	"unsafe"
)

/*
Node is one operation in the topology.

The Kind/Name/Operation/Weight/Inputs/Outputs/timestamp fields are the
original recipe-compiler representation. The ID/JitKernel/StreamID/
SyncBarriers fields are the planner-output additions per
ARCHITECTURE.md §6. The planner-output fields stay zero/nil until the
fusion + codegen + scheduler passes populate them.

Existing callers that read only the original fields continue to
compile and work unchanged.
*/
type Node struct {
	Kind        Kind
	Name        string
	Description string
	Created     *time.Time
	Updated     *time.Time
	Operation   Operation
	Weight      *Weight
	Inputs      []*Port
	Outputs     []*Port

	// Planner-output fields (ARCHITECTURE.md §6).
	//
	// ID is a topology-scope unique identifier. Assigned during
	// planning. Used as the index into flat per-node slices the
	// executor materializes at session init.
	ID int32
	// JitKernel points to the compiled native binary for fused
	// elementwise nodes. nil for nodes that dispatch to static kernels
	// rather than JIT-compiled fusion clusters.
	JitKernel unsafe.Pointer
	// StreamID assigns this node to a hardware stream/queue. Default
	// 0 = single-stream serial execution; planner emits values > 0
	// when independent subgraphs can run concurrently.
	StreamID int32
	// SyncBarriers are the wait/signal events the executor must encode
	// around this node's dispatch so cross-stream dependencies hold.
	// Empty when StreamID is 0 and the node has no inter-stream deps.
	SyncBarriers []SyncEvent
}
