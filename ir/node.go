package ir

import (
	"time"
	"unsafe"

	"github.com/theapemachine/manifesto/types"
)

/*
Node is one operation in the topology.

Op is the manifest operation identifier from the topology recipe
(e.g. "activation.gelu", "projection.linear"). It matches the op:
field in template/operation/*.yml and drives bind.method resolution
through types.OperationRegistry.

The ID/JitKernel/StreamID/SyncBarriers fields are the planner-output
additions per ARCHITECTURE.md §6. They stay zero/nil until the fusion
+ codegen + scheduler passes populate them.
*/
type Node struct {
	Kind        Kind
	Name        string
	Description string
	Created     *time.Time
	Updated     *time.Time
	Op          types.Op
	BindMethod  string
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
