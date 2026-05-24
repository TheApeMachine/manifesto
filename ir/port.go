package ir

import (
	"github.com/theapemachine/manifesto/tensor"
)

/*
Port is one input or output on a node.

The Tensor field is the original lightweight representation used by
the recipe compiler — it points to the concrete Tensor backing this
port at execution time. The Type and Allocation fields are the
planner/typer-output additions per ARCHITECTURE.md §4.1 and §6.

Existing callers that read only Tensor continue to compile and work
unchanged. Future compiler passes populate Type during PortType
unification (Phase 2.2) and Allocation during the planning pass
(Phase 4).
*/
type Port struct {
	Tensor *tensor.Tensor

	// ID is a topology-scope unique identifier. Zero before planning;
	// nonzero after the planner assigns IDs.
	ID int32
	// Type is the typed contract for this port. Zero-valued before
	// PortType unification runs; the DType / ShapeSchema / Layout /
	// Kind fields settle to their unified values once the compiler's
	// unification pass completes.
	Type PortType
	// Allocation is the static memory plan for this port. nil before
	// the planner runs; non-nil after Phase 4 completes.
	Allocation *PortAllocation
}
