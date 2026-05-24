package ir

/*
WorkspaceLayout is the planner's output: a single contiguous device
memory region with statically-assigned offsets for every port in the
topology. ARCHITECTURE.md §5.

The layout is built once per compiled topology; it does not change
between dispatches unless the dynamic-symbol bindings change (in which
case stride math is re-evaluated but offsets stay put).
*/
type WorkspaceLayout struct {
	// Size is the total byte size of the workspace region the executor
	// must allocate before the first dispatch. Includes alignment
	// padding.
	Size int64
	// Align is the minimum base-pointer alignment the workspace
	// allocator must guarantee. Spec mandates 64 bytes (AVX-512 line
	// size) across all backends per §5.1.
	Align int64
	// Allocations are the per-port intervals the coloring allocator
	// produced. Multiple ports whose lifetimes do not overlap may
	// share the same Offset.
	Allocations []Interval
}

/*
Interval is one port's allocation in the workspace, with its liveness
range. The interval-coloring allocator (ARCHITECTURE.md §5.1) maps
non-overlapping intervals to the same Offset to minimize workspace
size.
*/
type Interval struct {
	// PortID matches Port.ID.
	PortID int32
	// Start is the topological step index where this port is produced.
	Start int
	// End is the topological step index of the final consumer of this
	// port. Inclusive.
	End int
	// Offset is the byte offset within the workspace. Always
	// WorkspaceLayout.Align-aligned.
	Offset int64
	// Size is the byte size of the interval. Rounded up to
	// WorkspaceLayout.Align by the allocator.
	Size int64
}

/*
Overlaps reports whether this interval's liveness range overlaps with
another's. Two intervals may share an Offset only when they do NOT
overlap. Used by the interval-coloring allocator and by tests that
verify the allocator's output is sound.
*/
func (interval Interval) Overlaps(other Interval) bool {
	if interval.End < other.Start {
		return false
	}

	if other.End < interval.Start {
		return false
	}

	return true
}
