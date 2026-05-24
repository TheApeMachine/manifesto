package ir

import (
	"fmt"
	"sort"
)

/*
PlanWorkspaceOptions configures the static memory planner.

Bindings resolves any symbolic dimensions in port types to concrete
values. These typically come from the PortType unifier (Unify) after
all node-to-node connections in the topology have been unified; the
planner cannot compute byte sizes for ports whose shape symbols are
not bound.

Align is the minimum workspace base-pointer alignment in bytes.
ARCHITECTURE.md §5.1 mandates 64 (AVX-512 cache line and the largest
SIMD lane width across CPU and GPU targets). Each interval's offset
and size is rounded up to this boundary.
*/
type PlanWorkspaceOptions struct {
	Bindings SymbolMap
	Align    int64
}

/*
PlanWorkspace runs liveness analysis followed by the interval-coloring
allocator over the topology's nodes, populating Topology.Workspace and
every Port.Allocation in place.

The topology's Nodes must already be in topological execution order —
the planner does not re-sort them. Edges are used to identify shared
Ports across producer/consumer boundaries via the Port pointer
(producer.Outputs[i] and consumer.Inputs[j] must be the same pointer
when they refer to the same logical tensor).

This implements ARCHITECTURE.md §5.1 (Static Liveness Analysis &
Offset Allocation) plus the alignment rule from §5.1's note that the
workspace base must itself be 64-byte aligned.
*/
func PlanWorkspace(topology *Topology, options PlanWorkspaceOptions) error {
	if topology == nil {
		return fmt.Errorf("planner: topology is required")
	}

	if options.Align <= 0 {
		options.Align = 64
	}

	if options.Bindings == nil {
		options.Bindings = make(SymbolMap)
	}

	AssignPortIDs(topology)

	intervals, err := AnalyzeLiveness(topology.Nodes, options.Bindings)
	if err != nil {
		return err
	}

	totalSize, allocated := AllocateOffsets(intervals, options.Align)

	topology.Workspace = WorkspaceLayout{
		Size:        totalSize,
		Align:       options.Align,
		Allocations: allocated,
	}

	byPortID := make(map[int32]Interval, len(allocated))

	for _, interval := range allocated {
		byPortID[interval.PortID] = interval
	}

	attachAllocationsToPorts(topology, byPortID)

	return nil
}

/*
AssignPortIDs walks every node and gives each input/output Port a
unique sequential ID. Idempotent: Ports that already carry a non-zero
ID are left alone, and the next free ID continues from there.

Liveness analysis and the allocator key off Port.ID, so this must run
first. The IDs are also what the executor uses to resolve a node's
inputs/outputs to workspace offsets at session init.
*/
func AssignPortIDs(topology *Topology) {
	if topology == nil {
		return
	}

	seen := make(map[*Port]bool)
	nextID := int32(1)

	bumpToCurrent := func(id int32) {
		if id >= nextID {
			nextID = id + 1
		}
	}

	for _, node := range topology.Nodes {
		for _, port := range node.Outputs {
			if port == nil || seen[port] {
				continue
			}

			bumpToCurrent(port.ID)
			seen[port] = true
		}

		for _, port := range node.Inputs {
			if port == nil || seen[port] {
				continue
			}

			bumpToCurrent(port.ID)
			seen[port] = true
		}
	}

	seen = make(map[*Port]bool)

	for _, node := range topology.Nodes {
		for _, port := range node.Outputs {
			if port == nil || seen[port] {
				continue
			}

			if port.ID == 0 {
				port.ID = nextID
				nextID++
			}

			seen[port] = true
		}

		for _, port := range node.Inputs {
			if port == nil || seen[port] {
				continue
			}

			if port.ID == 0 {
				port.ID = nextID
				nextID++
			}

			seen[port] = true
		}
	}
}

/*
AnalyzeLiveness walks `nodes` in the given order (assumed topological)
and produces one Interval per distinct output Port. Each interval's
Start is the step index at which the producing node runs, and End is
the step index of the latest consumer.

Inputs that the planner cannot find a producer for (manifest inputs,
weight tensors) are treated as live for the entire range [0, lastStep]
— they enter the workspace at session init and remain pinned until
the last node that reads them runs.

Returns an error when any port's PortType refers to an unbound dynamic
symbol (the planner cannot compute byte sizes without that binding).
*/
func AnalyzeLiveness(nodes []*Node, bindings SymbolMap) ([]Interval, error) {
	if len(nodes) == 0 {
		return nil, nil
	}

	intervals := make(map[int32]*Interval)

	registerOutput := func(port *Port, stepIndex int) error {
		if port == nil || port.ID == 0 {
			return nil
		}

		if _, exists := intervals[port.ID]; exists {
			return nil
		}

		size, err := PortByteSize(port, bindings)
		if err != nil {
			return fmt.Errorf("port id=%d: %w", port.ID, err)
		}

		intervals[port.ID] = &Interval{
			PortID: port.ID,
			Start:  stepIndex,
			End:    stepIndex,
			Size:   size,
		}

		return nil
	}

	extendForConsumer := func(port *Port, stepIndex int) error {
		if port == nil || port.ID == 0 {
			return nil
		}

		existing, found := intervals[port.ID]
		if !found {
			// Unproduced input (graph input / weight): live from step 0.
			size, err := PortByteSize(port, bindings)
			if err != nil {
				return fmt.Errorf("port id=%d (unproduced input): %w", port.ID, err)
			}

			intervals[port.ID] = &Interval{
				PortID: port.ID,
				Start:  0,
				End:    stepIndex,
				Size:   size,
			}

			return nil
		}

		if stepIndex > existing.End {
			existing.End = stepIndex
		}

		return nil
	}

	for stepIndex, node := range nodes {
		if node == nil {
			continue
		}

		for _, output := range node.Outputs {
			if err := registerOutput(output, stepIndex); err != nil {
				return nil, err
			}
		}

		for _, input := range node.Inputs {
			if err := extendForConsumer(input, stepIndex); err != nil {
				return nil, err
			}
		}
	}

	ordered := make([]Interval, 0, len(intervals))

	for _, interval := range intervals {
		ordered = append(ordered, *interval)
	}

	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].Start != ordered[j].Start {
			return ordered[i].Start < ordered[j].Start
		}

		return ordered[i].PortID < ordered[j].PortID
	})

	return ordered, nil
}

/*
AllocateOffsets runs the interval-coloring allocator from
ARCHITECTURE.md §5.1 / §3 implementation blueprint. Intervals whose
liveness ranges do not overlap may share an Offset; intervals that do
overlap get distinct Offsets.

Each interval's Size is rounded up to `align` before allocation, and
each chosen Offset is `align`-aligned. The total workspace size
returned is the high-water mark across all allocated intervals
(already align-aligned).

Returns the total size and the input intervals with their Offset and
Size fields populated.
*/
func AllocateOffsets(intervals []Interval, align int64) (int64, []Interval) {
	if align <= 0 {
		align = 64
	}

	if len(intervals) == 0 {
		return 0, nil
	}

	work := make([]Interval, len(intervals))
	copy(work, intervals)

	sort.Slice(work, func(i, j int) bool {
		if work[i].Start != work[j].Start {
			return work[i].Start < work[j].Start
		}

		return work[i].PortID < work[j].PortID
	})

	var live []liveBlock
	var highWater int64

	for index := range work {
		work[index].Size = alignUp(work[index].Size, align)

		// Evict expired blocks (those whose End is strictly less than
		// the new interval's Start; touching intervals at the same step
		// still overlap by inclusive semantics).
		var stillLive []liveBlock

		for _, block := range live {
			if block.end >= work[index].Start {
				stillLive = append(stillLive, block)
			}
		}

		live = stillLive

		offset := chooseLowestNonConflictingOffset(work[index].Size, live, align)

		work[index].Offset = offset
		live = append(live, liveBlock{
			offset: offset,
			size:   work[index].Size,
			end:    work[index].End,
		})

		if highWater < offset+work[index].Size {
			highWater = offset + work[index].Size
		}
	}

	return alignUp(highWater, align), work
}

/*
liveBlock is one currently-live allocation tracked by the
interval-coloring loop in AllocateOffsets. Defined at package scope
because chooseLowestNonConflictingOffset (a top-level helper) needs
to reference it.
*/
type liveBlock struct {
	offset int64
	size   int64
	end    int
}

/*
chooseLowestNonConflictingOffset scans the currently-live allocations
and returns the smallest align-aligned offset where a new block of
`size` bytes fits without overlapping any live block.

Algorithm: walk live blocks sorted by their Offset. Try inserting at
the current candidate offset; if it would overlap the next live block,
advance the candidate to right after that block (still aligned).
Repeat until a gap is found. Worst case is O(n²) but n is small (live
set size is bounded by the workgraph width) so the simple algorithm is
the right tradeoff against asymptotic improvements.
*/
func chooseLowestNonConflictingOffset(size int64, live []liveBlock, align int64) int64 {
	if len(live) == 0 {
		return 0
	}

	sorted := make([]liveBlock, len(live))
	copy(sorted, live)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].offset < sorted[j].offset
	})

	candidate := int64(0)

	for {
		conflict := false

		for _, block := range sorted {
			if candidate >= block.offset+block.size {
				continue
			}

			if candidate+size <= block.offset {
				continue
			}

			// Overlap: jump past this block.
			candidate = alignUp(block.offset+block.size, align)
			conflict = true
			break
		}

		if !conflict {
			return candidate
		}
	}
}

func alignUp(value, align int64) int64 {
	if align <= 1 {
		return value
	}

	return (value + align - 1) &^ (align - 1)
}

/*
PortByteSize computes the byte size of a port's tensor, resolving any
symbolic dimensions through the bindings map.

Returns an error if the port has no PortType (DType=Invalid), if any
dimension references an unbound symbol, or if dtype.BytesFor rejects
the element count.
*/
func PortByteSize(port *Port, bindings SymbolMap) (int64, error) {
	if port == nil {
		return 0, fmt.Errorf("port is nil")
	}

	dimensions := port.Type.ShapeSchema.Dimensions

	if len(dimensions) == 0 {
		// Scalar destination (e.g., a reduction output): one element.
		bytes, err := port.Type.DType.BytesFor(1)
		if err != nil {
			return 0, fmt.Errorf("dtype %s rejected scalar size: %w", port.Type.DType, err)
		}

		return int64(bytes), nil
	}

	var elementCount int64 = 1

	for index, dimension := range dimensions {
		if !dimension.IsSymbolic() {
			elementCount *= dimension.Static

			continue
		}

		value, bound := bindings[dimension.Symbol]
		if !bound {
			return 0, fmt.Errorf(
				"dim[%d] symbol %q is unbound (run unifier first to populate bindings)",
				index, dimension.Symbol,
			)
		}

		elementCount *= value
	}

	bytes, err := port.Type.DType.BytesFor(int(elementCount))
	if err != nil {
		return 0, fmt.Errorf("dtype %s rejected %d elements: %w", port.Type.DType, elementCount, err)
	}

	return int64(bytes), nil
}

func attachAllocationsToPorts(topology *Topology, byPortID map[int32]Interval) {
	seen := make(map[*Port]bool)

	attach := func(port *Port) {
		if port == nil || seen[port] {
			return
		}

		seen[port] = true

		interval, found := byPortID[port.ID]
		if !found {
			return
		}

		port.Allocation = &PortAllocation{
			PortID:     port.ID,
			BaseOffset: interval.Offset,
			PortType:   port.Type,
		}
	}

	for _, node := range topology.Nodes {
		for _, port := range node.Outputs {
			attach(port)
		}

		for _, port := range node.Inputs {
			attach(port)
		}
	}
}
