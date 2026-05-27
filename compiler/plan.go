package compiler

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
)

/*
TopologyForPlanning builds an *ir.Topology from a typed *ast.Graph so the
static memory planner (ir.PlanWorkspace) has a structure it can run
liveness + interval coloring over.

Why this bridge exists:

  - The runtime dispatcher walks an *ast.Graph (the lowered topology
    output of compiler.LowerTopology). Each GraphNode carries the
    operation kind, its bound weights, and — after the typer pass —
    the unified PortType for each input slot and the output.
  - The static memory planner in manifesto/ir operates on *ir.Topology,
    which encodes the same shape with *ir.Node and *ir.Port. PlanWorkspace
    needs every Port.Type populated (DType, ShapeSchema) before it can
    compute byte sizes via PortByteSize.
  - The planner also needs the producer's output Port and every
    downstream consumer's input Port at that edge to be the *same Port
    pointer, so liveness analysis correctly extends the producer's
    lifetime to the last consumer's step.

This function takes the typed *ast.Graph and produces the equivalent
*ir.Topology with the sharing relationship intact. Graph-boundary inputs
(those declared on graph.Inputs without a producing node) become Port
instances whose type is adopted from the first consumer's InputTypes
slot — the typer has already unified the boundary's permissive type with
that consumer's expected type, so InputTypes carries the concrete shape
the planner needs.

The graph must be typed (typer.Run has been called) for the resulting
topology to be plannable. Untyped graphs produce Ports with dtype.Invalid
which PortByteSize rejects with a clear error.
*/
func TopologyForPlanning(graph *ast.Graph) *ir.Topology {
	if graph == nil {
		return nil
	}

	topology := &ir.Topology{
		Kind: ir.KindTopology,
	}

	// producerOutputs[node.ID] points at the *ir.Port that node produces.
	// Every downstream input that references node.ID resolves to this
	// exact pointer so the planner sees a shared edge.
	producerOutputs := make(map[string]*ir.Port, len(graph.Nodes))

	// boundaryInputs[name] points at the *ir.Port created for one graph-
	// level input declaration. Shared across all consumers that read the
	// same boundary input. Type is adopted from the first consumer.
	boundaryInputs := make(map[string]*ir.Port, len(graph.Inputs))

	for _, name := range graph.Inputs {
		boundaryInputs[name] = &ir.Port{}
	}

	for _, node := range graph.Nodes {
		if node == nil {
			continue
		}

		irNode := &ir.Node{
			Kind:    ir.KindNode,
			Name:    node.ID,
			Inputs:  make([]*ir.Port, len(node.Inputs)),
			Outputs: nil,
		}

		for slotIndex, producerID := range node.Inputs {
			if port, ok := producerOutputs[producerID]; ok {
				irNode.Inputs[slotIndex] = port
				continue
			}

			boundary, isBoundary := boundaryInputs[producerID]

			if !isBoundary {
				// Producer that hasn't been seen yet — graph isn't
				// topologically clean. Fabricate a placeholder so the
				// planner can still walk the structure; PortByteSize will
				// reject it with a clear "unbound" error.
				boundary = &ir.Port{}
				boundaryInputs[producerID] = boundary
			}

			// The typer's edge-unified InputTypes carry the concrete
			// PortType for this boundary edge. Adopt it the first time
			// we see a consumer for this boundary name so the planner
			// can size the boundary tensor.
			if boundary.Type.DType == dtype.Invalid && slotIndex < len(node.InputTypes) {
				boundary.Type = node.InputTypes[slotIndex]
			}

			irNode.Inputs[slotIndex] = boundary
		}

		// Every GraphNode produces exactly one output, identified by
		// node.ID. The typer's OutputType is the unified producer type
		// after all downstream consumers have had a chance to constrain
		// it; that's the PortType the planner uses to size this output.
		outputPort := planningOutputPort(node, irNode)

		irNode.Outputs = []*ir.Port{outputPort}
		producerOutputs[node.ID] = outputPort

		topology.Nodes = append(topology.Nodes, irNode)
	}

	return topology
}

func planningOutputPort(node *ast.GraphNode, irNode *ir.Node) *ir.Port {
	if planningAliasOp(node.Op) && len(irNode.Inputs) > 0 && irNode.Inputs[0] != nil {
		return irNode.Inputs[0]
	}

	return &ir.Port{
		Type: node.OutputType,
	}
}

func planningAliasOp(op string) bool {
	switch op {
	case "shape.view_as_heads", "shape.merge_heads":
		return true
	default:
		return false
	}
}

/*
PlanGraph types-then-plans a single *ast.Graph and returns the resulting
*ir.Topology with its Workspace populated. The graph must already have
been processed by the typer (graph.Bindings and per-node InputTypes /
OutputType set) — typically this means programCompiler.CompileAssets has
run typer.Run on it before reaching the planner.

Plan options:

  - Bindings comes from graph.Bindings, the SymbolMap the typer
    accumulated by unifying every edge. Any dynamic dimension that did
    not get bound during unification will surface as an "unbound symbol"
    error from PortByteSize, which is the right failure mode — the
    program author needs to provide a binding (max sequence length, max
    batch, etc.) for the planner to size the workspace.
  - Align defaults to 64 bytes, matching ARCHITECTURE.md §5.1's
    AVX-512-cache-line / Apple Silicon / CUDA-cache-line floor.
*/
func PlanGraph(graph *ast.Graph) (*ir.Topology, error) {
	if graph == nil {
		return nil, fmt.Errorf("compiler: graph is required")
	}

	topology := TopologyForPlanning(graph)

	bindings := graph.Bindings

	if bindings == nil {
		bindings = ir.SymbolMap{}
	}

	if err := ir.PlanWorkspace(topology, ir.PlanWorkspaceOptions{
		Bindings: bindings,
		Align:    64,
	}); err != nil {
		return topology, fmt.Errorf("compiler: plan workspace: %w", err)
	}

	return topology, nil
}
