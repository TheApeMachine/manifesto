package compiler

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/optimizer"
	"github.com/theapemachine/manifesto/types"
)

/*
planningBridge is the typed *ir.Topology equivalent of one *ast.Graph,
together with the port maps needed to populate ARCHITECTURE.md §6
InputPorts / OutputPorts after workspace planning.
*/
type planningBridge struct {
	Topology      *ir.Topology
	BoundaryPorts map[string]*ir.Port
	ProducerPorts map[string]*ir.Port
}

/*
buildPlanningTopology converts a typed *ast.Graph into *ir.Topology with
shared *ir.Port pointers across producer/consumer edges. graph.Nodes must
already be in topological order (NormalizeGraph).
*/
func buildPlanningTopology(
	graph *ast.Graph,
	registry *types.OperationRegistry,
) (*planningBridge, error) {
	if graph == nil {
		return nil, fmt.Errorf("compiler: graph is required")
	}

	topology := &ir.Topology{
		Kind: ir.KindTopology,
	}

	producerOutputs := make(map[string]*ir.Port, len(graph.Nodes))
	boundaryInputs := make(map[string]*ir.Port, len(graph.Inputs))
	boundarySet := boundaryInputSet(graph)

	for _, name := range graph.Inputs {
		if name == "" {
			continue
		}

		boundaryInputs[name] = &ir.Port{}
	}

	for _, node := range graph.Nodes {
		if node == nil {
			continue
		}

		irNode, err := newPlanningNode(node, registry)

		if err != nil {
			return nil, fmt.Errorf("compiler: node %q: %w", node.ID, err)
		}

		for slotIndex, producerID := range node.Inputs {
			port, err := resolvePlanningInputPort(
				node, slotIndex, producerID,
				producerOutputs, boundaryInputs, boundarySet,
			)

			if err != nil {
				return nil, err
			}

			irNode.Inputs[slotIndex] = port
		}

		outputPort := planningOutputPort(node, irNode)

		irNode.Outputs = []*ir.Port{outputPort}
		producerOutputs[node.ID] = outputPort

		topology.Nodes = append(topology.Nodes, irNode)
	}

	return &planningBridge{
		Topology:      topology,
		BoundaryPorts: boundaryInputs,
		ProducerPorts: producerOutputs,
	}, nil
}

func newPlanningNode(node *ast.GraphNode, registry *types.OperationRegistry) (*ir.Node, error) {
	op := types.Op(node.Op)

	irNode := &ir.Node{
		Kind:   ir.KindNode,
		Name:   node.ID,
		Op:     op,
		Inputs: make([]*ir.Port, len(node.Inputs)),
	}

	if node.Op == optimizer.FuseOp || isCompilerIntrinsicOp(node.Op) {
		return irNode, nil
	}

	if registry == nil {
		return irNode, nil
	}

	bindMethod, err := op.BindMethod(registry)

	if err != nil {
		return nil, err
	}

	irNode.BindMethod = bindMethod

	return irNode, nil
}

func resolvePlanningInputPort(
	node *ast.GraphNode,
	slotIndex int,
	producerID string,
	producerOutputs map[string]*ir.Port,
	boundaryInputs map[string]*ir.Port,
	boundarySet map[string]struct{},
) (*ir.Port, error) {
	if port, ok := producerOutputs[producerID]; ok {
		return port, nil
	}

	boundary, isBoundary := boundaryInputs[producerID]

	if !isBoundary {
		if _, isDeclaredNode := boundarySet[producerID]; isDeclaredNode {
			return nil, fmt.Errorf(
				"compiler: node %q input %q: graph input used before declaration in graph.Inputs",
				node.ID, producerID,
			)
		}

		return nil, fmt.Errorf(
			"compiler: node %q input %q: producer %q is not defined before consumer (run NormalizeGraph)",
			node.ID, producerID, producerID,
		)
	}

	if boundary.Type.DType == dtype.Invalid && slotIndex < len(node.InputTypes) {
		boundary.Type = node.InputTypes[slotIndex]
	}

	return boundary, nil
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
attachGraphIOPorts maps manifest graph inputs and outputs to workspace byte
offsets per ARCHITECTURE.md §6.
*/
func attachGraphIOPorts(
	topology *ir.Topology,
	graph *ast.Graph,
	bridge *planningBridge,
) error {
	if topology == nil || graph == nil || bridge == nil {
		return fmt.Errorf("compiler: attach graph io ports: missing topology or graph")
	}

	topology.InputPorts = make(map[string]int32, len(graph.Inputs))

	for _, inputName := range graph.Inputs {
		if inputName == "" {
			continue
		}

		port := bridge.BoundaryPorts[inputName]

		if port == nil || port.Allocation == nil {
			return fmt.Errorf("compiler: graph input %q has no workspace allocation", inputName)
		}

		topology.InputPorts[inputName] = int32(port.Allocation.BaseOffset)
	}

	topology.OutputPorts = make(map[string]int32, len(graph.Outputs))

	for outputName, producerID := range graph.Outputs {
		if outputName == "" {
			continue
		}

		port := bridge.ProducerPorts[producerID]

		if port == nil || port.Allocation == nil {
			return fmt.Errorf(
				"compiler: graph output %q producer %q has no workspace allocation",
				outputName, producerID,
			)
		}

		topology.OutputPorts[outputName] = int32(port.Allocation.BaseOffset)
	}

	return nil
}

/*
PlanGraphOptions configures workspace planning and stream scheduling for
one *ast.Graph.
*/
type PlanGraphOptions struct {
	Registry       *types.OperationRegistry
	Bindings       ir.SymbolMap
	Align          int64
	StreamSchedule ir.StreamScheduleOptions
}

/*
PlanGraph normalizes, bridges, plans workspace, maps graph I/O ports, and
runs stream scheduling. The graph must already have been processed by the
typer (graph.Bindings and per-node InputTypes / OutputType set).
*/
func PlanGraph(graph *ast.Graph, options PlanGraphOptions) (*ir.Topology, error) {
	if graph == nil {
		return nil, fmt.Errorf("compiler: graph is required")
	}

	if err := NormalizeGraph(graph); err != nil {
		return nil, fmt.Errorf("compiler: plan graph: %w", err)
	}

	bridge, err := buildPlanningTopology(graph, options.Registry)

	if err != nil {
		return nil, fmt.Errorf("compiler: plan graph: %w", err)
	}

	bindings := options.Bindings

	if bindings == nil {
		bindings = graph.Bindings
	}

	if bindings == nil {
		bindings = ir.SymbolMap{}
	}

	align := options.Align

	if align <= 0 {
		align = 64
	}

	if err := ir.PlanWorkspace(bridge.Topology, ir.PlanWorkspaceOptions{
		Bindings: bindings,
		Align:    align,
	}); err != nil {
		return bridge.Topology, fmt.Errorf("compiler: plan workspace: %w", err)
	}

	if err := attachGraphIOPorts(bridge.Topology, graph, bridge); err != nil {
		return bridge.Topology, fmt.Errorf("compiler: plan graph: %w", err)
	}

	if err := ir.ScheduleStreams(bridge.Topology, options.StreamSchedule); err != nil {
		return bridge.Topology, fmt.Errorf("compiler: schedule streams: %w", err)
	}

	return bridge.Topology, nil
}

// TopologyForPlanning is retained for tests that build planning bridges
// without running the full PlanGraph pipeline.
func TopologyForPlanning(graph *ast.Graph) *ir.Topology {
	bridge, err := buildPlanningTopology(graph, nil)

	if err != nil || bridge == nil {
		return nil
	}

	return bridge.Topology
}
