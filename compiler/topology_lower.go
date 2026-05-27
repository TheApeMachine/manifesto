package compiler

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/ir/dag"
	"github.com/theapemachine/manifesto/tensor"
)

/*
LoweredGraph holds both representations of a compiled compute graph:

  - The semantic *ast.Graph carries op kinds, attributes, and weight bindings
    used by the execution backend to dispatch device.Backend methods.
  - The parallel *dag.Graph carries the scheduling DAG used by runtime
    plan generation (topology layers).

Both representations share node IDs and input wiring.
*/
type LoweredGraph struct {
	AST *ast.Graph
	DAG *dag.Graph
}

/*
LowerTopology expands a parsed topology (possibly containing `repeat`
templates) into a flat ast.Graph + dag.Graph pair. Inputs become dag input
nodes; every other topology node becomes one ast.GraphNode plus one dag.Node
that points at its declared inputs by ID. The function performs basic
validation (no duplicate IDs, every input refers to a known producer).
*/
func LowerTopology(topology *ast.Topology) (*LoweredGraph, error) {
	if topology == nil {
		return nil, fmt.Errorf("compiler: lower topology: topology is required")
	}

	expanded, err := expandTopology(topology)

	if err != nil {
		return nil, err
	}

	astGraph := &ast.Graph{
		Inputs:   append([]string(nil), expanded.Inputs...),
		Outputs:  make(map[string]string),
		Metadata: make(map[string]any),
		Bindings: symbolMapFromTopology(expanded.Bindings),
	}

	dagGraph := dag.NewGraph()

	producers := make(map[string]*dag.Node)

	for _, inputName := range expanded.Inputs {
		if inputName == "" {
			continue
		}

		if _, exists := producers[inputName]; exists {
			return nil, fmt.Errorf("compiler: lower topology: duplicate input %q", inputName)
		}

		inputNode := dag.NewNode(inputName, dag.OpInput, tensor.Shape{})
		dagGraph.AddNode(inputNode)
		producers[inputName] = inputNode
	}

	for _, node := range expanded.Nodes {
		if err := lowerOneNode(node, astGraph, dagGraph, producers); err != nil {
			return nil, err
		}
	}

	if err := resolveTopologyOutputs(expanded.Outputs, astGraph, producers); err != nil {
		return nil, err
	}

	if err := dagGraph.Verify(); err != nil {
		return nil, fmt.Errorf("compiler: lower topology: %w", err)
	}

	return &LoweredGraph{
		AST: astGraph,
		DAG: dagGraph,
	}, nil
}

func lowerOneNode(
	node ast.Node,
	astGraph *ast.Graph,
	dagGraph *dag.Graph,
	producers map[string]*dag.Node,
) error {
	if node.ID == "" {
		return fmt.Errorf("compiler: lower topology: node missing id (op %q)", node.Op)
	}

	if _, exists := producers[node.ID]; exists {
		return fmt.Errorf("compiler: lower topology: duplicate node id %q", node.ID)
	}

	// Resolve every raw YAML input name to its producer node ID so that
	// ast.GraphNode.Inputs is always keyed by producer ID. Topology
	// authors reference upstream nodes by their declared `out:` names,
	// but downstream consumers (typer, optimizer, dispatcher) all key
	// off node IDs — without this resolution they can't find producers
	// whose output name differs from their node ID.
	resolvedInputs := make([]string, 0, len(node.In))

	for _, inputName := range node.In {
		if inputName == "" {
			continue
		}

		producer, ok := producers[inputName]

		if !ok {
			return fmt.Errorf(
				"compiler: lower topology: node %q references unknown input %q",
				node.ID, inputName,
			)
		}

		resolvedInputs = append(resolvedInputs, producer.ID())
	}

	graphNode := &ast.GraphNode{
		ID:         node.ID,
		Op:         node.Op,
		Inputs:     resolvedInputs,
		Attributes: cloneAttributes(node.Config),
		Metadata:   make(map[string]any),
	}

	weightSpec, err := weightSpecForNode(node)

	if err != nil {
		return err
	}

	if weightSpec != nil {
		weightSlice, err := weightSliceFromSpec(weightSpec)

		if err != nil {
			return err
		}

		graphNode.Weights = &ast.BoundWeight{
			TensorName: weightSpec.Weight,
			BiasName:   weightSpec.Bias,
			Slice:      weightSlice,
		}
	}

	astGraph.Nodes = append(astGraph.Nodes, graphNode)

	dagNode := dag.NewNode(node.ID, opTypeForOp(node.Op), tensor.Shape{})

	for _, inputName := range node.In {
		if inputName == "" {
			continue
		}

		producer := producers[inputName]

		if err := dagNode.AddInput(producer); err != nil {
			return fmt.Errorf("compiler: lower topology: %w", err)
		}
	}

	dagGraph.AddNode(dagNode)

	// A node produces one output per declared out name. Later declarations
	// of the same value name rebind downstream consumers to this producer,
	// which is how manifest state updates express write-then-read ordering.
	producers[node.ID] = dagNode

	for _, outputName := range node.Out {
		if outputName == "" || outputName == node.ID {
			continue
		}

		producers[outputName] = dagNode
	}

	return nil
}

func weightSpecForNode(node ast.Node) (*ast.WeightSpec, error) {
	if node.Weights != nil {
		return node.Weights, nil
	}

	fromSafeTensors, err := weightSpecFromSafeTensors(node.Config)

	if err != nil {
		return nil, err
	}

	if fromSafeTensors != nil {
		return fromSafeTensors, nil
	}

	if !opUsesDefaultWeight(node.Op) {
		return nil, nil
	}

	return &ast.WeightSpec{Weight: node.ID + ".weight"}, nil
}

func weightSpecFromSafeTensors(config map[string]any) (*ast.WeightSpec, error) {
	if config == nil {
		return nil, nil
	}

	raw, ok := config["from_safetensors"]

	if !ok {
		return nil, nil
	}

	section, ok := raw.(map[string]any)

	if !ok {
		return nil, fmt.Errorf("compiler: lower topology: from_safetensors must be a map")
	}

	spec := &ast.WeightSpec{}

	if err := setStringWeightField(section, "weight", &spec.Weight); err != nil {
		return nil, err
	}

	if err := setStringWeightField(section, "bias", &spec.Bias); err != nil {
		return nil, err
	}

	if err := setStringWeightField(section, "slice_axis", &spec.SliceAxis); err != nil {
		return nil, err
	}

	spec.SliceStart = section["slice_start"]
	spec.SliceEnd = section["slice_end"]

	if spec.Weight == "" {
		return nil, fmt.Errorf("compiler: lower topology: from_safetensors requires weight")
	}

	return spec, nil
}

func setStringWeightField(section map[string]any, key string, target *string) error {
	raw, ok := section[key]

	if !ok || raw == nil {
		return nil
	}

	value, ok := raw.(string)

	if !ok {
		return fmt.Errorf("compiler: lower topology: from_safetensors.%s must be a string", key)
	}

	*target = value

	return nil
}

func opUsesDefaultWeight(op string) bool {
	switch op {
	case "convolution.conv2d",
		"embedding.token",
		"math.batchnorm_denorm",
		"math.groupnorm",
		"math.layernorm",
		"math.rmsnorm",
		"projection.linear":
		return true
	default:
		return false
	}
}

func weightSliceFromSpec(spec *ast.WeightSpec) (*ast.WeightSlice, error) {
	if spec.SliceAxis == "" && spec.SliceStart == nil && spec.SliceEnd == nil {
		return nil, nil
	}

	start, err := weightSliceIndex(spec.SliceStart)

	if err != nil {
		return nil, fmt.Errorf("compiler: lower topology: invalid weight slice start: %w", err)
	}

	end, err := weightSliceIndex(spec.SliceEnd)

	if err != nil {
		return nil, fmt.Errorf("compiler: lower topology: invalid weight slice end: %w", err)
	}

	return &ast.WeightSlice{
		Axis:  spec.SliceAxis,
		Start: start,
		End:   end,
	}, nil
}

func weightSliceIndex(value any) (int64, error) {
	switch typed := value.(type) {
	case nil:
		return 0, nil
	case int:
		return int64(typed), nil
	case int64:
		return typed, nil
	case float64:
		return int64(typed), nil
	case string:
		parsed, err := strconv.ParseInt(strings.TrimSpace(typed), 10, 64)

		if err != nil {
			return 0, err
		}

		return parsed, nil
	default:
		return 0, fmt.Errorf("unsupported type %T", value)
	}
}

func resolveTopologyOutputs(
	outputs map[string]string,
	astGraph *ast.Graph,
	producers map[string]*dag.Node,
) error {
	for outputName, outputRef := range outputs {
		if outputName == "" {
			continue
		}

		if outputRef == "" {
			outputRef = outputName
		}

		producer, ok := producers[outputRef]

		if !ok {
			return fmt.Errorf(
				"compiler: lower topology: output %q references unknown value %q",
				outputName,
				outputRef,
			)
		}

		astGraph.Outputs[outputName] = producer.ID()
	}

	return nil
}

/*
opTypeForOp picks a coarse dag OpType for one ast Op kind. Today the dag
layer only distinguishes inputs from compute nodes; everything that isn't a
plain Input maps to OpMatmul as a generic compute marker. The real op
identity lives on the ast.GraphNode and is read by the dispatcher.
*/
func opTypeForOp(op string) dag.OpType {
	op = strings.ToLower(strings.TrimSpace(op))

	if op == "input" || op == "io.input" {
		return dag.OpInput
	}

	return dag.OpMatmul
}

func cloneAttributes(config map[string]any) map[string]any {
	if len(config) == 0 {
		return make(map[string]any)
	}

	out := make(map[string]any, len(config))

	for key, value := range config {
		out[key] = value
	}

	return out
}

func symbolMapFromTopology(bindings map[string]int64) ir.SymbolMap {
	if len(bindings) == 0 {
		return nil
	}

	out := make(ir.SymbolMap, len(bindings))

	for symbol, value := range bindings {
		out[symbol] = value
	}

	return out
}
