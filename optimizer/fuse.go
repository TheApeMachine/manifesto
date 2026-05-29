package optimizer

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
)

/*
FuseOp is the synthetic ast.GraphNode.Op string assigned to a node that
holds a FusionAST in its Attributes. Codegen recognises this op and emits a
single kernel for the whole subgraph.
*/
const FuseOp = "optimizer.fusion"

/*
FuseAttributeAST is the ast.GraphNode.Attributes key under which the
FusionAST pointer is stored.
*/
const FuseAttributeAST = "fusion_ast"

/*
FusionStats summarizes one fusion-pass invocation.
*/
type FusionStats struct {
	Clusters     int
	NodesFused   int
	NodesRemoved int
}

/*
Fuse clusters contiguous elementwise nodes into FusionAST subgraphs.

The pass walks the graph in reverse order and, for each unvisited elementwise
node, recursively builds an expression tree rooted at it. A predecessor is
absorbed into the same FusionAST when:

  - The predecessor's op is elementwise (per isElementwiseOp).
  - The predecessor's output has exactly one consumer (the current cluster).
  - The predecessor is not already visited as part of another cluster.
  - The predecessor is not referenced by graph.Outputs (those values must
    remain materialized so callers can observe them).

Predecessors that fail any condition become external inputs to the FusionAST.
This captures both linear chains (the original pass) and DAG-shaped
patterns like `Mul(Sigmoid(x), y)` (SwiGLU) and `Add(Add(a, b), c)`
(residual fan-in).

graph is mutated in place: clustered nodes are replaced by a single
synthetic node carrying the FusionAST in its Attributes.
*/
func Fuse(graph *ast.Graph) (FusionStats, error) {
	if graph == nil {
		return FusionStats{}, fmt.Errorf("optimizer: graph is required")
	}

	stats := FusionStats{}

	producers := make(map[string]int)

	for index, node := range graph.Nodes {
		producers[node.ID] = index
	}

	consumerCount := make(map[string]int)

	for _, node := range graph.Nodes {
		for _, inputName := range node.Inputs {
			consumerCount[inputName]++
		}
	}

	for _, outputRef := range graph.Outputs {
		consumerCount[outputRef]++
	}

	visited := make([]bool, len(graph.Nodes))
	absorbed := make([]bool, len(graph.Nodes))

	// Walk in reverse so consumer-first traversal lets each root absorb
	// every eligible predecessor before that predecessor is independently
	// considered as a root.
	for index := len(graph.Nodes) - 1; index >= 0; index-- {
		if visited[index] || absorbed[index] {
			continue
		}

		node := graph.Nodes[index]

		_, eligible := isElementwiseOp(node.Op)

		if !eligible {
			visited[index] = true
			continue
		}

		cluster := newClusterBuilder(graph, producers, consumerCount, absorbed)

		root := cluster.build(node)

		if cluster.size() <= 1 {
			visited[index] = true
			continue
		}

		fusionAST := &FusionAST{
			Root:             root,
			InputPorts:       cluster.inputs,
			OutputPort:       node.ID,
			DType:            graph.ExecutionDType,
			ContainedNodeIDs: cluster.containedIDs(),
		}

		fused := &ast.GraphNode{
			ID:         node.ID,
			Op:         FuseOp,
			Inputs:     append([]string(nil), cluster.inputs...),
			InputTypes: fusionInputTypes(graph, cluster.inputs),
			OutputType: node.OutputType,
			Attributes: map[string]any{
				FuseAttributeAST: fusionAST,
			},
			Metadata: map[string]any{
				"fused_node_count": cluster.size(),
			},
		}

		graph.Nodes[index] = fused
		visited[index] = true

		for clusterIndex := range cluster.absorbedIndices {
			if clusterIndex == index {
				continue
			}

			absorbed[clusterIndex] = true
		}

		stats.Clusters++
		stats.NodesFused += cluster.size()
		stats.NodesRemoved += cluster.size() - 1
	}

	result := make([]*ast.GraphNode, 0, len(graph.Nodes))

	for index, node := range graph.Nodes {
		if absorbed[index] {
			continue
		}

		result = append(result, node)
	}

	graph.Nodes = result

	return stats, nil
}

/*
clusterBuilder grows one FusionAST recursively. It tracks which producer
node indices have been absorbed, deduplicates external input ports, and
exposes the running ContainedNodeIDs list.
*/
type clusterBuilder struct {
	graph           *ast.Graph
	producers       map[string]int
	consumerCount   map[string]int
	absorbed        []bool
	absorbedIndices map[int]struct{}
	inputs          []string
	inputIndex      map[string]int
}

func newClusterBuilder(
	graph *ast.Graph,
	producers map[string]int,
	consumerCount map[string]int,
	absorbed []bool,
) *clusterBuilder {
	return &clusterBuilder{
		graph:           graph,
		producers:       producers,
		consumerCount:   consumerCount,
		absorbed:        absorbed,
		absorbedIndices: make(map[int]struct{}),
		inputs:          make([]string, 0),
		inputIndex:      make(map[string]int),
	}
}

/*
build constructs the AST subtree for one node, recursing into eligible
predecessors. The returned ASTNode references either inline subexpressions
(for absorbed predecessors) or external InputPorts indices (for everything
else).
*/
func (builder *clusterBuilder) build(node *ast.GraphNode) *ASTNode {
	nodeType, eligible := isElementwiseOp(node.Op)

	if !eligible {
		return builder.externalInput(node.ID)
	}

	producerIndex := builder.producers[node.ID]
	builder.absorbedIndices[producerIndex] = struct{}{}

	arity := nodeType.Arity()
	children := make([]*ASTNode, 0, arity)

	for _, inputName := range node.Inputs {
		children = append(children, builder.buildOperand(inputName))
	}

	return &ASTNode{
		Type:     nodeType,
		Children: children,
	}
}

/*
buildOperand decides whether to inline a producer or keep it external.
*/
func (builder *clusterBuilder) buildOperand(inputName string) *ASTNode {
	producerIndex, ok := builder.producers[inputName]

	if !ok {
		return builder.externalInput(inputName)
	}

	if builder.absorbed[producerIndex] {
		return builder.externalInput(inputName)
	}

	if _, already := builder.absorbedIndices[producerIndex]; already {
		// Producer already absorbed by a sibling branch within the same
		// cluster; keep the value external so it is materialized once.
		return builder.externalInput(inputName)
	}

	if builder.consumerCount[inputName] != 1 {
		return builder.externalInput(inputName)
	}

	producer := builder.graph.Nodes[producerIndex]

	if _, eligible := isElementwiseOp(producer.Op); !eligible {
		return builder.externalInput(inputName)
	}

	return builder.build(producer)
}

func (builder *clusterBuilder) externalInput(name string) *ASTNode {
	if existing, ok := builder.inputIndex[name]; ok {
		return &ASTNode{Type: NodeInput, InputIndex: existing}
	}

	index := len(builder.inputs)
	builder.inputs = append(builder.inputs, name)
	builder.inputIndex[name] = index

	return &ASTNode{Type: NodeInput, InputIndex: index}
}

func (builder *clusterBuilder) size() int {
	return len(builder.absorbedIndices)
}

func (builder *clusterBuilder) containedIDs() []string {
	out := make([]string, 0, len(builder.absorbedIndices))

	for index := range builder.absorbedIndices {
		out = append(out, builder.graph.Nodes[index].ID)
	}

	return out
}

func fusionInputTypes(graph *ast.Graph, inputNames []string) []ir.PortType {
	producerTypes := make(map[string]ir.PortType, len(graph.Nodes))

	for _, graphNode := range graph.Nodes {
		if graphNode == nil || graphNode.OutputType.DType == dtype.Invalid {
			continue
		}

		producerTypes[graphNode.ID] = graphNode.OutputType
	}

	inputTypes := make([]ir.PortType, len(inputNames))

	for index, inputName := range inputNames {
		if portType, ok := producerTypes[inputName]; ok {
			inputTypes[index] = portType
			continue
		}

		inputTypes[index] = boundaryInputPortType(graph, inputName)
	}

	return inputTypes
}

func boundaryInputPortType(graph *ast.Graph, inputName string) ir.PortType {
	for _, graphNode := range graph.Nodes {
		if graphNode == nil {
			continue
		}

		for slotIndex, producerID := range graphNode.Inputs {
			if producerID != inputName {
				continue
			}

			if slotIndex < len(graphNode.InputTypes) {
				return graphNode.InputTypes[slotIndex]
			}
		}
	}

	return ir.PortType{}
}
