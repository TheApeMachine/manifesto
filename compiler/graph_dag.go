package compiler

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/ir/dag"
	"github.com/theapemachine/manifesto/optimizer"
	"github.com/theapemachine/manifesto/tensor"
)

/*
BuildDAGFromGraph materializes a verified dag.Graph from the final ast.Graph
after every graph-mutating compiler pass. ComputeGraphs must always be built
from the same ast.Graph the executor dispatches — never from the pre-typer
snapshot produced by LowerTopology alone.
*/
func BuildDAGFromGraph(graph *ast.Graph) (*dag.Graph, error) {
	if graph == nil {
		return nil, fmt.Errorf("compiler: graph is required")
	}

	if err := NormalizeGraph(graph); err != nil {
		return nil, fmt.Errorf("compiler: build dag: %w", err)
	}

	dagGraph := dag.NewGraph()
	producers := make(map[string]*dag.Node, len(graph.Inputs)+len(graph.Nodes))

	for _, inputName := range graph.Inputs {
		if inputName == "" {
			continue
		}

		if _, exists := producers[inputName]; exists {
			return nil, fmt.Errorf("compiler: build dag: duplicate input %q", inputName)
		}

		inputNode := dag.NewNode(inputName, dag.OpInput, tensor.Shape{})
		dagGraph.AddNode(inputNode)
		producers[inputName] = inputNode
	}

	for _, node := range graph.Nodes {
		if err := addGraphNodeToDAG(node, dagGraph, producers); err != nil {
			return nil, err
		}
	}

	if err := dagGraph.Verify(); err != nil {
		return nil, fmt.Errorf("compiler: build dag: %w", err)
	}

	return dagGraph, nil
}

func addGraphNodeToDAG(
	node *ast.GraphNode,
	dagGraph *dag.Graph,
	producers map[string]*dag.Node,
) error {
	if node == nil {
		return fmt.Errorf("compiler: build dag: nil node")
	}

	if _, exists := producers[node.ID]; exists {
		return fmt.Errorf("compiler: build dag: duplicate node id %q", node.ID)
	}

	dagNode := dag.NewNode(node.ID, dagOpTypeForGraphNode(node), tensor.Shape{})

	for _, inputName := range node.Inputs {
		if inputName == "" {
			continue
		}

		producer, ok := producers[inputName]

		if !ok {
			return fmt.Errorf(
				"compiler: build dag: node %q references unknown input %q",
				node.ID, inputName,
			)
		}

		if err := dagNode.AddInput(producer); err != nil {
			return fmt.Errorf("compiler: build dag: %w", err)
		}
	}

	dagGraph.AddNode(dagNode)
	producers[node.ID] = dagNode

	return nil
}

func dagOpTypeForGraphNode(node *ast.GraphNode) dag.OpType {
	if node == nil {
		return dag.OpMatmul
	}

	op := strings.ToLower(strings.TrimSpace(node.Op))

	if op == "input" || op == "io.input" {
		return dag.OpInput
	}

	if op == optimizer.FuseOp {
		return dag.OpMatmul
	}

	return dag.OpMatmul
}
