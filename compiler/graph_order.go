package compiler

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
)

/*
NormalizeGraph validates dependency integrity and reorders graph.Nodes into
topological execution order. Every node input must name either a graph
boundary (graph.Inputs) or a node that appears earlier after sorting.

The typer, optimizer, and adaptor passes mutate graph.Nodes in place; this
pass is the single authority on execution order before planning and DAG
materialization. Call it after every graph-changing compiler stage.
*/
func NormalizeGraph(graph *ast.Graph) error {
	if graph == nil {
		return fmt.Errorf("compiler: graph is required")
	}

	nodeByID, err := indexGraphNodes(graph)

	if err != nil {
		return err
	}

	boundaryInputs := boundaryInputSet(graph)

	if err := validateGraphEdges(graph, nodeByID, boundaryInputs); err != nil {
		return err
	}

	sorted, err := topologicalSort(graph.Nodes, boundaryInputs, nodeByID)

	if err != nil {
		return err
	}

	graph.Nodes = sorted

	return nil
}

func indexGraphNodes(graph *ast.Graph) (map[string]*ast.GraphNode, error) {
	nodeByID := make(map[string]*ast.GraphNode, len(graph.Nodes))

	for _, node := range graph.Nodes {
		if node == nil {
			return nil, fmt.Errorf("compiler: graph contains a nil node")
		}

		if node.ID == "" {
			return nil, fmt.Errorf("compiler: graph node is missing id (op %q)", node.Op)
		}

		if _, exists := nodeByID[node.ID]; exists {
			return nil, fmt.Errorf("compiler: duplicate node id %q", node.ID)
		}

		nodeByID[node.ID] = node
	}

	return nodeByID, nil
}

func boundaryInputSet(graph *ast.Graph) map[string]struct{} {
	boundaryInputs := make(map[string]struct{}, len(graph.Inputs))

	for _, inputName := range graph.Inputs {
		if inputName == "" {
			continue
		}

		boundaryInputs[inputName] = struct{}{}
	}

	return boundaryInputs
}

func validateGraphEdges(
	graph *ast.Graph,
	nodeByID map[string]*ast.GraphNode,
	boundaryInputs map[string]struct{},
) error {
	for _, node := range graph.Nodes {
		for _, inputName := range node.Inputs {
			if inputName == "" {
				continue
			}

			if _, isBoundary := boundaryInputs[inputName]; isBoundary {
				continue
			}

			if _, isNode := nodeByID[inputName]; !isNode {
				return fmt.Errorf(
					"compiler: node %q references unknown input %q",
					node.ID, inputName,
				)
			}
		}
	}

	for outputName, producerID := range graph.Outputs {
		if outputName == "" {
			continue
		}

		if producerID == "" {
			return fmt.Errorf("compiler: graph output %q has no producer", outputName)
		}

		if _, isBoundary := boundaryInputs[producerID]; isBoundary {
			continue
		}

		if _, isNode := nodeByID[producerID]; !isNode {
			return fmt.Errorf(
				"compiler: graph output %q references unknown producer %q",
				outputName, producerID,
			)
		}
	}

	return nil
}

func topologicalSort(
	nodes []*ast.GraphNode,
	boundaryInputs map[string]struct{},
	nodeByID map[string]*ast.GraphNode,
) ([]*ast.GraphNode, error) {
	if len(nodes) == 0 {
		return nil, nil
	}

	inDegree := make(map[string]int, len(nodes))
	successors := make(map[string][]string, len(nodes))

	for _, node := range nodes {
		inDegree[node.ID] = 0
	}

	for _, node := range nodes {
		for _, inputName := range node.Inputs {
			if inputName == "" {
				continue
			}

			if _, isBoundary := boundaryInputs[inputName]; isBoundary {
				continue
			}

			if _, isNode := nodeByID[inputName]; !isNode {
				continue
			}

			inDegree[node.ID]++
			successors[inputName] = append(successors[inputName], node.ID)
		}
	}

	ready := make([]*ast.GraphNode, 0, len(nodes))

	for _, node := range nodes {
		if inDegree[node.ID] == 0 {
			ready = append(ready, node)
		}
	}

	sorted := make([]*ast.GraphNode, 0, len(nodes))

	for len(ready) > 0 {
		current := ready[0]
		ready = ready[1:]

		sorted = append(sorted, current)

		for _, successorID := range successors[current.ID] {
			inDegree[successorID]--

			if inDegree[successorID] == 0 {
				ready = append(ready, nodeByID[successorID])
			}
		}
	}

	if len(sorted) != len(nodes) {
		return nil, fmt.Errorf("compiler: graph contains a cycle")
	}

	return sorted, nil
}
