package dag

import (
	"fmt"
)

/*
Graph is a compute DAG used for execution scheduling.
*/
type Graph struct {
	nodes []*Node
}

/*
NewGraph constructs an empty compute DAG.
*/
func NewGraph() *Graph {
	return &Graph{
		nodes: make([]*Node, 0),
	}
}

/*
AddNode registers one node in the graph.
*/
func (graph *Graph) AddNode(node *Node) {
	if graph == nil || node == nil {
		return
	}

	graph.nodes = append(graph.nodes, node)
}

/*
Nodes returns all graph nodes in insertion order.
*/
func (graph *Graph) Nodes() []*Node {
	if graph == nil {
		return nil
	}

	out := make([]*Node, len(graph.nodes))
	copy(out, graph.nodes)

	return out
}

type graphIndex struct {
	nodes map[string]*Node
	users map[string][]*Node
}

func (graph *Graph) index() (*graphIndex, error) {
	nodes := graph.Nodes()
	index := &graphIndex{
		nodes: make(map[string]*Node, len(nodes)),
		users: make(map[string][]*Node, len(nodes)),
	}

	for _, node := range nodes {
		if node.ID() == "" {
			return nil, fmt.Errorf("dag: node ID is required")
		}

		if _, exists := index.nodes[node.ID()]; exists {
			return nil, fmt.Errorf("dag: duplicate node ID %q", node.ID())
		}

		index.nodes[node.ID()] = node
	}

	for _, node := range nodes {
		for _, input := range node.Inputs() {
			if input == nil || input.ID() == "" {
				return nil, fmt.Errorf("dag: node %q has invalid input", node.ID())
			}

			if _, ok := index.nodes[input.ID()]; !ok {
				return nil, fmt.Errorf("dag: node %q references unknown input %q", node.ID(), input.ID())
			}

			index.users[input.ID()] = append(index.users[input.ID()], node)
		}
	}

	return index, nil
}

/*
Verify checks node IDs and dependency integrity.
*/
func (graph *Graph) Verify() error {
	if graph == nil {
		return fmt.Errorf("dag: graph is required")
	}

	_, err := graph.index()

	return err
}

/*
TopologyLayers groups nodes into sequential execution layers.
Independent nodes within a layer may run concurrently.
*/
func (graph *Graph) TopologyLayers() ([][]*Node, error) {
	index, err := graph.index()

	if err != nil {
		return nil, err
	}

	remaining := make(map[string]int, len(index.nodes))

	for nodeID := range index.nodes {
		remaining[nodeID] = len(index.nodes[nodeID].Inputs())
	}

	layers := make([][]*Node, 0, len(index.nodes))
	scheduled := 0

	for scheduled < len(index.nodes) {
		layer := make([]*Node, 0)

		for nodeID, pendingInputs := range remaining {
			if pendingInputs != 0 {
				continue
			}

			layer = append(layer, index.nodes[nodeID])
		}

		if len(layer) == 0 {
			return nil, fmt.Errorf("dag: cycle detected")
		}

		for _, node := range layer {
			delete(remaining, node.ID())
			scheduled++

			for _, user := range index.users[node.ID()] {
				remaining[user.ID()]--
			}
		}

		layers = append(layers, layer)
	}

	return layers, nil
}
