package runtime

import (
	"fmt"

	"github.com/theapemachine/manifesto/ir"
)

/*
ExecutionPlan is a verified schedule for one compute graph.
Layers group node IDs that may run concurrently; layers themselves run in order.
*/
type ExecutionPlan struct {
	GraphName string
	Layers    [][]string
}

/*
NewExecutionPlan verifies compute IR and derives parallel execution layers.
*/
func NewExecutionPlan(graphName string, compute *ir.Graph) (*ExecutionPlan, error) {
	if graphName == "" {
		return nil, fmt.Errorf("runtime plan: graph name is required")
	}

	if compute == nil {
		return nil, fmt.Errorf("runtime plan: compute graph is required")
	}

	if err := compute.Verify(); err != nil {
		return nil, fmt.Errorf("runtime plan: verify graph %q: %w", graphName, err)
	}

	layerNodes, err := compute.TopologyLayers()

	if err != nil {
		return nil, fmt.Errorf("runtime plan: topology layers for %q: %w", graphName, err)
	}

	layers := make([][]string, 0, len(layerNodes))

	for _, layer := range layerNodes {
		nodeIDs := make([]string, 0, len(layer))

		for _, node := range layer {
			if node.OpType() == ir.OpInput {
				continue
			}

			nodeIDs = append(nodeIDs, node.ID())
		}

		if len(nodeIDs) == 0 {
			continue
		}

		layers = append(layers, nodeIDs)
	}

	return &ExecutionPlan{
		GraphName: graphName,
		Layers:    layers,
	}, nil
}
