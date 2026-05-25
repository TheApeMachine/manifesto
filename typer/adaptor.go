package typer

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
)

/*
SynthesisStats summarizes one adaptor-synthesis pass.
*/
type SynthesisStats struct {
	CastsInserted     int
	TransposesInserted int
	ReshapesInserted   int
	Unresolved         int
}

/*
SynthesizeAdaptors consumes the EdgeErrors produced by Infer and inserts
the matching adaptor node between offending producer and consumer. The
graph is mutated in place; node IDs for new adaptors are derived from
the consumer's ID + slot so re-running the typer on the rewritten graph
is idempotent.

Per ARCHITECTURE.md §4.2:
  - AdaptorHint "cast"      → shape.cast      (dtype change)
  - AdaptorHint "transpose" → shape.transpose (layout change)
  - AdaptorHint "reshape"   → shape.reshape   (rank-preserving shape change)

Edge errors with an empty AdaptorHint (rank mismatch, semantic-kind
mismatch, constraint violation) are returned as unresolved — the
caller must fix them in the manifest or fail compilation.
*/
func SynthesizeAdaptors(graph *ast.Graph, edgeErrors []EdgeError) (SynthesisStats, []EdgeError, error) {
	if graph == nil {
		return SynthesisStats{}, nil, fmt.Errorf("typer: graph is required")
	}

	stats := SynthesisStats{}
	var unresolved []EdgeError

	indexByID := make(map[string]int, len(graph.Nodes))

	for index, node := range graph.Nodes {
		indexByID[node.ID] = index
	}

	for _, edgeError := range edgeErrors {
		if edgeError.AdaptorHint == "" {
			unresolved = append(unresolved, edgeError)
			stats.Unresolved++
			continue
		}

		adaptor, err := buildAdaptorNode(edgeError)

		if err != nil {
			return stats, unresolved, err
		}

		consumerIndex, ok := indexByID[edgeError.Consumer]

		if !ok {
			return stats, unresolved, fmt.Errorf(
				"typer: consumer %q from edge error not in graph", edgeError.Consumer,
			)
		}

		consumer := graph.Nodes[consumerIndex]

		if edgeError.ConsumerSlot < 0 || edgeError.ConsumerSlot >= len(consumer.Inputs) {
			return stats, unresolved, fmt.Errorf(
				"typer: consumer %q slot %d out of range", edgeError.Consumer, edgeError.ConsumerSlot,
			)
		}

		insertBefore := consumerIndex

		// Place the adaptor node directly in front of its consumer so
		// topological order remains valid.
		graph.Nodes = append(graph.Nodes, nil)
		copy(graph.Nodes[insertBefore+1:], graph.Nodes[insertBefore:])
		graph.Nodes[insertBefore] = adaptor

		consumer.Inputs[edgeError.ConsumerSlot] = adaptor.ID

		switch edgeError.AdaptorHint {
		case "cast":
			stats.CastsInserted++
		case "transpose":
			stats.TransposesInserted++
		case "reshape":
			stats.ReshapesInserted++
		}

		// Rebuild the ID index after insertion since indices shifted.
		for index, node := range graph.Nodes {
			indexByID[node.ID] = index
		}
	}

	return stats, unresolved, nil
}

/*
buildAdaptorNode constructs the synthetic node for one EdgeError.
*/
func buildAdaptorNode(edgeError EdgeError) (*ast.GraphNode, error) {
	adaptorID := fmt.Sprintf("adaptor_%s_%s_%d", edgeError.AdaptorHint, edgeError.Consumer, edgeError.ConsumerSlot)

	switch edgeError.AdaptorHint {
	case "cast":
		return &ast.GraphNode{
			ID:     adaptorID,
			Op:     "shape.cast",
			Inputs: []string{edgeError.Producer},
			Attributes: map[string]any{
				"dtype":      edgeError.ExpectedType.DType,
				"from_dtype": edgeError.ProducerType.DType,
			},
			Metadata: map[string]any{
				"synthesized_for_edge": edgeError.Reason,
			},
			InputTypes: []ir.PortType{edgeError.ProducerType},
			OutputType: castOutputType(edgeError),
		}, nil
	case "transpose":
		return &ast.GraphNode{
			ID:     adaptorID,
			Op:     "shape.transpose",
			Inputs: []string{edgeError.Producer},
			Attributes: map[string]any{
				"layout":      edgeError.ExpectedType.Layout,
				"from_layout": edgeError.ProducerType.Layout,
			},
			Metadata: map[string]any{
				"synthesized_for_edge": edgeError.Reason,
			},
			InputTypes: []ir.PortType{edgeError.ProducerType},
			OutputType: transposeOutputType(edgeError),
		}, nil
	case "reshape":
		return &ast.GraphNode{
			ID:     adaptorID,
			Op:     "shape.reshape",
			Inputs: []string{edgeError.Producer},
			Attributes: map[string]any{
				"shape":      edgeError.ExpectedType.ShapeSchema,
				"from_shape": edgeError.ProducerType.ShapeSchema,
			},
			Metadata: map[string]any{
				"synthesized_for_edge": edgeError.Reason,
			},
			InputTypes: []ir.PortType{edgeError.ProducerType},
			OutputType: reshapeOutputType(edgeError),
		}, nil
	default:
		return nil, fmt.Errorf("typer: unknown adaptor hint %q", edgeError.AdaptorHint)
	}
}

func castOutputType(edgeError EdgeError) ir.PortType {
	result := edgeError.ProducerType

	if edgeError.ExpectedType.DType != dtype.Invalid {
		result.DType = edgeError.ExpectedType.DType
	}

	return result
}

func transposeOutputType(edgeError EdgeError) ir.PortType {
	result := edgeError.ProducerType

	if edgeError.ExpectedType.Layout != ir.LayoutUnspecified {
		result.Layout = edgeError.ExpectedType.Layout
	}

	return result
}

func reshapeOutputType(edgeError EdgeError) ir.PortType {
	result := edgeError.ProducerType

	if len(edgeError.ExpectedType.ShapeSchema.Dimensions) > 0 {
		result.ShapeSchema = edgeError.ExpectedType.ShapeSchema
	}

	return result
}
