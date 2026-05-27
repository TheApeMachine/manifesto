package typer

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
)

/*
EdgeError captures one failed unification between a producer's output
PortType and a consumer's expected input PortType. The AdaptorHint is
carried through from ir.UnificationError so the synthesis pass knows
which adaptor node to insert.

Producer / Consumer are node IDs; ConsumerSlot is the index into the
consumer's Inputs slice where the offending edge lives. This is exactly
the information the synthesizer needs to rewire one edge through a new
adaptor node.
*/
type EdgeError struct {
	Producer     string
	Consumer     string
	ConsumerSlot int
	ProducerType ir.PortType
	ExpectedType ir.PortType
	Reason       string
	AdaptorHint  string
}

func (edgeError *EdgeError) Error() string {
	return fmt.Sprintf(
		"typer: edge %s → %s[%d]: %s",
		edgeError.Producer, edgeError.Consumer, edgeError.ConsumerSlot, edgeError.Reason,
	)
}

/*
InferStats summarizes the typer's traversal.
*/
type InferStats struct {
	NodesTyped       int
	EdgesUnified     int
	BindingsResolved int
}

/*
Infer assigns PortTypes to every node's inputs/outputs and unifies across
every edge. The graph is mutated in place: each GraphNode gets InputTypes
and OutputType populated; graph.Bindings is replaced with the cumulative
SymbolMap; constraint violations and unrecoverable mismatches surface as
errors. Recoverable mismatches (with AdaptorHints) are returned as
EdgeErrors so the synthesis pass can rewire them.

Nodes are processed in their existing graph order, which the lowerer
already produced in topological order. Symbolic dimensions on the
producer side bind once and propagate downstream through Bindings —
that's what makes this Hindley-Milner rather than a one-shot type check.
*/
func Infer(graph *ast.Graph) (InferStats, []EdgeError, error) {
	if graph == nil {
		return InferStats{}, nil, fmt.Errorf("typer: graph is required")
	}

	stats := InferStats{}
	bindings := make(ir.SymbolMap)

	if graph.Bindings != nil {
		for symbol, value := range graph.Bindings {
			bindings[symbol] = value
		}
	}

	producerTypes := make(map[string]ir.PortType, len(graph.Nodes))

	// Graph inputs are typed permissively; they enter the graph as
	// unconstrained PortTypes (Float32, single symbolic dim N, contiguous).
	for _, inputName := range graph.Inputs {
		if portType, ok := graphInputPortType(inputName); ok {
			producerTypes[inputName] = portType
			continue
		}

		producerTypes[inputName] = anyTensor()
	}

	var edgeErrors []EdgeError

	for _, node := range graph.Nodes {
		// Idempotency: nodes that already carry a populated OutputType
		// (typically the adaptor nodes the synthesis pass just inserted)
		// keep their existing types so re-running the typer doesn't
		// trigger spurious rank/shape mismatches against the adaptor's
		// permissive Inputs spec.
		if node.OutputType.DType != dtype.Invalid {
			producerTypes[node.ID] = node.OutputType
			stats.NodesTyped++
			continue
		}

		spec, ok := LookupSpec(node.Op)

		if !ok {
			// Unknown ops are not a typer error — the dispatcher will
			// reject them at execution time if they're really missing.
			// We still propagate the node's first input type as a
			// best-effort output, AND we write it back to node.OutputType
			// so the planner's TopologyForPlanning sees a typed port
			// (otherwise port.Type stays at the zero PortType and
			// PortByteSize fails with "unsupported dtype 0").
			fallback := bestEffortPassthrough(node, producerTypes)
			node.OutputType = fallback
			producerTypes[node.ID] = fallback
			continue
		}

		boundInputs, slotErrors := unifyNodeInputs(node, spec, producerTypes, bindings)
		stats.EdgesUnified += len(node.Inputs)

		if len(slotErrors) > 0 {
			edgeErrors = append(edgeErrors, slotErrors...)
		}

		node.InputTypes = append([]ir.PortType(nil), boundInputs...)

		outputType, err := deriveOutputType(node, spec, boundInputs, bindings)

		if err != nil {
			return stats, edgeErrors, err
		}

		node.OutputType = outputType
		producerTypes[node.ID] = outputType
		stats.NodesTyped++
	}

	stats.BindingsResolved = len(bindings)
	graph.Bindings = bindings

	return stats, edgeErrors, nil
}

func unifyNodeInputs(
	node *ast.GraphNode,
	spec OpSpec,
	producerTypes map[string]ir.PortType,
	bindings ir.SymbolMap,
) ([]ir.PortType, []EdgeError) {
	bound := make([]ir.PortType, len(node.Inputs))
	var edgeErrors []EdgeError

	for slotIndex, inputName := range node.Inputs {
		producerType, exists := producerTypes[inputName]

		if !exists {
			// Producer not seen yet — graph isn't topologically clean;
			// treat as a permissive placeholder so downstream stages
			// still get a value to look at.
			producerType = anyTensor()
		}

		expected := expectedInputType(spec, slotIndex, producerType)
		expected = adoptProducerShapeWhenWildcard(expected, producerType)

		result, err := ir.Unify(producerType, expected)

		if err != nil {
			edgeErrors = append(edgeErrors, edgeErrorFromUnify(node, slotIndex, inputName, producerType, expected, err))
			bound[slotIndex] = producerType
			continue
		}

		bound[slotIndex] = result.Unified

		for symbol, value := range result.Bindings {
			bindings[symbol] = value
		}
	}

	return bound, edgeErrors
}

/*
adoptProducerShapeWhenWildcard implements the typer's "rank-agnostic"
opt-out: when an OpSpec input declares its shape as the single-symbol
wildcard [N] (the shape anyTensor() returns), the typer treats it as
"any shape" and adopts the producer's actual shape before unifying.
This lets one OpSpec describe elementwise ops that operate on tensors
of any rank — math.add over [B, T, D] is still elementwise even though
the spec writes the input as [N].

The rule does not loosen dtype, layout, or kind checks. Only shape is
adopted.
*/
func adoptProducerShapeWhenWildcard(expected, producer ir.PortType) ir.PortType {
	if !isWildcardShape(expected.ShapeSchema) {
		return expected
	}

	expected.ShapeSchema = producer.ShapeSchema

	return expected
}

func isWildcardShape(shape ir.ShapeSchema) bool {
	if len(shape.Dimensions) != 1 {
		return false
	}

	return shape.Dimensions[0].IsSymbolic() && shape.Dimensions[0].Symbol == "N"
}

func expectedInputType(spec OpSpec, slotIndex int, fallback ir.PortType) ir.PortType {
	if slotIndex < len(spec.Inputs) {
		return spec.Inputs[slotIndex]
	}

	// Op accepts more inputs than the spec enumerates (e.g. variadic
	// ops). Fall back to the producer's type so the edge still types.
	return fallback
}

func edgeErrorFromUnify(
	node *ast.GraphNode,
	slotIndex int,
	producerName string,
	producerType, expectedType ir.PortType,
	err error,
) EdgeError {
	hint := ""
	reason := err.Error()

	if unificationErr, ok := err.(*ir.UnificationError); ok {
		hint = unificationErr.AdaptorHint
		reason = unificationErr.Reason
	}

	return EdgeError{
		Producer:     producerName,
		Consumer:     node.ID,
		ConsumerSlot: slotIndex,
		ProducerType: producerType,
		ExpectedType: expectedType,
		Reason:       reason,
		AdaptorHint:  hint,
	}
}

func deriveOutputType(
	node *ast.GraphNode,
	spec OpSpec,
	boundInputs []ir.PortType,
	bindings ir.SymbolMap,
) (ir.PortType, error) {
	if spec.OutputDeriver != nil {
		return spec.OutputDeriver(node, boundInputs, bindings)
	}

	return spec.Output, nil
}

func bestEffortPassthrough(node *ast.GraphNode, producerTypes map[string]ir.PortType) ir.PortType {
	for _, inputName := range node.Inputs {
		if producerType, ok := producerTypes[inputName]; ok {
			return producerType
		}
	}

	return anyTensor()
}
