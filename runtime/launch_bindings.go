package runtime

import (
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/ir"
)

/*
DeriveLaunchBindings resolves dynamic shape symbols for one graph.call
from the caller's concrete inputs.

The static memory planner sizes workspace regions against upper-bound
SymbolMap entries (max sequence length, max batch, …). At launch time
ARCHITECTURE.md requires those symbols be bound to the live values the
current invocation actually uses so kernels touch only the active prefix
and shape intrinsics like shape.last_token select the real final row.

Token-axis length is inferred from the first graph input that carries
integer token IDs ([]int or []int32). When the graph declares boundary
inputs, those names are consulted first so unrelated scalar inputs do
not win the scan order.

KV is the total cached sequence length after the current step:
position_offset + len(input_ids).
*/
func DeriveLaunchBindings(graph *ast.Graph, inputs map[string]any) ir.SymbolMap {
	sequenceLength := int64(0)

	if graph != nil {
		for _, inputName := range graph.Inputs {
			length, ok := tokenSequenceLength(inputs[inputName])

			if ok {
				sequenceLength = length

				break
			}
		}
	}

	if sequenceLength == 0 {
		for _, value := range inputs {
			length, ok := tokenSequenceLength(value)

			if ok {
				sequenceLength = length

				break
			}
		}
	}

	if sequenceLength == 0 {
		return nil
	}

	positionOffset, err := scalarInt32Value(inputs["position_offset"])

	if err != nil {
		positionOffset = 0
	}

	kvLength := int64(positionOffset) + sequenceLength

	return ir.SymbolMap{
		"N":  sequenceLength,
		"T":  sequenceLength,
		"KV": kvLength,
	}
}

func tokenSequenceLength(value any) (int64, bool) {
	switch typed := value.(type) {
	case []int:
		if len(typed) == 0 {
			return 0, false
		}

		return int64(len(typed)), true
	case []int32:
		if len(typed) == 0 {
			return 0, false
		}

		return int64(len(typed)), true
	case []int64:
		if len(typed) == 0 {
			return 0, false
		}

		return int64(len(typed)), true
	case int:
		return 1, true
	case int32:
		return 1, true
	case int64:
		return 1, true
	default:
		return 0, false
	}
}
