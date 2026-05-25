package optimizer

import (
	"fmt"
	"math"

	"github.com/theapemachine/manifesto/ast"
)

/*
ConstantAttributeValue is the ast.GraphNode.Attributes key under which a
constant node's scalar value lives. Hosts that generate a math.constant
node (or the constant-folder itself) write float64 here.
*/
const ConstantAttributeValue = "value"

/*
constantOp is the synthetic op string the folder emits when collapsing a
chain of elementwise nodes whose inputs are all constants.
*/
const constantOp = "math.constant"

/*
ConstantFoldStats summarizes one constant-fold pass.
*/
type ConstantFoldStats struct {
	NodesFolded int
}

/*
ConstantFold walks the graph and evaluates any elementwise node whose inputs
are all constants. The result replaces the original node as a new
math.constant. Downstream consumers are rewired automatically because the
replacement keeps the original node ID.

This pass only handles scalar constants. Tensor-shaped constants (e.g.
pre-computed lookup tables) require materializing a buffer and are left
to a follow-up pass that runs after the static memory planner.
*/
func ConstantFold(graph *ast.Graph) (ConstantFoldStats, error) {
	if graph == nil {
		return ConstantFoldStats{}, fmt.Errorf("optimizer: graph is required")
	}

	stats := ConstantFoldStats{}

	// Build a map of producer ID → scalar value for nodes the folder
	// considers constant. The map grows as the pass evaluates more
	// expressions.
	values := make(map[string]float64)

	for _, node := range graph.Nodes {
		if node.Op != constantOp {
			continue
		}

		value, ok := constantValueFrom(node)

		if !ok {
			continue
		}

		values[node.ID] = value
	}

	for _, node := range graph.Nodes {
		if node.Op == constantOp {
			continue
		}

		nodeType, eligible := isElementwiseOp(node.Op)

		if !eligible {
			continue
		}

		operands := make([]float64, 0, len(node.Inputs))
		allConstant := true

		for _, inputName := range node.Inputs {
			value, ok := values[inputName]

			if !ok {
				allConstant = false
				break
			}

			operands = append(operands, value)
		}

		if !allConstant {
			continue
		}

		folded, ok := evaluateNodeType(nodeType, operands)

		if !ok {
			continue
		}

		node.Op = constantOp
		node.Inputs = nil

		if node.Attributes == nil {
			node.Attributes = make(map[string]any)
		}

		node.Attributes[ConstantAttributeValue] = folded
		values[node.ID] = folded
		stats.NodesFolded++
	}

	return stats, nil
}

func constantValueFrom(node *ast.GraphNode) (float64, bool) {
	if node == nil || node.Attributes == nil {
		return 0, false
	}

	raw, ok := node.Attributes[ConstantAttributeValue]

	if !ok {
		return 0, false
	}

	switch typed := raw.(type) {
	case float64:
		return typed, true
	case float32:
		return float64(typed), true
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	default:
		return 0, false
	}
}

/*
evaluateNodeType evaluates one node-type expression over scalar operands.
Returns (result, true) when the operation is foldable and the operand
arity matches; otherwise (0, false).
*/
func evaluateNodeType(nodeType NodeType, operands []float64) (float64, bool) {
	switch nodeType.Arity() {
	case 1:
		if len(operands) != 1 {
			return 0, false
		}

		return evaluateUnary(nodeType, operands[0])
	case 2:
		if len(operands) != 2 {
			return 0, false
		}

		return evaluateBinary(nodeType, operands[0], operands[1])
	default:
		return 0, false
	}
}

func evaluateUnary(nodeType NodeType, operand float64) (float64, bool) {
	switch nodeType {
	case NodeNeg:
		return -operand, true
	case NodeAbs:
		return math.Abs(operand), true
	case NodeSqrt:
		return math.Sqrt(operand), true
	case NodeExp:
		return math.Exp(operand), true
	case NodeLog:
		return math.Log(operand), true
	case NodeReLU:
		if operand > 0 {
			return operand, true
		}

		return 0, true
	case NodeSigmoid:
		return 1.0 / (1.0 + math.Exp(-operand)), true
	case NodeTanh:
		return math.Tanh(operand), true
	case NodeSilu:
		return operand / (1.0 + math.Exp(-operand)), true
	case NodeGelu:
		// Tanh approximation matches activation/gated_packed.go default.
		inner := math.Sqrt(2.0/math.Pi) * (operand + 0.044715*operand*operand*operand)
		return 0.5 * operand * (1.0 + math.Tanh(inner)), true
	case NodeLeakyReLU:
		if operand >= 0 {
			return operand, true
		}

		return 0.01 * operand, true
	default:
		return 0, false
	}
}

func evaluateBinary(nodeType NodeType, left, right float64) (float64, bool) {
	switch nodeType {
	case NodeAdd:
		return left + right, true
	case NodeSub:
		return left - right, true
	case NodeMul:
		return left * right, true
	case NodeDiv:
		if right == 0 {
			return 0, false
		}

		return left / right, true
	case NodeMax:
		return math.Max(left, right), true
	case NodeMin:
		return math.Min(left, right), true
	default:
		return 0, false
	}
}
