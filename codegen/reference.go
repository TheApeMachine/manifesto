package codegen

import (
	"math"

	"github.com/theapemachine/manifesto/optimizer"
)

/*
EvalScalarReference evaluates one FusionAST root at elementIndex using
the scalar float32 reference semantics shared by CPUKernel and LLVM JIT
parity tests.
*/
func EvalScalarReference(
	root *optimizer.ASTNode,
	inputs [][]float32,
	elementIndex int,
) float32 {
	return evalScalarReferenceNode(root, inputs, elementIndex)
}

func evalScalarReferenceNode(
	node *optimizer.ASTNode,
	inputs [][]float32,
	elementIndex int,
) float32 {
	switch node.Type {
	case optimizer.NodeInput:
		return inputs[node.InputIndex][elementIndex]
	case optimizer.NodeConstant:
		return float32(node.Value)
	case optimizer.NodeAdd:
		return evalScalarReferenceNode(node.Children[0], inputs, elementIndex) +
			evalScalarReferenceNode(node.Children[1], inputs, elementIndex)
	case optimizer.NodeSub:
		return evalScalarReferenceNode(node.Children[0], inputs, elementIndex) -
			evalScalarReferenceNode(node.Children[1], inputs, elementIndex)
	case optimizer.NodeMul:
		return evalScalarReferenceNode(node.Children[0], inputs, elementIndex) *
			evalScalarReferenceNode(node.Children[1], inputs, elementIndex)
	case optimizer.NodeDiv:
		return evalScalarReferenceNode(node.Children[0], inputs, elementIndex) /
			evalScalarReferenceNode(node.Children[1], inputs, elementIndex)
	case optimizer.NodeMax:
		left := evalScalarReferenceNode(node.Children[0], inputs, elementIndex)
		right := evalScalarReferenceNode(node.Children[1], inputs, elementIndex)

		if left > right {
			return left
		}

		return right
	case optimizer.NodeMin:
		left := evalScalarReferenceNode(node.Children[0], inputs, elementIndex)
		right := evalScalarReferenceNode(node.Children[1], inputs, elementIndex)

		if left < right {
			return left
		}

		return right
	case optimizer.NodeNeg:
		return -evalScalarReferenceNode(node.Children[0], inputs, elementIndex)
	case optimizer.NodeAbs:
		value := evalScalarReferenceNode(node.Children[0], inputs, elementIndex)

		if value < 0 {
			return -value
		}

		return value
	case optimizer.NodeSqrt:
		return float32(math.Sqrt(float64(evalScalarReferenceNode(node.Children[0], inputs, elementIndex))))
	case optimizer.NodeExp:
		return float32(math.Exp(float64(evalScalarReferenceNode(node.Children[0], inputs, elementIndex))))
	case optimizer.NodeLog:
		return float32(math.Log(float64(evalScalarReferenceNode(node.Children[0], inputs, elementIndex))))
	case optimizer.NodeReLU:
		value := evalScalarReferenceNode(node.Children[0], inputs, elementIndex)

		if value < 0 {
			return 0
		}

		return value
	case optimizer.NodeSigmoid:
		value := evalScalarReferenceNode(node.Children[0], inputs, elementIndex)

		return float32(1.0 / (1.0 + math.Exp(-float64(value))))
	case optimizer.NodeTanh:
		return float32(math.Tanh(float64(evalScalarReferenceNode(node.Children[0], inputs, elementIndex))))
	case optimizer.NodeSilu:
		value := evalScalarReferenceNode(node.Children[0], inputs, elementIndex)

		return float32(float64(value) / (1.0 + math.Exp(-float64(value))))
	case optimizer.NodeGelu:
		value := float64(evalScalarReferenceNode(node.Children[0], inputs, elementIndex))
		inner := math.Sqrt(2.0/math.Pi) * (value + 0.044715*value*value*value)

		return float32(0.5 * value * (1.0 + math.Tanh(inner)))
	case optimizer.NodeLeakyReLU:
		value := evalScalarReferenceNode(node.Children[0], inputs, elementIndex)

		if value >= 0 {
			return value
		}

		return 0.01 * value
	default:
		return 0
	}
}
