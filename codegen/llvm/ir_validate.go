package llvm

import (
	"fmt"

	"github.com/theapemachine/manifesto/optimizer"
)

func validateFusionAST(fusion *optimizer.FusionAST) error {
	if fusion.Root == nil {
		return fmt.Errorf("codegen llvm: fusion root is required")
	}

	return validateFusionNode(fusion.Root, len(fusion.InputPorts))
}

func validateFusionNode(node *optimizer.ASTNode, inputCount int) error {
	if node == nil {
		return fmt.Errorf("codegen llvm: nil fusion node")
	}

	switch node.Type {
	case optimizer.NodeInput:
		if node.InputIndex < 0 || node.InputIndex >= inputCount {
			return fmt.Errorf(
				"codegen llvm: input index %d out of range for %d ports",
				node.InputIndex, inputCount,
			)
		}

		return nil
	case optimizer.NodeConstant:
		return nil
	case optimizer.NodeAdd, optimizer.NodeSub, optimizer.NodeMul, optimizer.NodeDiv,
		optimizer.NodeMax, optimizer.NodeMin:
		return validateBinaryNode(node, inputCount)
	case optimizer.NodeNeg, optimizer.NodeAbs, optimizer.NodeSqrt, optimizer.NodeExp,
		optimizer.NodeLog, optimizer.NodeReLU, optimizer.NodeSigmoid, optimizer.NodeTanh,
		optimizer.NodeSilu, optimizer.NodeGelu, optimizer.NodeLeakyReLU:
		return validateUnaryNode(node, inputCount)
	default:
		return fmt.Errorf("codegen llvm: unsupported fusion node %q", node.Type.String())
	}
}

func validateUnaryNode(node *optimizer.ASTNode, inputCount int) error {
	if len(node.Children) != 1 {
		return fmt.Errorf("codegen llvm: unary node %q expects 1 child", node.Type.String())
	}

	return validateFusionNode(node.Children[0], inputCount)
}

func validateBinaryNode(node *optimizer.ASTNode, inputCount int) error {
	if len(node.Children) != 2 {
		return fmt.Errorf("codegen llvm: binary node %q expects 2 children", node.Type.String())
	}

	if err := validateFusionNode(node.Children[0], inputCount); err != nil {
		return err
	}

	return validateFusionNode(node.Children[1], inputCount)
}
