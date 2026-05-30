package llvm

import (
	"github.com/theapemachine/manifesto/optimizer"
)

/*
fusionVectorizable reports whether the fusion AST can lower to explicit SIMD
vector IR. Transcendental activations and libm calls stay on the scalar loop.
*/
func fusionVectorizable(root *optimizer.ASTNode) bool {
	if root == nil {
		return false
	}

	switch root.Type {
	case optimizer.NodeSqrt, optimizer.NodeExp, optimizer.NodeLog,
		optimizer.NodeSigmoid, optimizer.NodeTanh, optimizer.NodeSilu, optimizer.NodeGelu:
		return false
	case optimizer.NodeInput, optimizer.NodeConstant:
		return true
	case optimizer.NodeAdd, optimizer.NodeSub, optimizer.NodeMul, optimizer.NodeDiv,
		optimizer.NodeMax, optimizer.NodeMin, optimizer.NodeNeg, optimizer.NodeAbs,
		optimizer.NodeReLU, optimizer.NodeLeakyReLU:
		return fusionChildrenVectorizable(root.Children)
	default:
		return false
	}
}

func fusionChildrenVectorizable(children []*optimizer.ASTNode) bool {
	for _, child := range children {
		if !fusionVectorizable(child) {
			return false
		}
	}

	return true
}
