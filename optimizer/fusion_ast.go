/*
Package optimizer implements the manifesto compiler optimizer stage: the
pass that sits between front-end lowering (`compiler/topology_lower.go`) and
codegen (`manifesto/codegen`, future). It performs algebraic rewrites,
clusters contiguous elementwise nodes into FusionAST subgraphs, and
attaches cache-tiling annotations to heavy ops (Matmul, Conv2D).

The optimizer never executes — it only rewrites the ast.Graph it is given.
Mutations are observable so the next stage (codegen) sees a structurally
distinct graph with FusionAST nodes in place of clustered originals.

Per ARCHITECTURE.md §4.3 ("Elementwise Fusion & JIT Compilation"), the
FusionAST is the data structure all JIT codegen lowers from. CPU LLVM, MSL,
PTX, and HLO emitters all walk this same AST.
*/
package optimizer

import (
	"github.com/theapemachine/manifesto/dtype"
)

/*
NodeType identifies one algebraic operation inside a FusionAST.

The list is intentionally small — only ops that the fusion clusterer can
reason about as scalar transforms appear here. Heavy ops (Matmul,
Convolution, Attention) never enter the AST; they remain on the parent
ast.Graph as opaque calls.
*/
type NodeType int

const (
	// NodeInvalid is the zero value; reaching it is a bug.
	NodeInvalid NodeType = iota

	// NodeInput is a leaf referencing one input tensor by index into
	// FusionAST.InputPorts.
	NodeInput

	// NodeConstant is a leaf holding one scalar literal.
	NodeConstant

	// Two-operand arithmetic.
	NodeAdd
	NodeSub
	NodeMul
	NodeDiv
	NodeMax
	NodeMin

	// One-operand math primitives.
	NodeNeg
	NodeAbs
	NodeSqrt
	NodeExp
	NodeLog

	// One-operand activations.
	NodeReLU
	NodeSigmoid
	NodeTanh
	NodeSilu
	NodeGelu
	NodeLeakyReLU
)

/*
ASTNode is one algebraic operation in a fused loop.
*/
type ASTNode struct {
	Type       NodeType
	Value      float64    // Used if Type == NodeConstant.
	InputIndex int        // Maps to FusionAST.InputPorts if Type == NodeInput.
	Children   []*ASTNode // Operands. Length determined by Type's arity.
	DType      dtype.DType
}

/*
FusionAST represents a consolidated elementwise mathematical expression
spanning one or more originally-separate ast.GraphNodes.

InputPorts and OutputPort are workspace offsets (or, before the static
memory planner lands, ast.Graph value names). CountExpr is the dynamic
symbol expression representing element count (e.g. "B*T*D").
*/
type FusionAST struct {
	// Root is the top-level expression node; its evaluation produces the
	// FusionAST output value.
	Root *ASTNode

	// InputPorts lists the names of every value-producing input the fused
	// expression consumes, in the same order ASTNode.InputIndex addresses
	// them.
	InputPorts []string

	// OutputPort is the name under which the fused result is written.
	OutputPort string

	// CountExpr is the symbolic element count (e.g. "B*T*D"). Empty when
	// the count is not yet known; the codegen stage resolves it against
	// the runtime SymbolMap.
	CountExpr string

	// DType is the activation dtype used inside the fused loop. Inputs at
	// other dtypes must be cast before entering the fusion (the adaptor-
	// synthesis pass is responsible for this).
	DType dtype.DType

	// ContainedNodeIDs lists every original ast.GraphNode whose work was
	// absorbed into this fusion. Used by the rewriter to remove the
	// originals from the parent graph and by debug tooling.
	ContainedNodeIDs []string
}

/*
NewInputNode constructs an input leaf.
*/
func NewInputNode(inputIndex int, dataType dtype.DType) *ASTNode {
	return &ASTNode{
		Type:       NodeInput,
		InputIndex: inputIndex,
		DType:      dataType,
	}
}

/*
NewConstantNode constructs a scalar literal leaf.
*/
func NewConstantNode(value float64, dataType dtype.DType) *ASTNode {
	return &ASTNode{
		Type:  NodeConstant,
		Value: value,
		DType: dataType,
	}
}

/*
NewUnaryNode constructs a one-child AST node (math primitive or activation).
*/
func NewUnaryNode(nodeType NodeType, child *ASTNode) *ASTNode {
	return &ASTNode{
		Type:     nodeType,
		Children: []*ASTNode{child},
		DType:    child.DType,
	}
}

/*
NewBinaryNode constructs a two-child AST node.
*/
func NewBinaryNode(nodeType NodeType, left, right *ASTNode) *ASTNode {
	dataType := left.DType

	if dataType == dtype.Invalid {
		dataType = right.DType
	}

	return &ASTNode{
		Type:     nodeType,
		Children: []*ASTNode{left, right},
		DType:    dataType,
	}
}

/*
Arity returns the number of children a node type expects.
*/
func (nodeType NodeType) Arity() int {
	switch nodeType {
	case NodeInput, NodeConstant:
		return 0
	case NodeNeg, NodeAbs, NodeSqrt, NodeExp, NodeLog,
		NodeReLU, NodeSigmoid, NodeTanh, NodeSilu, NodeGelu, NodeLeakyReLU:
		return 1
	case NodeAdd, NodeSub, NodeMul, NodeDiv, NodeMax, NodeMin:
		return 2
	default:
		return -1
	}
}

/*
String returns a stable name for one node type, used by debug dumps and
codegen.
*/
func (nodeType NodeType) String() string {
	switch nodeType {
	case NodeInput:
		return "Input"
	case NodeConstant:
		return "Constant"
	case NodeAdd:
		return "Add"
	case NodeSub:
		return "Sub"
	case NodeMul:
		return "Mul"
	case NodeDiv:
		return "Div"
	case NodeMax:
		return "Max"
	case NodeMin:
		return "Min"
	case NodeNeg:
		return "Neg"
	case NodeAbs:
		return "Abs"
	case NodeSqrt:
		return "Sqrt"
	case NodeExp:
		return "Exp"
	case NodeLog:
		return "Log"
	case NodeReLU:
		return "ReLU"
	case NodeSigmoid:
		return "Sigmoid"
	case NodeTanh:
		return "Tanh"
	case NodeSilu:
		return "Silu"
	case NodeGelu:
		return "Gelu"
	case NodeLeakyReLU:
		return "LeakyReLU"
	default:
		return "Invalid"
	}
}
