package optimizer

import "strings"

/*
elementwiseTable maps ast.GraphNode.Op strings to the AST node type they
become when absorbed into a FusionAST.

Operations not in this table are never fused. The list is intentionally
conservative — adding a new op requires confirming it has identical input
and output shapes, no parameters (other than scalar constants the
constant-fold pass already absorbs), and a deterministic per-element
formula.
*/
var elementwiseTable = map[string]NodeType{
	"math.add":  NodeAdd,
	"math.sub":  NodeSub,
	"math.mul":  NodeMul,
	"math.div":  NodeDiv,
	"math.max":  NodeMax,
	"math.min":  NodeMin,
	"math.neg":  NodeNeg,
	"math.abs":  NodeAbs,
	"math.sqrt": NodeSqrt,
	"math.exp":  NodeExp,
	"math.log":  NodeLog,
	"math.sign": NodeNeg, // sign-of-x via x/abs(x); placeholder mapping.
	"math.sin":  NodeInvalid,
	"math.cos":  NodeInvalid,

	"activation.relu":       NodeReLU,
	"activation.sigmoid":    NodeSigmoid,
	"activation.tanh":       NodeTanh,
	"activation.swish":      NodeSilu, // swish ≡ silu
	"activation.gelu":       NodeGelu,
	"activation.leaky_relu": NodeLeakyReLU,
}

/*
isElementwiseOp reports whether the given Op string maps to a fusion-eligible
elementwise primitive. Disabled mappings (NodeInvalid) are treated as
non-elementwise to keep the table grep-friendly without enabling premature
fusion paths.
*/
func isElementwiseOp(op string) (NodeType, bool) {
	normalized := strings.ToLower(strings.TrimSpace(op))

	nodeType, ok := elementwiseTable[normalized]

	if !ok || nodeType == NodeInvalid {
		return NodeInvalid, false
	}

	return nodeType, true
}

/*
ElementwiseOps returns a snapshot of every Op string the optimizer treats as
fusion-eligible. Useful for diagnostics, tests, and the future op coverage
audit in §3 of GAPS.md.
*/
func ElementwiseOps() []string {
	out := make([]string, 0, len(elementwiseTable))

	for op, nodeType := range elementwiseTable {
		if nodeType == NodeInvalid {
			continue
		}

		out = append(out, op)
	}

	return out
}
