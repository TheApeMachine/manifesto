package ir

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
)

/*
FusionNodeType identifies one algebraic operation in a FusionAST. The
codegen pass walks the AST and emits the corresponding native
instructions (FADD, FSUB, FMUL, etc. on CPU SIMD; MSL/PTX equivalents
on GPU; HLO ops on XLA).

The enum is intentionally narrow: only operations the JIT codegen can
emit as a single vector instruction (or a short polynomial sequence
for transcendentals) qualify. Adding a new fusible operation requires:
  1. Adding a NodeType constant here.
  2. Adding the matching op-name mapping to fusibleOps.
  3. Teaching every codegen backend to emit the operation.
*/
type FusionNodeType int

const (
	// NodeInput pulls one of the cluster's input ports into the AST.
	// InputIndex selects which input port (0-based into
	// FusionAST.InputPorts).
	NodeInput FusionNodeType = iota
	// NodeConstant is a literal scalar value. Value carries the
	// numeric constant; codegen emits it as a vector-broadcast load.
	NodeConstant

	// Binary elementwise.
	NodeAdd
	NodeSub
	NodeMul
	NodeDiv

	// Unary elementwise math.
	NodeNeg
	NodeAbs
	NodeSqrt

	// Transcendentals fused as polynomial approximations by codegen.
	NodeExp
	NodeLog

	// Activations.
	NodeReLU
	NodeSigmoid
	NodeTanh
)

/*
String returns a human-readable name for the node type, used by tests
and diagnostic dumps of the AST. Codegen does not consume String — it
switches on the typed constant.
*/
func (nodeType FusionNodeType) String() string {
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
	default:
		return fmt.Sprintf("FusionNodeType(%d)", int(nodeType))
	}
}

/*
ASTNode is one operation in a FusionAST. The tree shape mirrors the
algebraic expression the cluster computes: NodeInput / NodeConstant
are leaves, every other NodeType has Children (1 for unary, 2 for
binary).

Defined verbatim from ARCHITECTURE.md Detailed Implementation
Blueprints §1, with two field-name clarifications:
  - Type rather than NodeType to avoid stuttering ASTNode.NodeType.
  - DType is the element precision the entire AST operates at; mixed
    precision within a cluster is not supported (the unifier would
    have inserted a Cast adaptor before the fusion boundary).
*/
type ASTNode struct {
	Type       FusionNodeType
	Value      float64
	InputIndex int
	Children   []*ASTNode
	DType      dtype.DType
}

/*
FusionAST is the consolidated representation of one cluster of
contiguous elementwise operations. The JIT codegen consumes one
FusionAST per cluster, emits a native kernel that computes Root from
the InputPorts, and writes the result to OutputPort.

Defined from ARCHITECTURE.md Blueprints §1. CountExpr is a textual
representation of the symbolic element count (e.g. "B * T * 768");
the codegen passes it as a kernel parameter, resolved at launch time
through the SymbolMap.
*/
type FusionAST struct {
	Root       *ASTNode
	InputPorts []int32
	OutputPort int32
	CountExpr  string
}

/*
fusibleOps maps device.Backend Operation enum values to the
FusionNodeType the clustering pass should emit when it encounters that
op. Operations not present in this map are treated as fusion boundaries
— the cluster ends at them.

Only operations whose math fits one of the FusionNodeType constants
appear here. Activations like SwiGLU or HardGelu are technically
elementwise but require multiple-output or polynomial expansion that
the current codegen path does not yet emit; adding them is a follow-up.
*/
var fusibleOps = map[Operation]FusionNodeType{
	// Binary elementwise.
	OperationAdd: NodeAdd,
	OperationSub: NodeSub,
	OperationMul: NodeMul,
	OperationDiv: NodeDiv,

	// Unary elementwise math.
	OperationNeg:  NodeNeg,
	OperationAbs:  NodeAbs,
	OperationSqrt: NodeSqrt,

	// Transcendentals fused as polynomial approximations by codegen.
	OperationExp: NodeExp,
	OperationLog: NodeLog,

	// Activations whose math is a single-output elementwise transform.
	OperationReLU:    NodeReLU,
	OperationSigmoid: NodeSigmoid,
	OperationTanh:    NodeTanh,
}

/*
IsFusibleElementwise reports whether an Operation participates in
elementwise fusion clusters. Used by FindFusionClusters and exposed
publicly so other compiler passes (cost models, manual fusion hints)
can ask the same question.
*/
func IsFusibleElementwise(op Operation) bool {
	_, fusible := fusibleOps[op]
	return fusible
}

/*
FusionNodeTypeForOp returns the FusionNodeType corresponding to an
Operation, or (NodeInput, false) if the op is not fusible. NodeInput
is the zero sentinel because it never appears as the result of an op
lookup — input nodes are leaves the clustering pass constructs
directly.
*/
func FusionNodeTypeForOp(op Operation) (FusionNodeType, bool) {
	nodeType, fusible := fusibleOps[op]
	return nodeType, fusible
}

/*
FindFusionClusters walks `topology.Nodes` and returns one FusionAST
per maximal contiguous chain of fusible elementwise operations.

The clustering rules:
  - A node enters a cluster only if its op is in fusibleOps.
  - A cluster can absorb a producer node only if that producer's
    output is consumed by EXACTLY this consumer (single-consumer
    constraint). Fan-out at the producer breaks the cluster because
    fusing would either duplicate computation across consumers or
    require materializing the intermediate.
  - Singleton fusible nodes (one op, no fusible predecessors with
    single consumers) still produce a FusionAST with a single
    operation root. The JIT can still benefit from launching a
    custom kernel over a vector instruction sequence rather than
    going through the static-kernel dispatch path.

Returns FusionASTs in topological order (cluster i appears before
cluster j iff i's root node precedes j's root node in topology.Nodes).
The producerOf index built here uses shared Port pointers, matching
the convention established by AnalyzeLiveness and ScheduleStreams.

Each returned FusionAST has its Root, InputPorts, OutputPort, and
CountExpr fields populated. CountExpr is left empty when the cluster's
output shape uses no symbolic dimensions; otherwise it carries the
symbolic product (e.g. "B * T * 768") for the codegen to resolve.
*/
func FindFusionClusters(topology *Topology) []FusionAST {
	if topology == nil || len(topology.Nodes) == 0 {
		return nil
	}

	producerOf := buildProducerIndex(topology)
	consumerCount := buildConsumerCounts(topology)

	visited := make(map[int]bool, len(topology.Nodes))
	var clusters []FusionAST

	// Walk nodes in REVERSE topological order. The clustering recurses
	// from a seed node BACKWARDS into its producers, absorbing them
	// into the seed's AST. If we walked forward, an early fusible node
	// (say "add" at index 0) would be processed first as a singleton
	// cluster and marked visited before its downstream consumer ("mul"
	// at index 1) ever got a chance to absorb it. Walking backward
	// gives terminal nodes the first pass.
	for nodeIndex := len(topology.Nodes) - 1; nodeIndex >= 0; nodeIndex-- {
		node := topology.Nodes[nodeIndex]

		if node == nil || visited[nodeIndex] {
			continue
		}

		if !IsFusibleElementwise(node.Operation) {
			continue
		}

		cluster := growCluster(nodeIndex, topology, producerOf, consumerCount, visited)

		if cluster.Root != nil {
			clusters = append(clusters, cluster)
		}
	}

	// Reverse so clusters are returned in topological (forward) order
	// — callers expect cluster i to precede cluster j when i's root
	// node precedes j's in topology.Nodes.
	for low, high := 0, len(clusters)-1; low < high; low, high = low+1, high-1 {
		clusters[low], clusters[high] = clusters[high], clusters[low]
	}

	return clusters
}

/*
buildConsumerCounts returns the number of times each Port pointer
appears as an input to a node in the topology. The clustering pass
uses this to enforce the single-consumer rule: a producer can be
absorbed into a cluster only when its output port has exactly one
consumer (the cluster). Otherwise fusion would require duplicating
the producer's work across multiple consumers.
*/
func buildConsumerCounts(topology *Topology) map[*Port]int {
	counts := make(map[*Port]int)

	for _, node := range topology.Nodes {
		if node == nil {
			continue
		}

		for _, port := range node.Inputs {
			if port == nil {
				continue
			}

			counts[port]++
		}
	}

	return counts
}

/*
growCluster expands a cluster starting from `seedIndex` by walking
backwards through fusible single-consumer producers, accumulating
their operations into a single FusionAST. The seed is always the
cluster's terminal node — its output port becomes the cluster's
OutputPort, and any node whose output feeds (only) into this cluster
gets absorbed.

The returned FusionAST has Root pointing to the seed's algebraic
operation, with Children that recurse into absorbed producers'
operations. Inputs that come from non-fusible nodes (or fusible
nodes with multiple consumers) terminate as NodeInput leaves;
the producer port IDs are recorded in InputPorts.

`visited` is mutated: every node absorbed into the returned cluster
(including the seed) is marked visited so subsequent iterations of
FindFusionClusters do not re-emit the same nodes as singleton
clusters.
*/
func growCluster(
	seedIndex int,
	topology *Topology,
	producerOf map[*Port]int,
	consumerCount map[*Port]int,
	visited map[int]bool,
) FusionAST {
	seed := topology.Nodes[seedIndex]

	if len(seed.Outputs) == 0 {
		return FusionAST{}
	}

	output := seed.Outputs[0]

	var inputPorts []int32

	root := buildClusterNode(
		seedIndex,
		topology,
		producerOf,
		consumerCount,
		visited,
		&inputPorts,
	)

	if root == nil {
		return FusionAST{}
	}

	return FusionAST{
		Root:       root,
		InputPorts: inputPorts,
		OutputPort: output.ID,
		CountExpr:  fusionCountExpr(output),
	}
}

/*
buildClusterNode constructs the FusionAST subtree rooted at the node
at topology.Nodes[nodeIndex], recursively absorbing fusible
single-consumer producers and terminating at non-fusible inputs
(which become NodeInput leaves, with their port IDs appended to
*inputPorts).

Returns nil if the node has no fusible operation — the caller treats
nil as "this node is a fusion boundary."
*/
func buildClusterNode(
	nodeIndex int,
	topology *Topology,
	producerOf map[*Port]int,
	consumerCount map[*Port]int,
	visited map[int]bool,
	inputPorts *[]int32,
) *ASTNode {
	node := topology.Nodes[nodeIndex]

	if node == nil {
		return nil
	}

	nodeType, fusible := FusionNodeTypeForOp(node.Operation)
	if !fusible {
		return nil
	}

	visited[nodeIndex] = true

	clusterNode := &ASTNode{
		Type: nodeType,
	}

	if len(node.Outputs) > 0 && node.Outputs[0] != nil {
		clusterNode.DType = node.Outputs[0].Type.DType
	}

	for _, inputPort := range node.Inputs {
		if inputPort == nil {
			continue
		}

		child := childForInput(
			inputPort,
			topology,
			producerOf,
			consumerCount,
			visited,
			inputPorts,
		)

		clusterNode.Children = append(clusterNode.Children, child)
	}

	return clusterNode
}

/*
childForInput decides what AST subtree should represent the value
flowing into this consumer through `inputPort`:

  - If the input is produced by a fusible node AND that producer's
    output has exactly one consumer (this cluster), recurse to absorb
    the producer into the AST.
  - Otherwise treat the input as a cluster-external value: emit a
    NodeInput leaf, record the port ID in inputPorts, and assign the
    leaf the next sequential InputIndex.
*/
func childForInput(
	inputPort *Port,
	topology *Topology,
	producerOf map[*Port]int,
	consumerCount map[*Port]int,
	visited map[int]bool,
	inputPorts *[]int32,
) *ASTNode {
	producerIndex, hasProducer := producerOf[inputPort]

	if hasProducer && !visited[producerIndex] {
		producerNode := topology.Nodes[producerIndex]

		if producerNode != nil && IsFusibleElementwise(producerNode.Operation) {
			if consumerCount[inputPort] == 1 {
				subtree := buildClusterNode(
					producerIndex,
					topology,
					producerOf,
					consumerCount,
					visited,
					inputPorts,
				)

				if subtree != nil {
					return subtree
				}
			}
		}
	}

	leaf := &ASTNode{
		Type:       NodeInput,
		InputIndex: len(*inputPorts),
		DType:      inputPort.Type.DType,
	}

	*inputPorts = append(*inputPorts, inputPort.ID)

	return leaf
}

/*
fusionCountExpr returns a textual symbolic expression for the
element count of the cluster's output shape. Empty when the shape is
fully static (codegen can hard-code the constant instead of taking it
as a launch-time parameter).

Example: a [B, T, 768] output yields "B * T * 768"; a [4, 256, 768]
output yields "" because the count 786432 is compile-time known.
*/
func fusionCountExpr(port *Port) string {
	if port == nil {
		return ""
	}

	dimensions := port.Type.ShapeSchema.Dimensions

	if len(dimensions) == 0 {
		return ""
	}

	hasSymbolic := false

	for _, dimension := range dimensions {
		if dimension.IsSymbolic() {
			hasSymbolic = true
			break
		}
	}

	if !hasSymbolic {
		return ""
	}

	expression := ""

	for index, dimension := range dimensions {
		if index > 0 {
			expression += " * "
		}

		expression += dimension.String()
	}

	return expression
}
