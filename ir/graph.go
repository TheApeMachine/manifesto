package ir

import (
	"fmt"
)

/*
Graph represents a collection of interconnected computational nodes.
It serves as the intermediate representation of the execution flow.
*/
type Graph struct {
	nodes []*Node
}

type Index struct {
	nodes map[string]*Node
	users map[string][]*Node
}

/*
NewGraph instantiates a new Graph.
It is created to abstract physical execution from the mathematical intent.
*/
func NewGraph() *Graph {
	return &Graph{
		nodes: make([]*Node, 0),
	}
}

/*
Nodes returns a defensive copy of all nodes in the graph.
*/
func (graph *Graph) Nodes() []*Node {
	out := make([]*Node, len(graph.nodes))
	copy(out, graph.nodes)
	return out
}

/*
AddNode registers a node in the computational graph.
*/
func (graph *Graph) AddNode(node *Node) {
	if node == nil {
		return
	}
	graph.nodes = append(graph.nodes, node)
}

func (graph *Graph) Index() (*Index, error) {
	nodes := graph.Nodes()
	index := &Index{
		nodes: make(map[string]*Node, len(nodes)),
		users: make(map[string][]*Node, len(nodes)),
	}

	for _, node := range nodes {
		if node.ID() == "" {
			return nil, fmt.Errorf("graph: node ID is required")
		}

		if _, ok := index.nodes[node.ID()]; ok {
			return nil, fmt.Errorf("graph: duplicate node %q", node.ID())
		}

		index.nodes[node.ID()] = node
	}

	for _, node := range nodes {
		for _, input := range node.Inputs() {
			if _, ok := index.nodes[input.ID()]; !ok {
				return nil, fmt.Errorf(
					"graph: node %q has unregistered input %q",
					node.ID(),
					input.ID(),
				)
			}

			index.users[input.ID()] = append(index.users[input.ID()], node)
		}
	}

	return index, nil
}

func (index *Index) Node(id string) *Node {
	if index == nil {
		return nil
	}

	return index.nodes[id]
}

func (index *Index) Users(id string) []*Node {
	if index == nil {
		return nil
	}

	users := index.users[id]
	out := make([]*Node, len(users))
	copy(out, users)

	return out
}

func (graph *Graph) Verify() error {
	if _, err := graph.Index(); err != nil {
		return err
	}

	if _, err := graph.TopologyLayers(); err != nil {
		return err
	}

	return graph.verifyShapes()
}

func (graph *Graph) verifyShapes() error {
	for _, node := range graph.Nodes() {
		if err := verifyNodeShape(node); err != nil {
			return err
		}
	}

	return nil
}

func verifyNodeShape(node *Node) error {
	if node == nil {
		return fmt.Errorf("graph: nil node")
	}

	if !node.Shape().Valid() {
		return fmt.Errorf("graph: node %q has invalid shape", node.ID())
	}

	inputs := node.Inputs()

	switch node.OpType() {
	case OpInput:
		return verifyInputNode(node, inputs)
	case OpAdd, OpMul:
		return verifyElementwiseNode(node, inputs)
	case OpMatmul:
		return verifyMatmulNode(node, inputs)
	case OpReLU, OpLeakyReLU, OpGELU, OpTanh, OpSigmoid, OpSwish, OpSELU:
		return verifyUnarySameShapeNode(node, inputs)
	case OpSwiGLU:
		return verifySwiGLUNode(node, inputs)
	case OpFused:
		return verifyFusedNode(node, inputs)
	default:
		return nil
	}
}

func verifyInputNode(node *Node, inputs []*Node) error {
	if len(inputs) != 0 {
		return fmt.Errorf("graph: input node %q requires 0 inputs, got %d", node.ID(), len(inputs))
	}

	return nil
}

func verifyElementwiseNode(node *Node, inputs []*Node) error {
	if err := verifyInputCount(node, inputs, 2); err != nil {
		return err
	}

	shape := node.Shape()

	for _, input := range inputs {
		if !input.Shape().Equal(shape) {
			return fmt.Errorf(
				"graph: %s node %q shape %v is incompatible with input %q shape %v",
				node.OpType(), node.ID(), shape.Dims(), input.ID(), input.Shape().Dims(),
			)
		}
	}

	return nil
}

func verifyMatmulNode(node *Node, inputs []*Node) error {
	if err := verifyInputCount(node, inputs, 2); err != nil {
		return err
	}

	leftDims := inputs[0].Shape().Dims()
	rightDims := inputs[1].Shape().Dims()
	outputDims := node.Shape().Dims()

	if len(leftDims) != 2 || len(rightDims) != 2 || len(outputDims) != 2 {
		return fmt.Errorf("graph: Matmul node %q requires rank-2 input and output shapes", node.ID())
	}

	if leftDims[1] != rightDims[0] {
		return fmt.Errorf(
			"graph: Matmul node %q input shapes %v and %v have incompatible inner dimensions",
			node.ID(), leftDims, rightDims,
		)
	}

	if outputDims[0] != leftDims[0] || outputDims[1] != rightDims[1] {
		return fmt.Errorf(
			"graph: Matmul node %q output shape %v must equal [%d %d]",
			node.ID(), outputDims, leftDims[0], rightDims[1],
		)
	}

	return nil
}

func verifyUnarySameShapeNode(node *Node, inputs []*Node) error {
	if err := verifyInputCount(node, inputs, 1); err != nil {
		return err
	}

	if !inputs[0].Shape().Equal(node.Shape()) {
		return fmt.Errorf(
			"graph: %s node %q shape %v must equal input shape %v",
			node.OpType(), node.ID(), node.Shape().Dims(), inputs[0].Shape().Dims(),
		)
	}

	return nil
}

func verifySwiGLUNode(node *Node, inputs []*Node) error {
	if err := verifyInputCount(node, inputs, 1); err != nil {
		return err
	}

	inputDims := inputs[0].Shape().Dims()
	outputDims := node.Shape().Dims()

	if len(inputDims) == 0 {
		if len(outputDims) != 1 || outputDims[0] != inputs[0].Shape().Len()/2 {
			return fmt.Errorf("graph: SwiGLU node %q output shape %v is incompatible with scalar input", node.ID(), outputDims)
		}

		return nil
	}

	expected := append([]int(nil), inputDims...)
	lastIndex := len(expected) - 1

	if expected[lastIndex]%2 != 0 {
		return fmt.Errorf("graph: SwiGLU node %q input final dimension must be even", node.ID())
	}

	expected[lastIndex] /= 2

	if !sameDims(outputDims, expected) {
		return fmt.Errorf(
			"graph: SwiGLU node %q output shape %v must equal %v",
			node.ID(), outputDims, expected,
		)
	}

	return nil
}

func verifyFusedNode(node *Node, inputs []*Node) error {
	if len(inputs) != 2 && len(inputs) != 3 {
		return fmt.Errorf("graph: Fused node %q requires 2 or 3 inputs, got %d", node.ID(), len(inputs))
	}

	leftDims := inputs[0].Shape().Dims()
	rightDims := inputs[1].Shape().Dims()
	outputDims := node.Shape().Dims()

	if len(leftDims) != 2 || len(rightDims) != 2 || len(outputDims) != 2 {
		return fmt.Errorf("graph: Fused node %q requires rank-2 matmul shapes", node.ID())
	}

	if leftDims[1] != rightDims[0] || outputDims[0] != leftDims[0] || outputDims[1] != rightDims[1] {
		return fmt.Errorf("graph: Fused node %q has incompatible matmul shapes", node.ID())
	}

	if len(inputs) == 2 {
		return nil
	}

	biasLen := inputs[2].Shape().Len()
	if biasLen != outputDims[1] && biasLen != outputDims[0]*outputDims[1] {
		return fmt.Errorf(
			"graph: Fused node %q bias length %d must equal N=%d or M*N=%d",
			node.ID(), biasLen, outputDims[1], outputDims[0]*outputDims[1],
		)
	}

	return nil
}

func verifyInputCount(node *Node, inputs []*Node, expected int) error {
	if len(inputs) != expected {
		return fmt.Errorf(
			"graph: %s node %q requires %d inputs, got %d",
			node.OpType(), node.ID(), expected, len(inputs),
		)
	}

	return nil
}

func sameDims(left, right []int) bool {
	if len(left) != len(right) {
		return false
	}

	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}

	return true
}

func (graph *Graph) Clone() (*Graph, map[string]*Node, error) {
	if err := graph.Verify(); err != nil {
		return nil, nil, err
	}

	clone := NewGraph()
	replacements := make(map[string]*Node, len(graph.nodes))

	for _, node := range graph.nodes {
		newNode := cloneNode(node)
		replacements[node.ID()] = newNode
		clone.AddNode(newNode)
	}

	for _, node := range graph.nodes {
		newNode := replacements[node.ID()]

		for _, input := range node.Inputs() {
			newInput, ok := replacements[input.ID()]
			if !ok {
				return nil, nil, fmt.Errorf("graph: missing clone input %q", input.ID())
			}

			newNode.AddInput(newInput)
		}
	}

	return clone, replacements, nil
}

func cloneNode(node *Node) *Node {
	newNode := NewNode(node.ID(), node.OpType(), node.Shape())
	newNode.SetOperationID(node.OperationID())
	newNode.SetValueType(node.ValueType())
	newNode.SetEffect(node.Effect())
	newNode.SetAlias(node.Alias())
	newNode.SetInPlace(node.InPlace())

	for key, value := range node.Metadata() {
		newNode.SetMetadata(key, value)
	}

	for key, value := range node.Attributes() {
		newNode.SetAttribute(key, value)
	}

	return newNode
}

/*
Sinks returns nodes that have no dependents (i.e. output nodes).
*/
func (graph *Graph) Sinks() []*Node {
	hasDependent := make(map[string]bool)

	for _, node := range graph.nodes {
		for _, input := range node.Inputs() {
			hasDependent[input.ID()] = true
		}
	}

	var sinks []*Node

	for _, node := range graph.nodes {
		if !hasDependent[node.ID()] {
			sinks = append(sinks, node)
		}
	}

	return sinks
}

/*
TopologyLayers groups nodes into sequential execution layers.
Nodes in the same layer are completely independent of each other and can be
executed concurrently across multiple streams or command queues.
*/
func (graph *Graph) TopologyLayers() ([][]*Node, error) {
	layers := make([][]*Node, 0)

	type irNodeInfo struct {
		node       *Node
		dependents []*Node
	}

	inDegree := make(map[string]int)
	nodeMap := make(map[string]*irNodeInfo)

	for _, n := range graph.nodes {
		nodeMap[n.ID()] = &irNodeInfo{node: n, dependents: make([]*Node, 0)}
	}

	for _, n := range graph.nodes {
		inDegree[n.ID()] = len(n.Inputs())
		for _, in := range n.Inputs() {
			if info, ok := nodeMap[in.ID()]; ok {
				info.dependents = append(info.dependents, n)
			}
		}
	}

	var currentLayer []*Node
	for _, n := range graph.nodes {
		if inDegree[n.ID()] == 0 {
			currentLayer = append(currentLayer, n)
		}
	}

	processedCount := 0
	for len(currentLayer) > 0 {
		layers = append(layers, currentLayer)
		processedCount += len(currentLayer)
		var nextLayer []*Node

		for _, n := range currentLayer {
			for _, dep := range nodeMap[n.ID()].dependents {
				inDegree[dep.ID()]--
				if inDegree[dep.ID()] == 0 {
					nextLayer = append(nextLayer, dep)
				}
			}
		}

		currentLayer = nextLayer
	}

	if processedCount != len(graph.nodes) {
		return nil, fmt.Errorf("cycle detected in graph")
	}

	return layers, nil
}
