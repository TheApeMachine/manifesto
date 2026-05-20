package ir

import (
	"encoding/json"

	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/manifesto/dtype"
)

const graphCodecVersion = 1

type graphDocument struct {
	Version int            `json:"version"`
	Nodes   []nodeDocument `json:"nodes"`
}

type nodeDocument struct {
	ID         string               `json:"id"`
	Op         OpType               `json:"op"`
	Operation  OpID                 `json:"operation"`
	Shape      []int                `json:"shape"`
	DType      dtype.DType          `json:"dtype"`
	Precision  dtype.DType          `json:"precision"`
	Layout     Layout               `json:"layout"`
	Memory     MemoryClass          `json:"memory"`
	Effect     Effect               `json:"effect"`
	Alias      Alias                `json:"alias"`
	InPlace    bool                 `json:"inPlace"`
	Inputs     []string             `json:"inputs"`
	Attributes map[string]Attribute `json:"attributes"`
	Metadata   map[string]any       `json:"metadata"`
}

func EncodeGraph(graph *Graph) ([]byte, error) {
	if err := graph.Verify(); err != nil {
		return nil, err
	}

	document := graphDocument{
		Version: graphCodecVersion,
		Nodes:   make([]nodeDocument, 0, len(graph.Nodes())),
	}

	for _, node := range graph.Nodes() {
		inputs := node.Inputs()
		inputIDs := make([]string, len(inputs))

		for index, input := range inputs {
			inputIDs[index] = input.ID()
		}

		valueType := node.ValueType()
		document.Nodes = append(document.Nodes, nodeDocument{
			ID:         node.ID(),
			Op:         node.OpType(),
			Operation:  node.OperationID(),
			Shape:      node.Shape().Dims(),
			DType:      valueType.DType,
			Precision:  valueType.Precision,
			Layout:     valueType.Layout,
			Memory:     valueType.MemoryClass,
			Effect:     node.Effect(),
			Alias:      node.Alias(),
			InPlace:    node.InPlace(),
			Inputs:     inputIDs,
			Attributes: node.Attributes(),
			Metadata:   node.Metadata(),
		})
	}

	return json.Marshal(document)
}

func DecodeGraph(data []byte) (*Graph, error) {
	var document graphDocument

	if err := json.Unmarshal(data, &document); err != nil {
		return nil, err
	}

	graph := NewGraph()
	nodes := make(map[string]*Node, len(document.Nodes))

	for _, nodeData := range document.Nodes {
		shape, err := tensor.NewShape(nodeData.Shape)

		if err != nil {
			return nil, err
		}

		node := NewNode(nodeData.ID, nodeData.Op, shape)
		node.SetOperationID(nodeData.Operation)
		node.SetValueType(ValueType{
			Shape:       shape,
			DType:       nodeData.DType,
			Precision:   nodeData.Precision,
			Layout:      nodeData.Layout,
			MemoryClass: nodeData.Memory,
		})
		node.SetEffect(nodeData.Effect)
		node.SetAlias(nodeData.Alias)
		node.SetInPlace(nodeData.InPlace)

		for key, value := range nodeData.Attributes {
			node.SetAttribute(key, value)
		}

		for key, value := range nodeData.Metadata {
			node.SetMetadata(key, value)
		}

		nodes[node.ID()] = node
		graph.AddNode(node)
	}

	for _, nodeData := range document.Nodes {
		node := nodes[nodeData.ID]

		for _, inputID := range nodeData.Inputs {
			node.AddInput(nodes[inputID])
		}
	}

	return graph, graph.Verify()
}
