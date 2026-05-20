package weights

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
)

type safetensorsMeta struct {
	DType       string   `json:"dtype,omitempty"`
	Shape       []int64  `json:"shape,omitempty"`
	DataOffsets [2]int64 `json:"data_offsets,omitempty"`
}

/*
Binder indexes SafeTensors checkpoints and attaches tensors to graph nodes.
*/
type Binder struct{}

/*
NewBinder constructs a Binder.
*/
func NewBinder() *Binder {
	return &Binder{}
}

/*
Index reads a SafeTensors header and returns tensor metadata keyed by name.
*/
func (binder *Binder) Index(reader io.Reader) (map[string]safetensorsMeta, error) {
	var headerLength uint64

	if err := binary.Read(reader, binary.LittleEndian, &headerLength); err != nil {
		return nil, fmt.Errorf("safetensors index: read header length: %w", err)
	}

	headerBytes := make([]byte, headerLength)

	if _, err := io.ReadFull(reader, headerBytes); err != nil {
		return nil, fmt.Errorf("safetensors index: read header: %w", err)
	}

	raw := make(map[string]json.RawMessage)

	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return nil, fmt.Errorf("safetensors index: parse header: %w", err)
	}

	index := make(map[string]safetensorsMeta, len(raw))

	for name, rawMeta := range raw {
		if name == "__metadata__" {
			continue
		}

		meta := safetensorsMeta{}

		if err := json.Unmarshal(rawMeta, &meta); err != nil {
			return nil, fmt.Errorf("safetensors index: parse tensor %q: %w", name, err)
		}

		index[name] = meta
	}

	return index, nil
}

/*
Bind attaches checkpoint tensors to graph nodes using explicit weight maps and
prefix conventions.
*/
func (binder *Binder) Bind(
	graph *ast.Graph,
	index map[string]safetensorsMeta,
	weightMap map[string]string,
) error {
	if graph == nil {
		return fmt.Errorf("weights bind: graph is required")
	}

	for _, node := range graph.Nodes {
		tensorName := binder.resolveTensorName(node, weightMap)

		if tensorName == "" {
			tensorName = node.ID + ".weight"
		}

		meta, ok := index[tensorName]

		if !ok {
			if node.Weights != nil && node.Weights.TensorName != "" {
				return fmt.Errorf("weights bind: missing tensor %q for node %q", tensorName, node.ID)
			}

			continue
		}

		parsedDType, err := dtype.Parse(meta.DType)

		if err != nil {
			return fmt.Errorf("weights bind: parse dtype for tensor %q: %w", tensorName, err)
		}

		node.Weights = &ast.BoundWeight{
			TensorName: tensorName,
			Shape:      append([]int64(nil), meta.Shape...),
			DType:      parsedDType,
		}
	}

	return nil
}

func (binder *Binder) resolveTensorName(node *ast.GraphNode, weightMap map[string]string) string {
	if node.Weights != nil && node.Weights.TensorName != "" {
		return node.Weights.TensorName
	}

	if mapped, ok := weightMap[node.ID]; ok {
		return mapped
	}

	return ""
}
