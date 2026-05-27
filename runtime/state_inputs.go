package runtime

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

const (
	minTokenInt32 = -1 << 31
	maxTokenInt32 = 1<<31 - 1
)

/*
ResolveGraphInput normalizes graph.call boundary values for dispatch.
Paged state handles are expanded to their resident storage tensors; page
tables are materialized as int32 tensors covering the active prefix.
*/
func ResolveGraphInput(value any, memory tensor.Backend) (any, error) {
	switch typed := value.(type) {
	case *PagedTensorState:
		return resolvePagedTensorInput(typed)
	case *PageTableState:
		return resolvePageTableInput(typed, memory)
	case tensor.Tensor:
		return typed, nil
	default:
		return value, nil
	}
}

/*
ResolveGraphInputForGraph normalizes a graph.call boundary value using the
compiled graph's typed boundary contract.
*/
func ResolveGraphInputForGraph(
	graph *ast.Graph,
	inputName string,
	value any,
	memory tensor.Backend,
) (any, error) {
	resolved, err := ResolveGraphInput(value, memory)

	if err != nil {
		return nil, err
	}

	if !graphInputExpectsBatchedTokens(graph, inputName) {
		if !graphInputExpectsFloatVector(graph, inputName) {
			return resolved, nil
		}

		return float32TensorVector(resolved, memory)
	}

	return batchedTokenTensor(resolved, memory)
}

func graphInputExpectsBatchedTokens(graph *ast.Graph, inputName string) bool {
	portType, ok := graphBoundaryInputType(graph, inputName)

	if !ok {
		return false
	}

	return portType.Kind == ir.SemanticTokenIndex &&
		len(portType.ShapeSchema.Dimensions) == 2
}

func graphInputExpectsFloatVector(graph *ast.Graph, inputName string) bool {
	portType, ok := graphBoundaryInputType(graph, inputName)

	if !ok {
		return false
	}

	return portType.DType == dtype.Float32 &&
		len(portType.ShapeSchema.Dimensions) == 1
}

func graphBoundaryInputType(graph *ast.Graph, inputName string) (ir.PortType, bool) {
	if graph == nil {
		return ir.PortType{}, false
	}

	for _, node := range graph.Nodes {
		for slotIndex, producerName := range node.Inputs {
			if producerName != inputName || slotIndex >= len(node.InputTypes) {
				continue
			}

			return node.InputTypes[slotIndex], true
		}
	}

	return ir.PortType{}, false
}

func batchedTokenTensor(value any, memory tensor.Backend) (tensor.Tensor, error) {
	if tensorValue, ok := value.(tensor.Tensor); ok {
		if tensorValue.DType() != dtype.Int32 {
			return nil, fmt.Errorf("batched token tensor dtype is %s, expected int32", tensorValue.DType())
		}

		if tensorValue.Shape().Rank() != 2 {
			return nil, fmt.Errorf("batched token tensor shape is %v, expected rank 2", tensorValue.Shape().Dims())
		}

		return tensorValue, nil
	}

	tokenIDs, err := int32TokenVector(value)

	if err != nil {
		return nil, err
	}

	if memory == nil {
		return nil, fmt.Errorf("tensor backend is required to resolve batched tokens")
	}

	buffer := make([]byte, len(tokenIDs)*4)

	for index, tokenID := range tokenIDs {
		binary.LittleEndian.PutUint32(buffer[index*4:], uint32(tokenID))
	}

	shape, err := tensor.NewShape([]int{1, len(tokenIDs)})

	if err != nil {
		return nil, err
	}

	return memory.Upload(shape, dtype.Int32, buffer)
}

func float32TensorVector(value any, memory tensor.Backend) (tensor.Tensor, error) {
	if tensorValue, ok := value.(tensor.Tensor); ok {
		if !tensorValue.DType().IsFloat() {
			return nil, fmt.Errorf("float vector tensor dtype is %s, expected float", tensorValue.DType())
		}

		if tensorValue.Shape().Rank() != 1 {
			return nil, fmt.Errorf("float vector tensor shape is %v, expected rank 1", tensorValue.Shape().Dims())
		}

		return tensorValue, nil
	}

	values, err := float32VectorValue(value)

	if err != nil {
		return nil, err
	}

	if memory == nil {
		return nil, fmt.Errorf("tensor backend is required to resolve float vector")
	}

	buffer := make([]byte, len(values)*4)

	for index, value := range values {
		binary.LittleEndian.PutUint32(buffer[index*4:], math.Float32bits(value))
	}

	shape, err := tensor.NewShape([]int{len(values)})

	if err != nil {
		return nil, err
	}

	return memory.Upload(shape, dtype.Float32, buffer)
}

func float32VectorValue(value any) ([]float32, error) {
	switch typed := value.(type) {
	case []float32:
		return append([]float32(nil), typed...), nil
	case []float64:
		values := make([]float32, len(typed))

		for index, element := range typed {
			values[index] = float32(element)
		}

		return values, nil
	case float32:
		return []float32{typed}, nil
	case float64:
		return []float32{float32(typed)}, nil
	case int:
		return []float32{float32(typed)}, nil
	case int64:
		return []float32{float32(typed)}, nil
	default:
		return nil, fmt.Errorf("float vector input has unsupported type %T", value)
	}
}

func int32TokenVector(value any) ([]int32, error) {
	switch typed := value.(type) {
	case []int:
		tokens := make([]int32, len(typed))

		for index, tokenID := range typed {
			if tokenID < minTokenInt32 || tokenID > maxTokenInt32 {
				return nil, fmt.Errorf("token value %d overflows int32", tokenID)
			}

			tokens[index] = int32(tokenID)
		}

		return tokens, nil
	case []int32:
		return append([]int32(nil), typed...), nil
	case []int64:
		tokens := make([]int32, len(typed))

		for index, tokenID := range typed {
			if tokenID < minTokenInt32 || tokenID > maxTokenInt32 {
				return nil, fmt.Errorf("token value %d overflows int32", tokenID)
			}

			tokens[index] = int32(tokenID)
		}

		return tokens, nil
	case int:
		if typed < minTokenInt32 || typed > maxTokenInt32 {
			return nil, fmt.Errorf("token value %d overflows int32", typed)
		}

		return []int32{int32(typed)}, nil
	case int32:
		return []int32{typed}, nil
	case int64:
		if typed < minTokenInt32 || typed > maxTokenInt32 {
			return nil, fmt.Errorf("token value %d overflows int32", typed)
		}

		return []int32{int32(typed)}, nil
	default:
		return nil, fmt.Errorf("token input has unsupported type %T", value)
	}
}

/*
CommitGraphStateOutput writes graph outputs back into state slots. Tensor
outputs replace resident storage on paged/page-table handles instead of
replacing the whole handle.
*/
func CommitGraphStateOutput(reference string, previous any, output any) (any, error) {
	switch typedPrevious := previous.(type) {
	case *PagedTensorState:
		if updated, ok := output.(*PagedTensorState); ok {
			if updated != nil && updated.Storage != nil {
				typedPrevious.Storage = updated.Storage
			}

			return typedPrevious, nil
		}

		storage, err := asTensor(output)

		if err != nil {
			return nil, err
		}

		typedPrevious.Storage = storage

		return typedPrevious, nil
	case *PageTableState:
		if updated, ok := output.(*PageTableState); ok {
			if updated != nil && updated.Storage != nil {
				typedPrevious.Storage = updated.Storage
			}

			if updated != nil && len(updated.Pages) > 0 {
				typedPrevious.Pages = append([]int32(nil), updated.Pages...)
			}

			return typedPrevious, nil
		}

		storage, err := asTensor(output)

		if err != nil {
			return nil, err
		}

		typedPrevious.Storage = storage

		return typedPrevious, nil
	default:
		return output, nil
	}
}

func resolvePagedTensorInput(paged *PagedTensorState) (any, error) {
	if paged == nil {
		return nil, fmt.Errorf("paged tensor state is nil")
	}

	storage, ok := paged.Storage.(tensor.Tensor)

	if !ok || storage == nil {
		return nil, fmt.Errorf("paged tensor storage is not materialized")
	}

	return storage, nil
}

func resolvePageTableInput(table *PageTableState, memory tensor.Backend) (any, error) {
	if table == nil {
		return nil, fmt.Errorf("page table state is nil")
	}

	activeLength := len(table.Pages)

	if activeLength == 0 {
		shape, err := tensor.NewShape([]int{0})

		if err != nil {
			return nil, err
		}

		return memory.Upload(shape, dtype.Int32, nil)
	}

	if memory == nil {
		return nil, fmt.Errorf("tensor backend is required to resolve page table")
	}

	buffer := make([]byte, activeLength*4)

	for index, pageID := range table.Pages {
		binary.LittleEndian.PutUint32(buffer[index*4:], uint32(pageID))
	}

	shape, err := tensor.NewShape([]int{activeLength})

	if err != nil {
		return nil, err
	}

	return memory.Upload(shape, dtype.Int32, buffer)
}

func asTensor(value any) (tensor.Tensor, error) {
	tensorValue, ok := value.(tensor.Tensor)

	if !ok {
		return nil, fmt.Errorf("expected tensor.Tensor, got %T", value)
	}

	return tensorValue, nil
}
