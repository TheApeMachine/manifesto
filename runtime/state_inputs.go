package runtime

import (
	"encoding/binary"
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
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
