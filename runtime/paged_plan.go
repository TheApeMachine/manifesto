package runtime

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
PagedPlanConfig configures one paged KV write-plan step.
*/
type PagedPlanConfig struct {
	PageSize int
}

/*
PagedPlanRequest supplies the live position and token batch for one step.
*/
type PagedPlanRequest struct {
	PositionOffset any
	TokenIDs       []int
	PageSize       int
}

/*
PagedPlanResult is the host-side write plan for one graph.call step.
*/
type PagedPlanResult struct {
	WritePageIDs []int32
	WriteOffsets []int32
	PageTable    []int32
	KVLength     int
}

/*
BuildPagedPlan derives page write indices and the visible page table for
one forward step. Positions are absolute in the growing KV sequence.
*/
func BuildPagedPlan(request PagedPlanRequest) (PagedPlanResult, error) {
	if request.PageSize <= 0 {
		return PagedPlanResult{}, fmt.Errorf("paged plan: page_size must be positive")
	}

	startPosition, err := scalarInt32Value(request.PositionOffset)

	if err != nil {
		return PagedPlanResult{}, fmt.Errorf("paged plan: position_offset: %w", err)
	}

	tokenCount := len(request.TokenIDs)

	if tokenCount == 0 {
		return PagedPlanResult{}, fmt.Errorf("paged plan: token batch is empty")
	}

	writePageIDs := make([]int32, tokenCount)
	writeOffsets := make([]int32, tokenCount)
	pageSet := make(map[int32]struct{})

	for tokenIndex := range tokenCount {
		absolutePosition := int(startPosition) + tokenIndex
		pageID := int32(absolutePosition / request.PageSize)
		pageOffset := int32(absolutePosition % request.PageSize)

		writePageIDs[tokenIndex] = pageID
		writeOffsets[tokenIndex] = pageOffset
		pageSet[pageID] = struct{}{}
	}

	kvLength := int(startPosition) + tokenCount
	finalPageID := int32((kvLength - 1) / request.PageSize)
	pageTable := make([]int32, 0, finalPageID+1)

	for pageID := int32(0); pageID <= finalPageID; pageID++ {
		pageTable = append(pageTable, pageID)
	}

	return PagedPlanResult{
		WritePageIDs: writePageIDs,
		WriteOffsets: writeOffsets,
		PageTable:    pageTable,
		KVLength:     kvLength,
	}, nil
}

/*
UploadInt32Vector materializes one int32 vector on the state memory backend.
*/
func UploadInt32Vector(memory tensor.Backend, values []int32) (tensor.Tensor, error) {
	if memory == nil {
		return nil, fmt.Errorf("upload int32 vector: tensor backend is required")
	}

	buffer := make([]byte, len(values)*4)

	for index, value := range values {
		buffer[index*4] = byte(value)
		buffer[index*4+1] = byte(value >> 8)
		buffer[index*4+2] = byte(value >> 16)
		buffer[index*4+3] = byte(value >> 24)
	}

	shape, err := tensor.NewShape([]int{len(values)})

	if err != nil {
		return nil, err
	}

	return memory.Upload(shape, dtype.Int32, buffer)
}

func scalarInt32Value(value any) (int32, error) {
	switch typed := value.(type) {
	case int:
		return int32(typed), nil
	case int32:
		return typed, nil
	case int64:
		return int32(typed), nil
	case []int:
		if len(typed) != 1 {
			return 0, fmt.Errorf("expected scalar []int, got len %d", len(typed))
		}

		return int32(typed[0]), nil
	case []int32:
		if len(typed) != 1 {
			return 0, fmt.Errorf("expected scalar []int32, got len %d", len(typed))
		}

		return typed[0], nil
	case []float32:
		if len(typed) != 1 {
			return 0, fmt.Errorf("expected scalar []float32, got len %d", len(typed))
		}

		return int32(typed[0]), nil
	case tensor.Tensor:
		values, err := typed.Int32Native()

		if err != nil {
			return 0, err
		}

		if len(values) != 1 {
			return 0, fmt.Errorf("expected scalar tensor, got len %d", len(values))
		}

		return values[0], nil
	default:
		return 0, fmt.Errorf("unsupported scalar type %T", value)
	}
}
