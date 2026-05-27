package runtime

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/ast"
)

/*
runPagedPlan builds page write indices and refreshes the visible page
tables for one forward step.
*/
func (executor *Executor) runPagedPlan(
	ctx context.Context,
	step ast.Step,
	values map[string]any,
) error {
	_ = ctx

	pageSize := intFromConfig(step.Config, "page_size", 0)

	if pageSize <= 0 {
		return fmt.Errorf("state.paged_plan: page_size must be positive")
	}

	tokenRef, ok := step.In["token_ids"]

	if !ok {
		for _, ref := range step.In {
			if _, isTokens := tokenSequenceLength(values[ref]); isTokens {
				tokenRef = ref

				break
			}
		}
	}

	tokenValue, ok := values[tokenRef]

	if !ok {
		return fmt.Errorf("state.paged_plan: token_ids %q not found", tokenRef)
	}

	tokenIDs, ok := intVectorFromValue(tokenValue)

	if !ok || len(tokenIDs) == 0 {
		return fmt.Errorf("state.paged_plan: token batch is empty")
	}

	positionRef, ok := step.In["position_offset"]

	if !ok {
		return fmt.Errorf("state.paged_plan: position_offset input is required")
	}

	positionValue, err := executor.resolveValue(positionRef, values)

	if err != nil {
		return err
	}

	plan, err := BuildPagedPlan(PagedPlanRequest{
		PositionOffset: positionValue,
		TokenIDs:       tokenIDs,
		PageSize:       pageSize,
	})

	if err != nil {
		return fmt.Errorf("state.paged_plan: %w", err)
	}

	if executor.stateMemory == nil {
		return fmt.Errorf("state.paged_plan: state memory backend is required")
	}

	writePageIDs, err := UploadInt32Vector(executor.stateMemory, plan.WritePageIDs)

	if err != nil {
		return err
	}

	writeOffsets, err := UploadInt32Vector(executor.stateMemory, plan.WriteOffsets)

	if err != nil {
		return err
	}

	for _, ref := range step.Out {
		switch {
		case ref == "write_page_ids" || strings.HasSuffix(ref, ".write_page_ids"):
			if err := executor.setPagedPlanOutput(values, ref, writePageIDs); err != nil {
				return err
			}
		case ref == "write_offsets" || strings.HasSuffix(ref, ".write_offsets"):
			if err := executor.setPagedPlanOutput(values, ref, writeOffsets); err != nil {
				return err
			}
		case strings.HasSuffix(ref, "key_page_table"):
			if err := executor.setPageTableReference(ref, plan.PageTable); err != nil {
				return err
			}
		case strings.HasSuffix(ref, "value_page_table"):
			if err := executor.setPageTableReference(ref, plan.PageTable); err != nil {
				return err
			}
		default:
			if err := executor.setPagedPlanOutput(values, ref, writePageIDs); err != nil {
				return err
			}
		}
	}

	return nil
}

/*
runAdvancePosition increments the absolute KV cursor by the current
token batch length.
*/
func (executor *Executor) runAdvancePosition(
	ctx context.Context,
	step ast.Step,
	values map[string]any,
) error {
	_ = ctx

	if executor.state == nil {
		return fmt.Errorf("state.advance_position: state store is required")
	}

	positionRef, ok := step.In["position_offset"]

	if !ok {
		return fmt.Errorf("state.advance_position: position_offset input is required")
	}

	positionValue, err := executor.resolveValue(positionRef, values)

	if err != nil {
		return err
	}

	currentPosition, err := scalarInt32Value(positionValue)

	if err != nil {
		return fmt.Errorf("state.advance_position: %w", err)
	}

	tokenRef, ok := step.In["token_ids"]

	if !ok {
		return fmt.Errorf("state.advance_position: token_ids input is required")
	}

	tokenValue, ok := values[tokenRef]

	if !ok {
		return fmt.Errorf("state.advance_position: token_ids %q not found", tokenRef)
	}

	tokenIDs, ok := intVectorFromValue(tokenValue)

	if !ok {
		return fmt.Errorf("state.advance_position: token_ids must be []int")
	}

	nextPosition := currentPosition + int32(len(tokenIDs))
	nextTensor, err := UploadInt32Vector(executor.stateMemory, []int32{nextPosition})

	if err != nil {
		return err
	}

	for _, ref := range step.Out {
		if strings.HasPrefix(ref, "state.") {
			if err := executor.state.SetReference(ref, nextTensor); err != nil {
				return err
			}

			continue
		}

		setRuntimeValue(values, ref, nextTensor)
	}

	return nil
}

func (executor *Executor) setPagedPlanOutput(values map[string]any, ref string, value any) error {
	if strings.HasPrefix(ref, "state.") && executor.state != nil {
		return executor.state.SetReference(ref, value)
	}

	setRuntimeValue(values, ref, value)

	return nil
}

func (executor *Executor) setPageTableReference(reference string, pages []int32) error {
	if executor.state == nil {
		return fmt.Errorf("state.paged_plan: state store is required")
	}

	if !strings.HasPrefix(reference, "state.") {
		setRuntimeValue(map[string]any{}, reference, pages)

		return nil
	}

	name := reference[len("state."):]

	value, ok := executor.state.Get(name)

	if !ok {
		return fmt.Errorf("state.paged_plan: unknown state %q", name)
	}

	table, ok := value.(*PageTableState)

	if !ok {
		return fmt.Errorf("state.paged_plan: state %q is %T, expected *PageTableState", name, value)
	}

	table.Pages = append([]int32(nil), pages...)

	return executor.state.SetReference(reference, table)
}

func intVectorFromValue(value any) ([]int, bool) {
	switch typed := value.(type) {
	case []int:
		return typed, true
	case []int32:
		converted := make([]int, len(typed))

		for index, tokenID := range typed {
			converted[index] = int(tokenID)
		}

		return converted, true
	case int:
		return []int{typed}, true
	case int32:
		return []int{int(typed)}, true
	case int64:
		return []int{int(typed)}, true
	default:
		return nil, false
	}
}
