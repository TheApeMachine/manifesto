package typer

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
)

func deriveCastOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	_ = bindings

	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: shape.cast needs one input")
	}

	target, _ := node.Attributes["dtype"].(dtype.DType)

	if target == dtype.Invalid {
		target = inputs[0].DType
	}

	result := inputs[0]
	result.DType = target

	return result, nil
}

func deriveTransposeOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	_ = bindings

	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: shape.transpose needs one input")
	}

	targetLayout, _ := node.Attributes["layout"].(ir.LayoutSchema)
	result := inputs[0]
	dimensions := append([]ir.Dimension(nil), result.ShapeSchema.Dimensions...)

	if len(dimensions) == 0 {
		return ir.PortType{}, fmt.Errorf("typer: shape.transpose input has rank 0")
	}

	firstAxis, err := canonicalAxis(configInt64(node, "dim0"), len(dimensions))

	if err != nil {
		return ir.PortType{}, fmt.Errorf("typer: shape.transpose dim0: %w", err)
	}

	secondAxis := int64(1)

	if _, exists := node.Attributes["dim1"]; exists {
		secondAxis = configInt64(node, "dim1")
	}

	secondAxisIndex, err := canonicalAxis(secondAxis, len(dimensions))

	if err != nil {
		return ir.PortType{}, fmt.Errorf("typer: shape.transpose dim1: %w", err)
	}

	dimensions[firstAxis], dimensions[secondAxisIndex] = dimensions[secondAxisIndex], dimensions[firstAxis]
	result.ShapeSchema = ir.ShapeSchema{Dimensions: dimensions}

	if targetLayout != ir.LayoutUnspecified {
		result.Layout = targetLayout
	}

	return result, nil
}

func deriveReshapeOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: shape.reshape needs one input")
	}

	dims, err := shapeSchemaFromAttribute(node.Attributes["shape"], bindings)

	if err != nil {
		return ir.PortType{}, fmt.Errorf("typer: shape.reshape shape: %w", err)
	}

	if len(dims.Dimensions) == 0 {
		return ir.PortType{}, fmt.Errorf("typer: shape.reshape requires a target shape")
	}

	if err := validateReshapeElementCount(inputs[0].ShapeSchema, dims, bindings); err != nil {
		return ir.PortType{}, fmt.Errorf("typer: shape.reshape %q: %w", node.ID, err)
	}

	result := inputs[0]
	result.ShapeSchema = dims

	return result, nil
}

func shapeSchemaFromAttribute(value any, bindings ir.SymbolMap) (ir.ShapeSchema, error) {
	switch typed := value.(type) {
	case ir.ShapeSchema:
		return typed, nil
	case []ir.Dimension:
		return ir.ShapeSchema{Dimensions: append([]ir.Dimension(nil), typed...)}, nil
	case []int:
		return shapeSchemaFromIntSlice(typed), nil
	case []int64:
		return shapeSchemaFromInt64Slice(typed), nil
	case []float64:
		return shapeSchemaFromFloat64Slice(typed), nil
	case []any:
		return shapeSchemaFromAnySlice(typed, bindings)
	default:
		return ir.ShapeSchema{}, fmt.Errorf("unsupported type %T", value)
	}
}

func shapeSchemaFromIntSlice(values []int) ir.ShapeSchema {
	dimensions := make([]ir.Dimension, len(values))

	for index, value := range values {
		dimensions[index] = ir.Dimension{Static: int64(value)}
	}

	return ir.ShapeSchema{Dimensions: dimensions}
}

func shapeSchemaFromInt64Slice(values []int64) ir.ShapeSchema {
	dimensions := make([]ir.Dimension, len(values))

	for index, value := range values {
		dimensions[index] = ir.Dimension{Static: value}
	}

	return ir.ShapeSchema{Dimensions: dimensions}
}

func shapeSchemaFromFloat64Slice(values []float64) ir.ShapeSchema {
	dimensions := make([]ir.Dimension, len(values))

	for index, value := range values {
		dimensions[index] = ir.Dimension{Static: int64(value)}
	}

	return ir.ShapeSchema{Dimensions: dimensions}
}

func shapeSchemaFromAnySlice(values []any, bindings ir.SymbolMap) (ir.ShapeSchema, error) {
	dimensions := make([]ir.Dimension, len(values))

	for index, value := range values {
		dimension, err := dimensionFromAttribute(value, bindings)

		if err != nil {
			return ir.ShapeSchema{}, fmt.Errorf("dim %d: %w", index, err)
		}

		dimensions[index] = dimension
	}

	return ir.ShapeSchema{Dimensions: dimensions}, nil
}

func dimensionFromAttribute(value any, bindings ir.SymbolMap) (ir.Dimension, error) {
	switch typed := value.(type) {
	case int:
		return ir.Dimension{Static: int64(typed)}, nil
	case int64:
		return ir.Dimension{Static: typed}, nil
	case float64:
		return ir.Dimension{Static: int64(typed)}, nil
	case string:
		return dimensionFromStringAttribute(typed, bindings)
	default:
		return ir.Dimension{}, fmt.Errorf("unsupported type %T", value)
	}
}

func dimensionFromStringAttribute(value string, bindings ir.SymbolMap) (ir.Dimension, error) {
	trimmed := strings.TrimSpace(value)

	if trimmed == "" {
		return ir.Dimension{}, fmt.Errorf("empty string")
	}

	if parsed, err := strconv.ParseInt(trimmed, 10, 64); err == nil {
		return ir.Dimension{Static: parsed}, nil
	}

	bound, exists := bindings[trimmed]

	if exists {
		return ir.Dimension{Static: bound}, nil
	}

	return ir.Dimension{Symbol: trimmed}, nil
}

func validateReshapeElementCount(input ir.ShapeSchema, output ir.ShapeSchema, bindings ir.SymbolMap) error {
	inputCount, inputKnown := shapeElementCount(input, bindings)
	outputCount, outputKnown := shapeElementCount(output, bindings)

	if !inputKnown || !outputKnown {
		return nil
	}

	if inputCount != outputCount {
		return fmt.Errorf("element count %d does not match target %d", inputCount, outputCount)
	}

	return nil
}

func shapeElementCount(shape ir.ShapeSchema, bindings ir.SymbolMap) (int64, bool) {
	count := int64(1)

	for _, dimension := range shape.Dimensions {
		value, err := dimensionInt(dimension, bindings)

		if err != nil {
			return 0, false
		}

		count *= value
	}

	return count, true
}
