package typer

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/ir"
)

func deriveSwiGLUOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	if len(inputs) == 2 {
		return deriveSameAsFirstInput(ir.SemanticHiddenState)(node, inputs, bindings)
	}

	if len(inputs) != 1 {
		return ir.PortType{}, fmt.Errorf("typer: activation.swiglu needs one or two inputs")
	}

	dimensions := inputs[0].ShapeSchema.Dimensions

	if len(dimensions) == 0 {
		return ir.PortType{}, fmt.Errorf("typer: activation.swiglu input has rank 0")
	}

	lastValue, err := dimensionInt(dimensions[len(dimensions)-1], bindings)

	if err != nil {
		return ir.PortType{}, fmt.Errorf("typer: activation.swiglu last dim: %w", err)
	}

	if lastValue%2 != 0 {
		return ir.PortType{}, fmt.Errorf("typer: activation.swiglu last dim %d is not even", lastValue)
	}

	prefix := append([]ir.Dimension(nil), dimensions[:len(dimensions)-1]...)
	result := inputs[0]
	result.ShapeSchema = ir.ShapeSchema{
		Dimensions: append(prefix, ir.Dimension{Static: lastValue / 2}),
	}
	result.Kind = ir.SemanticHiddenState

	return result, nil
}

func deriveViewAsHeadsOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: shape.view_as_heads needs one input")
	}

	numHeads := configInt64(node, "num_heads")

	if numHeads <= 0 {
		return ir.PortType{}, fmt.Errorf("typer: shape.view_as_heads requires positive num_heads")
	}

	dimensions := inputs[0].ShapeSchema.Dimensions

	if len(dimensions) == 0 {
		return ir.PortType{}, fmt.Errorf("typer: shape.view_as_heads input has rank 0")
	}

	lastValue, err := dimensionInt(dimensions[len(dimensions)-1], bindings)

	if err != nil {
		return ir.PortType{}, fmt.Errorf("typer: shape.view_as_heads last dim: %w", err)
	}

	if lastValue%numHeads != 0 {
		return ir.PortType{}, fmt.Errorf(
			"typer: shape.view_as_heads last dim %d is not divisible by %d heads",
			lastValue, numHeads,
		)
	}

	prefix := append([]ir.Dimension(nil), dimensions[:len(dimensions)-1]...)
	result := inputs[0]
	result.ShapeSchema = ir.ShapeSchema{
		Dimensions: append(
			prefix,
			ir.Dimension{Static: numHeads},
			ir.Dimension{Static: lastValue / numHeads},
		),
	}

	return result, nil
}

func deriveMergeHeadsOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	_ = node

	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: shape.merge_heads needs one input")
	}

	dimensions := inputs[0].ShapeSchema.Dimensions

	if len(dimensions) < 2 {
		return ir.PortType{}, fmt.Errorf("typer: shape.merge_heads input rank must be >= 2")
	}

	numHeads, err := dimensionInt(dimensions[len(dimensions)-2], bindings)

	if err != nil {
		return ir.PortType{}, fmt.Errorf("typer: shape.merge_heads num_heads: %w", err)
	}

	headDim, err := dimensionInt(dimensions[len(dimensions)-1], bindings)

	if err != nil {
		return ir.PortType{}, fmt.Errorf("typer: shape.merge_heads head_dim: %w", err)
	}

	prefix := append([]ir.Dimension(nil), dimensions[:len(dimensions)-2]...)
	result := inputs[0]
	result.ShapeSchema = ir.ShapeSchema{
		Dimensions: append(prefix, ir.Dimension{Static: numHeads * headDim}),
	}

	return result, nil
}

func deriveConcatOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	if len(inputs) != 2 {
		return ir.PortType{}, fmt.Errorf("typer: shape.concat needs two inputs")
	}

	left := inputs[0]
	right := inputs[1]
	leftDimensions := left.ShapeSchema.Dimensions
	rightDimensions := right.ShapeSchema.Dimensions

	if len(leftDimensions) != len(rightDimensions) {
		return ir.PortType{}, fmt.Errorf("typer: shape.concat rank mismatch")
	}

	axis, err := canonicalAxis(configInt64(node, "dim"), len(leftDimensions))

	if err != nil {
		return ir.PortType{}, fmt.Errorf("typer: shape.concat dim: %w", err)
	}

	outputDimensions := append([]ir.Dimension(nil), leftDimensions...)

	for index := range leftDimensions {
		if index == axis {
			merged, err := concatAxisDimension(leftDimensions[index], rightDimensions[index], bindings)

			if err != nil {
				return ir.PortType{}, fmt.Errorf("typer: shape.concat axis %d: %w", axis, err)
			}

			outputDimensions[index] = ir.Dimension{Static: merged}
			continue
		}

		if !dimensionsMatch(leftDimensions[index], rightDimensions[index], bindings) {
			return ir.PortType{}, fmt.Errorf("typer: shape.concat dim %d mismatch", index)
		}
	}

	result := left
	result.ShapeSchema = ir.ShapeSchema{Dimensions: outputDimensions}

	return result, nil
}

func deriveSliceOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: shape.slice needs one input")
	}

	dimensions := inputs[0].ShapeSchema.Dimensions
	axis, err := canonicalAxis(configInt64(node, "dim"), len(dimensions))

	if err != nil {
		return ir.PortType{}, fmt.Errorf("typer: shape.slice dim: %w", err)
	}

	dimSize, err := dimensionInt(dimensions[axis], bindings)

	if err != nil {
		return ir.PortType{}, fmt.Errorf("typer: shape.slice dim %d: %w", axis, err)
	}

	start := configInt64(node, "start")
	end := configInt64(node, "end")

	if end == 0 {
		end = dimSize
	}

	if start < 0 || end < start || end > dimSize {
		return ir.PortType{}, fmt.Errorf(
			"typer: shape.slice range [%d:%d) out of bounds for dim %d size %d",
			start,
			end,
			axis,
			dimSize,
		)
	}

	outputDimensions := append([]ir.Dimension(nil), dimensions...)
	outputDimensions[axis] = ir.Dimension{Static: end - start}

	result := inputs[0]
	result.ShapeSchema = ir.ShapeSchema{Dimensions: outputDimensions}

	return result, nil
}

func deriveUpsampleNearest2DOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: shape.upsample_nearest2d needs one input")
	}

	dimensions := inputs[0].ShapeSchema.Dimensions

	if len(dimensions) != 4 {
		return ir.PortType{}, fmt.Errorf("typer: shape.upsample_nearest2d input rank must be 4")
	}

	scaleH := configInt64(node, "scale_h")
	scaleW := configInt64(node, "scale_w")

	if scaleH <= 0 || scaleW <= 0 {
		return ir.PortType{}, fmt.Errorf("typer: shape.upsample_nearest2d requires positive scales")
	}

	inputHeight, err := dimensionInt(dimensions[2], bindings)

	if err != nil {
		return ir.PortType{}, fmt.Errorf("typer: shape.upsample_nearest2d height: %w", err)
	}

	inputWidth, err := dimensionInt(dimensions[3], bindings)

	if err != nil {
		return ir.PortType{}, fmt.Errorf("typer: shape.upsample_nearest2d width: %w", err)
	}

	outputDimensions := append([]ir.Dimension(nil), dimensions...)
	outputDimensions[2] = ir.Dimension{Static: inputHeight * scaleH}
	outputDimensions[3] = ir.Dimension{Static: inputWidth * scaleW}

	result := inputs[0]
	result.ShapeSchema = ir.ShapeSchema{Dimensions: outputDimensions}

	return result, nil
}

func canonicalAxis(axis int64, rank int) (int, error) {
	if rank == 0 {
		return 0, fmt.Errorf("rank 0 tensor")
	}

	if axis < 0 {
		axis += int64(rank)
	}

	if axis < 0 || axis >= int64(rank) {
		return 0, fmt.Errorf("%d out of range for rank %d", axis, rank)
	}

	return int(axis), nil
}

func concatAxisDimension(left ir.Dimension, right ir.Dimension, bindings ir.SymbolMap) (int64, error) {
	leftValue, err := dimensionInt(left, bindings)

	if err != nil {
		return 0, err
	}

	rightValue, err := dimensionInt(right, bindings)

	if err != nil {
		return 0, err
	}

	return leftValue + rightValue, nil
}

func dimensionsMatch(left ir.Dimension, right ir.Dimension, bindings ir.SymbolMap) bool {
	leftValue, leftErr := dimensionInt(left, bindings)
	rightValue, rightErr := dimensionInt(right, bindings)

	if leftErr == nil && rightErr == nil {
		return leftValue == rightValue
	}

	return left == right
}

func deriveLastTokenOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	_ = node
	_ = bindings

	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: shape.last_token needs one input")
	}

	dimensions := inputs[0].ShapeSchema.Dimensions

	if len(dimensions) < 2 {
		return ir.PortType{}, fmt.Errorf("typer: shape.last_token input rank must be >= 2")
	}

	if len(dimensions) == 2 {
		result := inputs[0]
		result.ShapeSchema = ir.ShapeSchema{
			Dimensions: append([]ir.Dimension{{Static: 1}}, dimensions[1:]...),
		}

		return result, nil
	}

	result := inputs[0]
	result.ShapeSchema = ir.ShapeSchema{
		Dimensions: append([]ir.Dimension{dimensions[0]}, dimensions[2:]...),
	}

	return result, nil
}
