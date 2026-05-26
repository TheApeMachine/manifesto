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

	result := inputs[0]
	result.ShapeSchema = ir.ShapeSchema{
		Dimensions: append([]ir.Dimension{{Static: 1}}, dimensions[1:]...),
	}

	return result, nil
}
