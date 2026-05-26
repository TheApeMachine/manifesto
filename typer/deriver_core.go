package typer

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
)

func deriveSameAsFirstInput(kind ir.SemanticKind) OutputDeriver {
	return func(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
		_ = bindings

		if len(inputs) == 0 {
			return ir.PortType{}, fmt.Errorf("typer: %q has no inputs", node.Op)
		}

		result := inputs[0]

		if kind != ir.SemanticGeneric {
			result.Kind = kind
		}

		return result, nil
	}
}

func deriveEmbeddingOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: embedding.token needs one input")
	}

	hiddenSize := configInt64(node, "d_model")

	if hiddenSize == 0 {
		hiddenSize = configInt64(node, "hidden_size")
	}

	if hiddenSize == 0 {
		hiddenSize = boundWeightDim(node, 1)
	}

	if hiddenSize == 0 {
		hiddenSize = bindings["D"]
	}

	tokenDim := inputs[0].ShapeSchema.Dimensions
	hiddenDim := ir.Dimension{Symbol: "D"}

	if hiddenSize != 0 {
		hiddenDim = ir.Dimension{Static: hiddenSize}
	}

	return ir.PortType{
		DType: dtype.Float32,
		ShapeSchema: ir.ShapeSchema{
			Dimensions: append(append([]ir.Dimension(nil), tokenDim...), hiddenDim),
		},
		Layout: ir.LayoutContiguous,
		Kind:   ir.SemanticHiddenState,
	}, nil
}

func deriveNormOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	result, err := deriveSameAsFirstInput(ir.SemanticHiddenState)(node, inputs, bindings)

	if err != nil {
		return ir.PortType{}, err
	}

	dimensions := result.ShapeSchema.Dimensions

	if len(dimensions) == 0 {
		return ir.PortType{}, fmt.Errorf("typer: %s input has rank 0", node.Op)
	}

	if err := bindNormSymbols(dimensions, bindings); err != nil {
		return ir.PortType{}, err
	}

	return result, nil
}

func deriveLinearOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	_ = bindings

	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: projection.linear needs one input")
	}

	leading := inputs[0].ShapeSchema.Dimensions

	if len(leading) == 0 {
		return ir.PortType{}, fmt.Errorf("typer: projection.linear input has rank 0")
	}

	outFeatures := configInt64(node, "out_features")

	if outFeatures == 0 {
		return ir.PortType{}, fmt.Errorf(
			"typer: projection.linear %q requires an out_features config attribute",
			node.ID,
		)
	}

	prefix := append([]ir.Dimension(nil), leading[:len(leading)-1]...)

	return ir.PortType{
		DType:       dtype.Float32,
		ShapeSchema: ir.ShapeSchema{Dimensions: append(prefix, ir.Dimension{Static: outFeatures})},
		Layout:      ir.LayoutContiguous,
		Kind:        ir.SemanticHiddenState,
	}, nil
}

func deriveMatmulOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	_ = node
	_ = bindings

	if len(inputs) < 2 {
		return ir.PortType{}, fmt.Errorf("typer: math.matmul needs two inputs")
	}

	leftDims := inputs[0].ShapeSchema.Dimensions
	rightDims := inputs[1].ShapeSchema.Dimensions

	if len(leftDims) < 2 || len(rightDims) < 2 {
		return ir.PortType{}, fmt.Errorf("typer: math.matmul requires rank-2 operands")
	}

	return ir.PortType{
		DType: inputs[0].DType,
		ShapeSchema: ir.ShapeSchema{Dimensions: []ir.Dimension{
			leftDims[len(leftDims)-2],
			rightDims[len(rightDims)-1],
		}},
		Layout: ir.LayoutContiguous,
		Kind:   ir.SemanticGeneric,
	}, nil
}

func bindNormSymbols(dimensions []ir.Dimension, bindings ir.SymbolMap) error {
	lastValue, err := dimensionInt(dimensions[len(dimensions)-1], bindings)

	if err == nil {
		if err := bindSymbol(bindings, "D", lastValue); err != nil {
			return err
		}
	}

	if len(dimensions) == 1 {
		return nil
	}

	leading := int64(1)

	for _, dimension := range dimensions[:len(dimensions)-1] {
		value, err := dimensionInt(dimension, bindings)

		if err != nil {
			return nil
		}

		leading *= value
	}

	return bindSymbol(bindings, "N", leading)
}
