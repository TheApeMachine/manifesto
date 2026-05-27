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

func deriveTimestepEmbeddingOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	_ = bindings

	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: embedding.timestep needs one input")
	}

	embeddingDim := configInt64(node, "dim")

	if embeddingDim <= 0 {
		return ir.PortType{}, fmt.Errorf("typer: embedding.timestep %q requires a positive dim config", node.ID)
	}

	dimensions := append([]ir.Dimension(nil), inputs[0].ShapeSchema.Dimensions...)
	dimensions = append(dimensions, ir.Dimension{Static: embeddingDim})

	return ir.PortType{
		DType: dtype.Float32,
		ShapeSchema: ir.ShapeSchema{
			Dimensions: dimensions,
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

func deriveGatedResidualOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	if len(inputs) != 3 {
		return ir.PortType{}, fmt.Errorf("typer: math.gated_residual needs three inputs")
	}

	residual := inputs[0]
	branch := inputs[1]

	if len(residual.ShapeSchema.Dimensions) != len(branch.ShapeSchema.Dimensions) {
		return ir.PortType{}, fmt.Errorf("typer: math.gated_residual rank mismatch")
	}

	for index := range residual.ShapeSchema.Dimensions {
		if dimensionsMatch(residual.ShapeSchema.Dimensions[index], branch.ShapeSchema.Dimensions[index], bindings) {
			continue
		}

		return ir.PortType{}, fmt.Errorf("typer: math.gated_residual dim %d mismatch", index)
	}

	if err := validateGatedResidualModulation(node, residual, inputs[2], bindings); err != nil {
		return ir.PortType{}, err
	}

	result := residual
	result.Kind = ir.SemanticHiddenState

	return result, nil
}

func validateGatedResidualModulation(
	node *ast.GraphNode,
	residual ir.PortType,
	modulation ir.PortType,
	bindings ir.SymbolMap,
) error {
	residualDims := residual.ShapeSchema.Dimensions

	if len(residualDims) == 0 {
		return fmt.Errorf("typer: math.gated_residual residual rank 0")
	}

	modulationDims := modulation.ShapeSchema.Dimensions

	if len(modulationDims) == 0 {
		return fmt.Errorf("typer: math.gated_residual modulation rank 0")
	}

	lastDim, err := dimensionInt(residualDims[len(residualDims)-1], bindings)

	if err != nil {
		return fmt.Errorf("typer: math.gated_residual last dim: %w", err)
	}

	modulationCols, err := dimensionInt(modulationDims[len(modulationDims)-1], bindings)

	if err != nil {
		return fmt.Errorf("typer: math.gated_residual modulation cols: %w", err)
	}

	requiredCols := (configInt64(node, "set")*3 + 3) * lastDim

	if modulationCols < requiredCols {
		return fmt.Errorf(
			"typer: math.gated_residual modulation width %d is smaller than %d",
			modulationCols,
			requiredCols,
		)
	}

	return nil
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

func derivePageGatherOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	_ = node
	_ = bindings

	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: state.page_gather needs storage input")
	}

	storageDims := inputs[0].ShapeSchema.Dimensions

	if len(storageDims) < 2 {
		return ir.PortType{}, fmt.Errorf("typer: state.page_gather storage rank too low")
	}

	tail := storageDims[len(storageDims)-2:]

	return ir.PortType{
		DType: inputs[0].DType,
		ShapeSchema: ir.ShapeSchema{
			Dimensions: append([]ir.Dimension{{Symbol: "KV"}}, tail...),
		},
		Layout: ir.LayoutContiguous,
		Kind:   ir.SemanticHiddenState,
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

	if len(dimensions) != 2 {
		return nil
	}

	leading, err := dimensionInt(dimensions[0], bindings)

	if err != nil {
		return nil
	}

	return bindSymbol(bindings, "N", leading)
}
