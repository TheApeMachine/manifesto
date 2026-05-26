package typer

import (
	"fmt"

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

	if targetLayout != ir.LayoutUnspecified {
		result.Layout = targetLayout
	}

	return result, nil
}

func deriveReshapeOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	_ = bindings

	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: shape.reshape needs one input")
	}

	dims, _ := node.Attributes["shape"].(ir.ShapeSchema)
	result := inputs[0]

	if len(dims.Dimensions) > 0 {
		result.ShapeSchema = dims
	}

	return result, nil
}
