package typer

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/ir"
)

func dimensionInt(dimension ir.Dimension, bindings ir.SymbolMap) (int64, error) {
	if !dimension.IsSymbolic() {
		return dimension.Static, nil
	}

	value, ok := bindings[dimension.Symbol]

	if !ok {
		return 0, fmt.Errorf("symbol %q is unbound", dimension.Symbol)
	}

	return value, nil
}

func bindSymbol(bindings ir.SymbolMap, symbol string, value int64) error {
	if bindings == nil || symbol == "" || value == 0 {
		return nil
	}

	existing, ok := bindings[symbol]

	if ok && existing != value {
		return fmt.Errorf("typer: symbol %q already bound to %d, got %d", symbol, existing, value)
	}

	bindings[symbol] = value

	return nil
}

func boundWeightDim(node *ast.GraphNode, index int) int64 {
	if node == nil || node.Weights == nil {
		return 0
	}

	if index < 0 || index >= len(node.Weights.Shape) {
		return 0
	}

	return node.Weights.Shape[index]
}

func configInt64(node *ast.GraphNode, key string) int64 {
	if node == nil || node.Attributes == nil {
		return 0
	}

	raw, ok := node.Attributes[key]

	if !ok {
		return 0
	}

	switch typed := raw.(type) {
	case int:
		return int64(typed)
	case int32:
		return int64(typed)
	case int64:
		return typed
	case uint:
		return int64(typed)
	case uint32:
		return int64(typed)
	case uint64:
		return int64(typed)
	case float32:
		return int64(typed)
	case float64:
		return int64(typed)
	default:
		return 0
	}
}
