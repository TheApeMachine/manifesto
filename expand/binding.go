package expand

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
)

func (expander *Recipe) bindConfig(bindings map[string]ast.Binding, config map[string]any) (map[string]any, error) {
	values := make(map[string]any, len(bindings))

	for name, binding := range bindings {
		value, err := expander.evalBinding(binding, config)

		if err != nil {
			return nil, fmt.Errorf("expand config binding %q: %w", name, err)
		}

		values[name] = value
	}

	return values, nil
}

func (expander *Recipe) evalBinding(binding ast.Binding, config map[string]any) (any, error) {
	if binding.Literal != nil {
		return binding.Literal, nil
	}

	if binding.Config != "" {
		value, ok := config[binding.Config]

		if !ok {
			return nil, fmt.Errorf("missing config field %q", binding.Config)
		}

		return value, nil
	}

	if len(binding.Product) > 0 {
		return expander.evalProduct(binding.Product, config)
	}

	if len(binding.Sum) > 0 {
		return expander.evalSum(binding.Sum, config)
	}

	return nil, fmt.Errorf("empty binding")
}

func (expander *Recipe) evalProduct(terms []ast.Binding, config map[string]any) (int64, error) {
	product := int64(1)

	for _, term := range terms {
		value, err := expander.evalBinding(term, config)

		if err != nil {
			return 0, err
		}

		factor, err := dtype.Int64Value(value)

		if err != nil {
			return 0, err
		}

		product *= factor
	}

	return product, nil
}

func (expander *Recipe) evalSum(terms []ast.Binding, config map[string]any) (int64, error) {
	sum := int64(0)

	for _, term := range terms {
		value, err := expander.evalBinding(term, config)

		if err != nil {
			return 0, err
		}

		termValue, err := dtype.Int64Value(value)

		if err != nil {
			return 0, err
		}

		sum += termValue
	}

	return sum, nil
}
