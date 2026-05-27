package parse

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/ast"
)

const variablePrefix = "$variables."

func (parser *Parser) applyVariables(program *ast.Program) error {
	if len(program.Variables) == 0 {
		return nil
	}

	for stateIndex := range program.State {
		if err := parser.applyVariablesToState(&program.State[stateIndex], program.Variables); err != nil {
			return err
		}
	}

	return parser.applyVariablesToSteps(program.Steps, program.Variables)
}

func (parser *Parser) applyVariablesToState(
	state *ast.StateDeclaration,
	variables map[string]any,
) error {
	for shapeIndex, dimension := range state.Shape {
		resolved, err := parser.resolveVariableValue(dimension, variables)

		if err != nil {
			return err
		}

		state.Shape[shapeIndex] = resolved
	}

	seed, err := parser.resolveVariableValue(state.Seed, variables)

	if err != nil {
		return err
	}

	state.Seed = seed

	config, err := parser.resolveVariableMap(state.Config, variables)

	if err != nil {
		return err
	}

	state.Config = config

	return nil
}

func (parser *Parser) applyVariablesToSteps(
	steps []ast.Step,
	variables map[string]any,
) error {
	for stepIndex := range steps {
		config, err := parser.resolveVariableMap(steps[stepIndex].Config, variables)

		if err != nil {
			return err
		}

		steps[stepIndex].Config = config

		if steps[stepIndex].Loop != nil {
			if err := parser.applyVariablesToLoop(steps[stepIndex].Loop, variables); err != nil {
				return err
			}
		}

		if err := parser.applyVariablesToSteps(steps[stepIndex].Body, variables); err != nil {
			return err
		}
	}

	return nil
}

func (parser *Parser) applyVariablesToLoop(loop *ast.Loop, variables map[string]any) error {
	var err error

	if loop.Repeat, err = parser.resolveVariableString(loop.Repeat, variables); err != nil {
		return err
	}

	if loop.Over, err = parser.resolveVariableString(loop.Over, variables); err != nil {
		return err
	}

	if loop.As, err = parser.resolveVariableString(loop.As, variables); err != nil {
		return err
	}

	if loop.Until, err = parser.resolveVariableString(loop.Until, variables); err != nil {
		return err
	}

	return nil
}

func (parser *Parser) resolveVariableMap(
	values map[string]any,
	variables map[string]any,
) (map[string]any, error) {
	if values == nil {
		return nil, nil
	}

	resolved := make(map[string]any, len(values))

	for key, value := range values {
		next, err := parser.resolveVariableValue(value, variables)

		if err != nil {
			return nil, err
		}

		resolved[key] = next
	}

	return resolved, nil
}

func (parser *Parser) resolveVariableSlice(
	values []any,
	variables map[string]any,
) ([]any, error) {
	resolved := make([]any, len(values))

	for valueIndex, value := range values {
		next, err := parser.resolveVariableValue(value, variables)

		if err != nil {
			return nil, err
		}

		resolved[valueIndex] = next
	}

	return resolved, nil
}

func (parser *Parser) resolveVariableValue(value any, variables map[string]any) (any, error) {
	switch typed := value.(type) {
	case string:
		if !strings.HasPrefix(typed, variablePrefix) {
			return typed, nil
		}

		name := strings.TrimPrefix(typed, variablePrefix)
		resolved, ok := variables[name]

		if !ok {
			return nil, fmt.Errorf("parse program variable %q is not declared", name)
		}

		return resolved, nil
	case []any:
		return parser.resolveVariableSlice(typed, variables)
	case map[string]any:
		return parser.resolveVariableMap(typed, variables)
	default:
		return value, nil
	}
}

func (parser *Parser) resolveVariableString(
	value string,
	variables map[string]any,
) (string, error) {
	if !strings.HasPrefix(value, variablePrefix) {
		return value, nil
	}

	name := strings.TrimPrefix(value, variablePrefix)
	resolved, ok := variables[name]

	if !ok {
		return "", fmt.Errorf("parse program variable %q is not declared", name)
	}

	return fmt.Sprint(resolved), nil
}
