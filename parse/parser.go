package parse

import (
	"fmt"

	"gopkg.in/yaml.v3"

	"github.com/theapemachine/manifesto/ast"
)

/*
Parser loads manifest YAML into typed AST values.
It exists so program and include parsing share one entry type.
*/
type Parser struct{}

/*
NewParser constructs a Parser.
*/
func NewParser() *Parser {
	return &Parser{}
}

/*
Program parses YAML bytes into a manifest program AST.
*/
func (parser *Parser) Program(data []byte) (*ast.Program, error) {
	document := ProgramDocument{}

	if err := yaml.Unmarshal(data, &document); err != nil {
		return nil, fmt.Errorf("parse program yaml: %w", err)
	}

	rawSteps := document.Main

	if len(rawSteps) == 0 {
		rawSteps = document.System.Runtime.Program
	}

	steps, err := parser.normalizeSteps(rawSteps)

	if err != nil {
		return nil, err
	}

	return &ast.Program{
		Name:       document.Name,
		Includes:   document.Includes,
		Variables:  document.Variables,
		State:      document.System.Runtime.State,
		Schedulers: document.System.Runtime.Schedulers,
		Graphs:     document.System.Runtime.Graphs,
		Steps:      steps,
	}, nil
}

func (parser *Parser) normalizeSteps(rawSteps []rawStep) ([]ast.Step, error) {
	steps := make([]ast.Step, 0, len(rawSteps))

	for _, rawStep := range rawSteps {
		step := ast.Step{
			ID:     rawStep.ID,
			Op:     rawStep.Op,
			Graph:  rawStep.Graph,
			Config: rawStep.Config,
			Loop:   rawStep.Loop,
		}

		inputs, err := parser.normalizePorts(rawStep.In)

		if err != nil {
			return nil, err
		}

		outputs, err := parser.normalizePorts(rawStep.Out)

		if err != nil {
			return nil, err
		}

		step.In = inputs
		step.Out = outputs

		if len(rawStep.Body) == 0 {
			steps = append(steps, step)

			continue
		}

		body, bodyErr := parser.normalizeSteps(rawStep.Body)

		if bodyErr != nil {
			return nil, bodyErr
		}

		step.Body = body
		steps = append(steps, step)
	}

	return steps, nil
}

func (parser *Parser) normalizePorts(node yaml.Node) (map[string]string, error) {
	if node.Kind == yaml.DocumentNode && len(node.Content) > 0 {
		node = *node.Content[0]
	}

	if node.Kind == 0 {
		return nil, nil
	}

	switch node.Kind {
	case yaml.ScalarNode:
		return map[string]string{"value": node.Value}, nil
	case yaml.SequenceNode:
		ports := make(map[string]string, len(node.Content))

		for portIndex, item := range node.Content {
			ports[fmt.Sprintf("arg%d", portIndex)] = item.Value
		}

		return ports, nil
	case yaml.MappingNode:
		ports := make(map[string]string)

		for portIndex := 0; portIndex+1 < len(node.Content); portIndex += 2 {
			key := node.Content[portIndex].Value
			value := node.Content[portIndex+1].Value
			ports[key] = value
		}

		return ports, nil
	default:
		return nil, fmt.Errorf("parse program ports: unsupported yaml kind %v", node.Kind)
	}
}
