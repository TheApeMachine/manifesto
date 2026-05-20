package parse

import (
	"fmt"
	"strings"

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

	includes := document.Includes

	if len(includes) == 0 {
		includes = document.Include
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
		Includes:   includes,
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
		step, err := parser.normalizeStep(rawStep)

		if err != nil {
			return nil, err
		}

		steps = append(steps, step)
	}

	return steps, nil
}

func (parser *Parser) normalizeStep(rawStep rawStep) (ast.Step, error) {
	step := ast.Step{
		ID:     rawStep.ID,
		Op:     rawStep.Op,
		Graph:  rawStep.Graph,
		Config: rawStep.Config,
		Loop:   rawStep.Loop,
	}

	if step.Op == "" && rawStep.Repeat != "" {
		step.Op = parser.repeatOp(rawStep)
	}

	if step.Config == nil {
		step.Config = make(map[string]any)
	}

	if err := parser.mergePortMaps(rawStep, &step); err != nil {
		return ast.Step{}, err
	}

	if rawStep.Text != "" {
		step.In["text"] = rawStep.Text
	}

	if rawStep.Tokenizer != "" {
		step.Config["tokenizer"] = rawStep.Tokenizer
	}

	if rawStep.Scheduler != "" {
		step.Config["scheduler"] = rawStep.Scheduler
	}

	if rawStep.Image != "" {
		step.In["image"] = rawStep.Image
	}

	if rawStep.Source != "" {
		if step.Loop == nil {
			step.Loop = &ast.Loop{}
		}

		step.Loop.Over = rawStep.Source
		step.Loop.As = rawStep.As
	}

	if rawStep.StepIndex != "" {
		step.In["step_index"] = rawStep.StepIndex
	}

	if rawStep.Latents != "" {
		step.In["latents"] = rawStep.Latents
	}

	if rawStep.Velocity != "" {
		step.In["velocity"] = rawStep.Velocity
	}

	if rawStep.Update != "" {
		step.Config["update"] = rawStep.Update
	}

	if rawStep.Target != "" {
		step.Config["target"] = rawStep.Target
	}

	if len(rawStep.Body) == 0 {
		return step, nil
	}

	body, bodyErr := parser.normalizeSteps(rawStep.Body)

	if bodyErr != nil {
		return ast.Step{}, bodyErr
	}

	step.Body = body

	return step, nil
}

func (parser *Parser) repeatOp(rawStep rawStep) string {
	if rawStep.Until != "" || strings.EqualFold(rawStep.Repeat, "until_eof") {
		return "control.loop_until_eof"
	}

	return "control.loop_count"
}

func (parser *Parser) mergePortMaps(rawStep rawStep, step *ast.Step) error {
	inputs, err := parser.normalizePorts(rawStep.In)

	if err != nil {
		return err
	}

	if inputs == nil {
		inputs = make(map[string]string)
	}

	for key, value := range rawStep.Inputs {
		inputs[key] = value
	}

	step.In = inputs

	outputs, err := parser.normalizePorts(rawStep.Out)

	if err != nil {
		return err
	}

	if outputs == nil {
		outputs = make(map[string]string)
	}

	for key, value := range rawStep.Outputs {
		outputs[key] = value
	}

	step.Out = outputs

	if step.Loop == nil && rawStep.Repeat != "" && rawStep.Op == "" {
		step.Loop = &ast.Loop{Repeat: rawStep.Repeat}
	}

	return nil
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
