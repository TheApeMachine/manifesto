package parse

import (
	"gopkg.in/yaml.v3"

	"github.com/theapemachine/manifesto/ast"
)

/*
ProgramDocument is the raw YAML shape for a runtime program manifest.
*/
type ProgramDocument struct {
	Kind       string                              `yaml:"kind"`
	Name       string                              `yaml:"name"`
	Category   string                              `yaml:"category"`
	Includes   map[string]any                      `yaml:"includes"`
	Include    map[string]any                      `yaml:"include"`
	Variables  map[string]any                      `yaml:"variables"`
	State      []ast.StateDeclaration              `yaml:"state"`
	Schedulers map[string]ast.SchedulerDeclaration `yaml:"schedulers"`
	Graphs     map[string]ast.GraphModule          `yaml:"graphs"`
	Main       []rawStep                           `yaml:"main"`
	System     programSystem                       `yaml:"system"`
}

type programSystem struct {
	Runtime programRuntime `yaml:"runtime"`
}

type programRuntime struct {
	Type       string                              `yaml:"type"`
	State      []ast.StateDeclaration              `yaml:"state"`
	Schedulers map[string]ast.SchedulerDeclaration `yaml:"schedulers"`
	Graphs     map[string]ast.GraphModule          `yaml:"graphs"`
	Program    []rawStep                           `yaml:"program"`
}

type rawStep struct {
	ID        string            `yaml:"id"`
	Op        string            `yaml:"op"`
	In        yaml.Node         `yaml:"in"`
	Out       yaml.Node         `yaml:"out"`
	Inputs    map[string]string `yaml:"inputs"`
	Outputs   map[string]string `yaml:"outputs"`
	Graph     string            `yaml:"graph"`
	Config    map[string]any    `yaml:"config"`
	Loop      *ast.Loop         `yaml:"loop"`
	Body      []rawStep         `yaml:"steps"`
	Repeat    string            `yaml:"repeat"`
	Until     string            `yaml:"until_eof,omitempty"`
	Source    string            `yaml:"source"`
	As        string            `yaml:"as"`
	Text      string            `yaml:"text"`
	Tokenizer string            `yaml:"tokenizer"`
	Scheduler string            `yaml:"scheduler"`
	Image     string            `yaml:"image"`
	Update    string            `yaml:"update"`
	Target    string            `yaml:"target"`
	StepIndex string            `yaml:"step_index"`
	Latents   string            `yaml:"latents"`
	Velocity  string            `yaml:"velocity"`
}
