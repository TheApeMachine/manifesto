package parse

import (
	"gopkg.in/yaml.v3"

	"github.com/theapemachine/manifesto/ast"
)

/*
ProgramDocument is the raw YAML shape for a runtime program manifest.
*/
type ProgramDocument struct {
	Kind      string            `yaml:"kind"`
	Name      string            `yaml:"name"`
	Includes  map[string]string `yaml:"includes"`
	Variables map[string]any    `yaml:"variables"`
	Main      []rawStep         `yaml:"main"`
	System    programSystem     `yaml:"system"`
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
	ID     string         `yaml:"id"`
	Op     string         `yaml:"op"`
	In     yaml.Node      `yaml:"in"`
	Out    yaml.Node      `yaml:"out"`
	Graph  string         `yaml:"graph"`
	Config map[string]any `yaml:"config"`
	Loop   *ast.Loop      `yaml:"loop"`
	Body   []rawStep      `yaml:"steps"`
}
