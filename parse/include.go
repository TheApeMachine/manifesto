package parse

import (
	"fmt"
	"io/fs"
	"path"

	"gopkg.in/yaml.v3"

	"github.com/theapemachine/manifesto/ast"
)

/*
IncludeLoader resolves include directives against an fs.FS.
*/
type IncludeLoader struct {
	files fs.FS
}

func NewIncludeLoader(files fs.FS) *IncludeLoader {
	return &IncludeLoader{files: files}
}

/*
Topology loads a topology document, optionally via from_safetensors indirection.
*/
func (loader *IncludeLoader) Topology(filename string) (*ast.Topology, error) {
	raw, err := fs.ReadFile(loader.files, filename)

	if err != nil {
		return nil, fmt.Errorf("parse topology include %q: %w", filename, err)
	}

	document := topologyDocument{}

	if err := yaml.Unmarshal(raw, &document); err != nil {
		return nil, fmt.Errorf("parse topology yaml %q: %w", filename, err)
	}

	if document.Topology.FromSafeTensors != nil {
		return nil, fmt.Errorf("parse topology %q: from_safetensors requires repo resolution", filename)
	}

	if len(document.Topology.Nodes) > 0 {
		return &ast.Topology{
			Inputs: document.Topology.Inputs,
			Nodes:  document.Topology.Nodes,
		}, nil
	}

	if len(document.System.Topology.Nodes) > 0 {
		return &ast.Topology{
			Inputs: document.System.Topology.Inputs,
			Nodes:  document.System.Topology.Nodes,
		}, nil
	}

	return nil, fmt.Errorf("parse topology %q: no nodes found", filename)
}

type topologyDocument struct {
	System   systemTopology  `yaml:"system"`
	Topology topologySection `yaml:"topology"`
}

type topologySection struct {
	FromSafeTensors map[string]any `yaml:"from_safetensors,omitempty"`
	Inputs          []string       `yaml:"inputs,omitempty"`
	Nodes           []ast.Node     `yaml:"nodes,omitempty"`
}

type systemTopology struct {
	Topology topologySection `yaml:"topology"`
}

/*
ResolveIncludePath maps a dotted include name to a template path.
*/
func ResolveIncludePath(name string) string {
	return path.Join("model", path.Clean(name)+".yml")
}
