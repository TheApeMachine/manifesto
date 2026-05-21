package asset

import (
	"embed"
	"fmt"
	"io/fs"
	"path"
	"strings"

	"gopkg.in/yaml.v3"
)

//go:embed template/*
var embedded embed.FS

/*
OperationPort describes a single named input or output on a node.
*/
type OperationPort struct {
	Name        string `yaml:"name"        json:"name"`
	Type        string `yaml:"type"        json:"type"`
	Description string `yaml:"description" json:"description"`
}

/*
ConfigParam describes a configurable parameter for an operation node.
*/
type ConfigParam struct {
	Name        string `yaml:"name"        json:"name"`
	Type        string `yaml:"type"        json:"type"`
	Default     any    `yaml:"default"     json:"default"`
	Description string `yaml:"description" json:"description"`
}

/*
TopologyNode describes a single operation node inside a block's internal wiring.
*/
type TopologyNode struct {
	ID       string         `yaml:"id,omitempty"       json:"id,omitempty"`
	Op       string         `yaml:"op,omitempty"       json:"op,omitempty"`
	In       []string       `yaml:"in,omitempty"       json:"in,omitempty"`
	Out      []string       `yaml:"out,omitempty"      json:"out,omitempty"`
	Config   map[string]any `yaml:"config,omitempty"   json:"config,omitempty"`
	Repeat   int            `yaml:"repeat,omitempty"   json:"repeat,omitempty"`
	Index    string         `yaml:"index,omitempty"    json:"index,omitempty"`
	Template []TopologyNode `yaml:"template,omitempty" json:"template,omitempty"`
}

/*
Topology holds the internal wiring of a block.
*/
type Topology struct {
	Nodes []TopologyNode `yaml:"nodes" json:"nodes"`
}

/*
System wraps the block's topology definition.
*/
type System struct {
	Topology Topology `yaml:"topology" json:"topology"`
}

/*
Schema is the frontend-facing description of a single operation or optimizer node.
It is derived directly from the YAML manifest files under template/operation/ and
template/optimizer/.
*/
type Schema struct {
	Kind         string          `yaml:"kind"          json:"kind"`
	Category     string          `yaml:"category"      json:"category"`
	Op           string          `yaml:"op"            json:"op"`
	Name         string          `yaml:"name"          json:"name"`
	Label        string          `yaml:"label"         json:"label"`
	Description  string          `yaml:"description"   json:"description"`
	InitialWidth int             `yaml:"initial_width" json:"initial_width"`
	Inputs       []OperationPort `yaml:"inputs"        json:"inputs"`
	Outputs      []OperationPort `yaml:"outputs"       json:"outputs"`
	Config       []ConfigParam   `yaml:"config"        json:"config"`
	System       *System         `yaml:"system"        json:"system,omitempty"`
}

/*
ReadFile returns the raw bytes of an embedded template file relative to the
template root (e.g. "manifest/project.yml").
*/
func ReadFile(name string) ([]byte, error) {
	return embedded.ReadFile("template/" + name)
}

/*
TemplateFS returns a sub-fs rooted at the embedded template/ directory so
that callers (e.g. the manifest parser) can resolve include directives
relative to the same prefix that ReadFile uses. Paths inside the returned
fs.FS look identical to the names passed to ReadFile — e.g. "manifest/project.yml".
*/
func TemplateFS() fs.FS {
	sub, err := fs.Sub(embedded, "template")

	if err != nil {
		// fs.Sub only fails for invalid prefixes, which is a programmer
		// error in this package — surface it loudly rather than handing
		// back a half-broken FS.
		panic(fmt.Sprintf("asset: cannot sub template/ FS: %v", err))
	}

	return sub
}

/*
Walk returns every Schema found under the given sub-path of the embedded template
tree (e.g. "template/operation" or "template/optimizer"). The map key is the op
identifier (e.g. "activation.relu").
*/
func Walk(subPath string) (map[string]Schema, error) {
	schemas := make(map[string]Schema)

	err := fs.WalkDir(embedded, subPath, func(filePath string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil || entry.IsDir() || !strings.HasSuffix(filePath, ".yml") {
			return walkErr
		}

		data, readErr := embedded.ReadFile(filePath)

		if readErr != nil {
			return readErr
		}

		var raw map[string]any

		if parseErr := yaml.Unmarshal(data, &raw); parseErr != nil {
			return parseErr
		}

		if raw["kind"] == nil && raw["op"] == nil {
			return nil
		}

		var schema Schema

		if parseErr := yaml.Unmarshal(data, &schema); parseErr != nil {
			return parseErr
		}

		key := schema.Op

		if key == "" {
			key = strings.TrimSuffix(path.Base(filePath), ".yml")
		}

		schemas[key] = schema

		return nil
	})

	return schemas, err
}
