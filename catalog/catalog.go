package catalog

import (
	"fmt"
	"io/fs"
	"path"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/theapemachine/manifesto/ast"
)

const registryPath = "model/architecture/registry.yml"

/*
Catalog resolves recipes, blocks, and the architecture registry from an fs.FS.
Hosts mount embedded templates or an on-disk tree at the FS root.
*/
type Catalog interface {
	Registry() (map[string]ast.RegistryEntry, error)
	LoadRecipe(name string) (*ast.Recipe, error)
	LoadBlock(name string) (*ast.Topology, error)
	ReadRaw(filename string) ([]byte, error)
}

/*
FS implements Catalog over any fs.FS.
Hosts mount embedded templates or an on-disk tree at the FS root.
*/
type FS struct {
	files fs.FS
}

/*
NewFS constructs a catalog over the supplied filesystem.
*/
func NewFS(files fs.FS) *FS {
	return &FS{files: files}
}

func (catalog *FS) Registry() (map[string]ast.RegistryEntry, error) {
	raw, err := fs.ReadFile(catalog.files, registryPath)

	if err != nil {
		return nil, fmt.Errorf("catalog registry: %w", err)
	}

	document := struct {
		Architectures map[string]ast.RegistryEntry `yaml:"architectures"`
	}{}

	if err := yaml.Unmarshal(raw, &document); err != nil {
		return nil, fmt.Errorf("catalog registry parse: %w", err)
	}

	return document.Architectures, nil
}

func (catalog *FS) LoadRecipe(name string) (*ast.Recipe, error) {
	filename := recipePath(name)
	raw, err := fs.ReadFile(catalog.files, filename)

	if err == nil {
		recipe := &ast.Recipe{Name: name}

		if parseErr := yaml.Unmarshal(raw, recipe); parseErr != nil {
			return nil, fmt.Errorf("catalog recipe parse %q: %w", name, parseErr)
		}

		return recipe, nil
	}

	topology, blockErr := catalog.LoadBlock(name)

	if blockErr != nil {
		return nil, fmt.Errorf("catalog recipe %q: %w", name, blockErr)
	}

	return &ast.Recipe{
		Name:     name,
		Topology: topology,
	}, nil
}

func (catalog *FS) LoadBlock(name string) (*ast.Topology, error) {
	raw, err := catalog.readBlockBytes(name)

	if err != nil {
		return nil, err
	}

	topology := &ast.Topology{}
	topologyErr := yaml.Unmarshal(raw, topology)

	if topologyErr == nil && (len(topology.Inputs) > 0 || len(topology.Nodes) > 0) {
		return topology, nil
	}

	document := &blockDocument{}

	if err := yaml.Unmarshal(raw, document); err != nil {
		return nil, fmt.Errorf("catalog block document parse %q: %w", name, err)
	}

	if document.System.Topology != nil {
		return document.System.Topology, nil
	}

	if topologyErr != nil {
		return nil, fmt.Errorf("catalog block parse %q: %w", name, topologyErr)
	}

	return topology, nil
}

type blockDocument struct {
	System struct {
		Topology *ast.Topology `yaml:"topology"`
	} `yaml:"system"`
}

func (catalog *FS) readBlockBytes(name string) ([]byte, error) {
	candidates := []string{
		blockPath(name),
		architecturePath(name),
		modelTemplatePath(name),
	}

	var lastErr error

	for _, filename := range candidates {
		raw, err := fs.ReadFile(catalog.files, filename)

		if err == nil {
			return raw, nil
		}

		lastErr = err
	}

	return nil, fmt.Errorf("catalog block %q: %w", name, lastErr)
}

func (catalog *FS) ReadRaw(filename string) ([]byte, error) {
	raw, err := fs.ReadFile(catalog.files, filename)

	if err != nil {
		return nil, fmt.Errorf("catalog read %q: %w", filename, err)
	}

	return raw, nil
}

func recipePath(name string) string {
	return path.Join("model", "recipe", normalizeName(name)+".yml")
}

func blockPath(name string) string {
	return path.Join("model", "block", normalizeName(name)+".yml")
}

func architecturePath(name string) string {
	if !strings.HasPrefix(name, "model.architecture.") {
		return ""
	}

	suffix := strings.TrimPrefix(name, "model.architecture.")

	return path.Join("model", "architecture", suffix+".yml")
}

func modelTemplatePath(name string) string {
	if !strings.HasPrefix(name, "model.") {
		return ""
	}

	suffix := strings.TrimPrefix(name, "model.")

	return path.Join("model", normalizeName(suffix)+".yml")
}

func normalizeName(name string) string {
	return strings.NewReplacer(".", "/", "_", "/").Replace(name)
}
